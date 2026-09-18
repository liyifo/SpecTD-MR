from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


TGCT_ENCODER = "tgct"
SEQUENCE_ENCODER_CHOICES = (TGCT_ENCODER,)


@dataclass
class EncoderOutputs:
    final_hidden: torch.Tensor
    final_graph: torch.Tensor
    last_visit_repr: torch.Tensor
    visit_sequence: Optional[torch.Tensor] = None


class TGCTSequenceEncoder(nn.Module):
    """Two-layer Stage-II Visit-GRU over Stage-I visit embeddings."""

    def __init__(
            self,
            hidden_dim: int,
            visit_lookup: Dict[Tuple[int, int], int],
            dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.visit_lookup = visit_lookup
        self.visit_bias = nn.Parameter(torch.zeros(hidden_dim))
        self.input_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )

    def _stage1_visit_embedding(
            self,
            patient_index: int,
            order_index: int,
            visit_bank: torch.Tensor) -> torch.Tensor:
        idx = self.visit_lookup.get((patient_index, order_index))
        if idx is None or idx >= visit_bank.size(0):
            return self.visit_bias
        return visit_bank[idx]

    def forward(
            self,
            batch_inputs: Dict[str, torch.Tensor],
            node_bank: torch.Tensor,
            visit_bank: torch.Tensor) -> EncoderOutputs:
        device = node_bank.device
        visit_masks = batch_inputs["visit_node_masks"]
        visit_orders = batch_inputs["visit_orders"]
        visit_lengths = batch_inputs["visit_lengths"]
        patient_indices = batch_inputs["patient_index"]
        batch_size = patient_indices.size(0)
        max_visits = visit_masks.size(1)

        stage_visits = self.visit_bias.view(1, 1, -1).expand(
            batch_size, max_visits, -1,
        ).clone()
        visit_orders_list = visit_orders.cpu().tolist()
        patient_list = patient_indices.cpu().tolist()
        for batch_idx, patient_index in enumerate(patient_list):
            visit_count = int(visit_lengths[batch_idx].item())
            for visit_idx in range(visit_count):
                order_index = visit_orders_list[batch_idx][visit_idx]
                stage_visits[batch_idx, visit_idx] = self._stage1_visit_embedding(
                    patient_index, order_index, visit_bank,
                ).to(device)

        visit_sequence = self.input_dropout(stage_visits)
        packed = nn.utils.rnn.pack_padded_sequence(
            visit_sequence,
            visit_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, final_hidden = self.gru(packed)
        sequence_output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=max_visits,
        )
        batch_indices = torch.arange(batch_size, device=device)
        last_visit_repr = sequence_output[batch_indices, visit_lengths - 1]
        return EncoderOutputs(
            final_hidden=final_hidden[-1],
            final_graph=last_visit_repr,
            last_visit_repr=last_visit_repr,
            visit_sequence=sequence_output,
        )


class _BasePrescriptionDecoder(nn.Module):
    def __init__(self, hidden_dim: int, med_vocab_size: int, concat_graph: bool):
        super().__init__()
        input_dim = hidden_dim + (hidden_dim if concat_graph else 0)
        self.concat_graph = concat_graph
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, med_vocab_size),
        )

    def forward(
            self,
            hidden: torch.Tensor,
            graph_readout: torch.Tensor) -> torch.Tensor:
        features = (
            torch.cat([hidden, graph_readout], dim=-1)
            if self.concat_graph else hidden
        )
        return self.classifier(features)


class PrescriptionDecoder(nn.Module):
    """Four history experts plus a shared per-drug temporal GRU."""

    def __init__(
            self,
            hidden_dim: int,
            med_vocab_size: int,
            concat_graph: bool,
            dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.med_vocab_size = med_vocab_size
        self.concat_graph = concat_graph
        self.base_decoder = _BasePrescriptionDecoder(
            hidden_dim, med_vocab_size, concat_graph,
        )
        patient_dim = hidden_dim * (2 if concat_graph else 1)
        self.query_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Retained for exact checkpoint compatibility with the selected model.
        self.history_drug_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.drug_norm = nn.LayerNorm(hidden_dim)
        self.patient_projection = nn.Sequential(
            nn.LayerNorm(patient_dim),
            nn.Linear(patient_dim, hidden_dim),
            nn.GELU(),
        )
        self.scale_strength = nn.Parameter(torch.zeros(()))
        self.recency_log_rate = nn.Parameter(torch.zeros(()))

        def build_expert() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )

        self.history_channel_drug_projections = nn.ModuleList([
            build_expert() for _ in range(4)
        ])
        self.temporal_history_gru = nn.GRU(
            input_size=1,
            hidden_size=8,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )
        self.temporal_history_projection = nn.Sequential(
            nn.LayerNorm(8),
            nn.Linear(8, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.temporal_strength = nn.Parameter(torch.zeros(()))

    def _patient_features(
            self,
            hidden: torch.Tensor,
            graph_readout: torch.Tensor) -> torch.Tensor:
        return (
            torch.cat([hidden, graph_readout], dim=-1)
            if self.concat_graph else hidden
        )

    def aggregate_history(
            self,
            current_hidden: torch.Tensor,
            visit_sequence: torch.Tensor,
            visit_lengths: torch.Tensor,
            history_medications: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if visit_sequence.dim() != 3 or history_medications.dim() != 3:
            raise ValueError(
                "Visit states and historical medications must be rank-3 tensors.",
            )
        if visit_sequence.shape[:2] != history_medications.shape[:2]:
            raise ValueError(
                "Visit states and medications must share batch/visit dimensions.",
            )
        max_visits = visit_sequence.size(1)
        visit_index = torch.arange(
            max_visits, device=visit_sequence.device,
        ).unsqueeze(0)
        history_mask = visit_index < (
            visit_lengths - 1
        ).clamp_min(0).unsqueeze(1)
        query = self.query_projection(current_hidden)
        keys = self.key_projection(visit_sequence)
        scores = torch.einsum(
            "bvh,bh->bv", keys, query,
        ) / math.sqrt(self.hidden_dim)
        scores = scores.masked_fill(
            ~history_mask, torch.finfo(scores.dtype).min,
        )
        weights = torch.softmax(scores, dim=-1)
        weights = weights * history_mask.to(scores.dtype)
        weights = weights / weights.sum(
            dim=-1, keepdim=True,
        ).clamp_min(1e-8)
        memory = torch.bmm(
            weights.unsqueeze(1), history_medications,
        ).squeeze(1)
        return memory, weights

    def aggregate_history_channels(
            self,
            current_hidden: torch.Tensor,
            visit_sequence: torch.Tensor,
            visit_lengths: torch.Tensor,
            history_medications: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attention_memory, weights = self.aggregate_history(
            current_hidden,
            visit_sequence,
            visit_lengths,
            history_medications,
        )
        batch_size, max_visits, med_count = history_medications.shape
        visit_index = torch.arange(
            max_visits, device=history_medications.device,
        ).unsqueeze(0)
        history_count = (visit_lengths - 1).clamp_min(0)
        history_mask = visit_index < history_count.unsqueeze(1)
        masked_history = history_medications * history_mask.unsqueeze(-1).to(
            history_medications.dtype,
        )
        frequency = masked_history.sum(dim=1)
        frequency = frequency / history_count.clamp_min(1).unsqueeze(1)
        has_history = history_count > 0
        last_index = (visit_lengths - 2).clamp_min(0)
        batch_index = torch.arange(
            batch_size, device=history_medications.device,
        )
        last_use = history_medications[batch_index, last_index]
        last_use = last_use * has_history.unsqueeze(1).to(last_use.dtype)

        distance = (visit_lengths - 1).unsqueeze(1) - visit_index
        recency_rate = F.softplus(self.recency_log_rate)
        recency_weight = torch.exp(
            -recency_rate
            * (distance - 1).clamp_min(0).to(history_medications.dtype)
        )
        recency_weight = recency_weight * history_mask.to(
            recency_weight.dtype,
        )
        recency = (
            history_medications * recency_weight.unsqueeze(-1)
        ).amax(dim=1)
        channels = torch.stack([
            attention_memory, last_use, frequency, recency,
        ], dim=-1)
        if channels.shape != (batch_size, med_count, 4):
            raise RuntimeError("Unexpected medication history shape.")
        return channels, weights

    def encode_temporal_history(
            self,
            history_medications: torch.Tensor,
            visit_lengths: torch.Tensor) -> torch.Tensor:
        if history_medications.dim() != 3:
            raise ValueError("Historical medications must be rank-3.")
        batch_size, max_visits, med_count = history_medications.shape
        if visit_lengths.shape != (batch_size,):
            raise ValueError("Visit lengths must match the batch dimension.")
        visit_index = torch.arange(
            max_visits, device=history_medications.device,
        ).unsqueeze(0)
        history_count = (visit_lengths - 1).clamp_min(0)
        history_mask = visit_index < history_count.unsqueeze(1)
        masked_history = history_medications * history_mask.unsqueeze(-1).to(
            history_medications.dtype,
        )
        sequences = masked_history.unsqueeze(-1).permute(
            0, 2, 1, 3,
        ).reshape(batch_size * med_count, max_visits, 1)
        temporal_outputs, _ = self.temporal_history_gru(sequences)
        temporal_outputs = temporal_outputs.view(
            batch_size, med_count, max_visits, -1,
        )
        last_index = (history_count - 1).clamp_min(0)
        gather_index = last_index.view(
            batch_size, 1, 1, 1,
        ).expand(-1, med_count, 1, temporal_outputs.size(-1))
        temporal_state = temporal_outputs.gather(
            2, gather_index,
        ).squeeze(2)
        temporal_delta = self.temporal_history_projection(temporal_state)
        has_history = history_count > 0
        return temporal_delta * has_history.view(
            batch_size, 1, 1,
        ).to(temporal_delta.dtype)

    def forward(
            self,
            hidden: torch.Tensor,
            graph_readout: torch.Tensor,
            visit_sequence: torch.Tensor,
            visit_lengths: torch.Tensor,
            history_medications: torch.Tensor,
            drug_bank: torch.Tensor) -> torch.Tensor:
        if drug_bank.shape != (self.med_vocab_size, self.hidden_dim):
            raise ValueError(
                "Drug bank shape must be [med_vocab_size, hidden_dim].",
            )
        logits = self.base_decoder(hidden, graph_readout)
        history_channels, _ = self.aggregate_history_channels(
            hidden,
            visit_sequence,
            visit_lengths,
            history_medications,
        )
        patient_features = self._patient_features(hidden, graph_readout)
        expert_deltas = torch.stack([
            projection(drug_bank)
            for projection in self.history_channel_drug_projections
        ], dim=0)
        history_delta = torch.einsum(
            "bmc,cmh->bmh", history_channels, expert_deltas,
        )
        temporal_delta = self.encode_temporal_history(
            history_medications, visit_lengths,
        )
        history_delta = (
            history_delta + self.temporal_strength * temporal_delta
        )
        personalized_drugs = self.drug_norm(
            drug_bank.unsqueeze(0) + history_delta,
        )
        patient_query = self.patient_projection(patient_features)
        scale_logits = torch.einsum(
            "bmh,bh->bm", personalized_drugs, patient_query,
        ) / math.sqrt(self.hidden_dim)
        return logits + self.scale_strength * scale_logits


class DrugTGCTModel(nn.Module):
    """Stage-I hypergraph encoder and Stage-II TGCT recommendation model."""

    def __init__(
            self,
            stage1_model: nn.Module,
            stage1_data,
            stage1_visit_lookup: Dict[Tuple[int, int], int],
            concept_count: int,
            stage1_visit_count: int,
            hidden_dim: int,
            med_vocab_size: int,
            med_node_start: int = 0,
            concat_graph: bool = True,
            dropout: float = 0.1,
            sequence_encoder: str = TGCT_ENCODER):
        super().__init__()
        if concept_count <= 0:
            raise ValueError("concept_count must be positive.")
        if stage1_visit_count <= 0:
            raise ValueError("stage1_visit_count must be positive.")
        if sequence_encoder != TGCT_ENCODER:
            raise ValueError(f"Unknown sequence encoder: {sequence_encoder}")
        if med_node_start < 0 or (
                med_node_start + med_vocab_size > concept_count):
            raise ValueError(
                "Medication node range must fit inside the Stage-I concept bank.",
            )
        self.stage1_model = stage1_model
        self.stage1_data = stage1_data
        self.concept_count = concept_count
        self.stage1_visit_count = stage1_visit_count
        self.med_vocab_size = med_vocab_size
        self.med_node_start = med_node_start
        data_device = (
            stage1_data.struct_feat.device
            if hasattr(stage1_data, "struct_feat")
            else next(stage1_model.parameters()).device
        )
        model_device = next(stage1_model.parameters()).device
        if data_device != model_device:
            raise ValueError(
                "Stage-I encoder and data must reside on the same device.",
            )

        self.encoder = TGCTSequenceEncoder(
            hidden_dim,
            stage1_visit_lookup,
            dropout=dropout,
        )
        self.decoder = PrescriptionDecoder(
            hidden_dim,
            med_vocab_size,
            concat_graph,
            dropout=dropout,
        )
        self.concat_graph = concat_graph
        self.sequence_encoder = sequence_encoder

    def _stage1_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        _, edge_feat, node_feat = self.stage1_model(self.stage1_data)
        node_bank = node_feat[:self.concept_count]
        visit_bank = edge_feat[:self.stage1_visit_count]
        return visit_bank, node_bank

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        visit_bank, node_bank = self._stage1_embeddings()
        outputs = self.encoder(batch, node_bank, visit_bank)
        if outputs.visit_sequence is None:
            raise RuntimeError(
                "Medication history decoder requires per-visit encoder states.",
            )
        drug_bank = node_bank[
            self.med_node_start:self.med_node_start + self.med_vocab_size
        ]
        return self.decoder(
            outputs.final_hidden,
            outputs.final_graph,
            outputs.visit_sequence,
            batch["visit_lengths"],
            batch["history_medications"],
            drug_bank,
        )

    def stage1_parameters(self):
        return self.stage1_model.parameters()

    def downstream_parameters(self):
        stage1_param_ids = {
            id(parameter) for parameter in self.stage1_model.parameters()
        }
        for parameter in self.parameters():
            if id(parameter) not in stage1_param_ids:
                yield parameter
