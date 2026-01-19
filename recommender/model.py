from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EncoderOutputs:
    final_hidden: torch.Tensor
    final_graph: torch.Tensor
    last_visit_repr: torch.Tensor


class MYLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1, prior_scale: float = 1.0):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError('hidden_dim must be divisible by num_heads for MYLayer.')
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = prior_scale

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.q_fuse = nn.Linear(hidden_dim * 2, hidden_dim)
        self.k_fuse = nn.Linear(hidden_dim * 2, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.residual_norm = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)

    def forward(self,
                x: torch.Tensor,
                prior: torch.Tensor,
                mask: torch.Tensor,
                prev_query: torch.Tensor = None,
                prev_key: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_nodes, _ = x.shape
        mask_float = mask.float()
        residual = x
        q_proj = self.q_proj(x) * mask_float.unsqueeze(-1)
        k_proj = self.k_proj(x) * mask_float.unsqueeze(-1)
        v_proj = self.v_proj(x) * mask_float.unsqueeze(-1)
        if prev_query is not None:
            fused_q = self.q_fuse(torch.cat([q_proj, prev_query], dim=-1))
        else:
            fused_q = q_proj
        if prev_key is not None:
            fused_k = self.k_fuse(torch.cat([k_proj, prev_key], dim=-1))
        else:
            fused_k = k_proj

        q = fused_q.view(batch_size, num_nodes, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = fused_k.view(batch_size, num_nodes, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v_proj.view(batch_size, num_nodes, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if prior is not None:
            scores = scores + self.scale * prior.unsqueeze(1)
        inactive = (~mask).unsqueeze(1).unsqueeze(-2)
        mask_value = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(inactive, mask_value)
        scores = scores.masked_fill(inactive.transpose(-1, -2), mask_value)
        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
        attn = self.attn_dropout(attn)
        context = torch.matmul(attn, v)
        context = context.permute(0, 2, 1, 3).contiguous().view(batch_size, num_nodes, self.hidden_dim)
        x = self.residual_norm(residual + self.out_proj(context))
        x = x * mask_float.unsqueeze(-1)
        x = self.ffn_norm(x + self.ffn(x))
        return x, fused_q, fused_k


class ReadoutStack(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 hidden_dim: int,
                 num_layers: int,
                 num_heads: int,
                 dropout: float,
                 prior_scale: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = max(1, num_layers)
        self.input_proj = nn.Linear(embed_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            MYLayer(hidden_dim, num_heads, dropout, prior_scale) for _ in range(self.num_layers)
        ])
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(self,
                features: torch.Tensor,
                prior: torch.Tensor,
                mask: torch.Tensor,
                visit_active: torch.Tensor,
                prev_states: List[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        device = features.device
        batch_size = features.size(0)
        if prev_states is None:
            prev_states = [None] * self.num_layers
        x = self.dropout(self.input_proj(features))
        new_states: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for idx, layer in enumerate(self.layers):
            layer_prior = prior if idx == 0 else None
            prev_pair = prev_states[idx] if idx < len(prev_states) else None
            prev_q = prev_pair[0] if prev_pair is not None else None
            prev_k = prev_pair[1] if prev_pair is not None else None
            x, q_state, k_state = layer(x, layer_prior, mask, prev_q, prev_k)
            if prev_q is not None:
                q_state = torch.where(visit_active.view(batch_size, 1, 1), q_state, prev_q)
            if prev_k is not None:
                k_state = torch.where(visit_active.view(batch_size, 1, 1), k_state, prev_k)
            new_states.append((q_state, k_state))
        visit_token = x[:, 0, :]
        visit_state = self.readout(visit_token) 
        visit_state = visit_state * visit_active.unsqueeze(-1).float()
        return visit_state, new_states


class PrototypeMoE(nn.Module):
    def __init__(self, hidden_dim: int, prototypes: torch.Tensor):
        super().__init__()
        if prototypes.dim() != 2 or prototypes.size(1) != hidden_dim:
            raise ValueError('Prototype dimension must match hidden dimension.')
        self.hidden_dim = hidden_dim
        self.num_experts = prototypes.size(0)
        self.prototypes = nn.Parameter(prototypes.clone())
        self.W_p = nn.Linear(hidden_dim, hidden_dim)
        self.weight_z = nn.Parameter(torch.empty(self.num_experts, hidden_dim, hidden_dim))
        self.weight_r = nn.Parameter(torch.empty(self.num_experts, hidden_dim, hidden_dim))
        self.weight_h = nn.Parameter(torch.empty(self.num_experts, hidden_dim, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_z)
        nn.init.xavier_uniform_(self.weight_r)
        nn.init.xavier_uniform_(self.weight_h)

    def forward(self, stage_visit: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        squeeze = False
        if stage_visit.dim() == 1:
            stage_visit = stage_visit.unsqueeze(0)
            squeeze = True
        prototypes = self.prototypes
        projected = self.W_p(stage_visit)
        scores = torch.matmul(projected, prototypes.t())
        mixture = torch.softmax(scores, dim=-1)
        W_z = torch.einsum('be,ehd->bhd', mixture, self.weight_z)
        W_r = torch.einsum('be,ehd->bhd', mixture, self.weight_r)
        W_h = torch.einsum('be,ehd->bhd', mixture, self.weight_h)
        if squeeze:
            return W_z[0], W_r[0], W_h[0], mixture[0]
        return W_z, W_r, W_h, mixture


class GraphGRUCell(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.U_z = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.U_r = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.U_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.progress_gate = nn.Linear(3, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def forward(self,
                graph_state: torch.Tensor,
                prev_hidden: torch.Tensor,
                W_z: torch.Tensor,
                W_r: torch.Tensor,
                W_h: torch.Tensor,
                progress_features: torch.Tensor,
                visit_active: torch.Tensor) -> torch.Tensor:
        g_t = torch.sigmoid(self.progress_gate(progress_features))
        z_t = torch.sigmoid(torch.bmm(W_z, graph_state.unsqueeze(-1)).squeeze(-1) + self.U_z(prev_hidden))
        r_t = torch.sigmoid(torch.bmm(W_r, graph_state.unsqueeze(-1)).squeeze(-1) + self.U_r(prev_hidden))
        candidate = torch.tanh(torch.bmm(W_h, graph_state.unsqueeze(-1)).squeeze(-1) + self.U_h(r_t * prev_hidden))
        combined = (1 - z_t) * prev_hidden + z_t * candidate
        h_t = (1 - g_t) * prev_hidden + g_t * combined
        visit_mask = visit_active.unsqueeze(-1)
        h_t = torch.where(visit_mask, h_t, prev_hidden)
        return self.dropout(h_t)


class SequenceEncoder(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 visit_lookup: Dict[Tuple[int, int], int],
                 visit_prototypes: torch.Tensor,
                 readout_mode: str = 'shared',
                 window_size: int = 3,
                 readout_layers: int = 2,
                 readout_heads: int = 2,
                 prior_scale: float = 1.0,
                 dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.readout_mode = readout_mode
        self.window_size = window_size
        if self.readout_mode == 'windowed':
            if self.window_size <= 0:
                raise ValueError('window_size must be positive when -readout-mode=windowed.')
            self.graph_readouts = nn.ModuleList([
                ReadoutStack(
                    embed_dim=hidden_dim,
                    hidden_dim=hidden_dim,
                    num_layers=readout_layers,
                    num_heads=readout_heads,
                    dropout=dropout,
                    prior_scale=prior_scale,
                )
                for _ in range(self.window_size)
            ])
        else:
            self.graph_readout = ReadoutStack(
                embed_dim=hidden_dim,
                hidden_dim=hidden_dim,
                num_layers=readout_layers,
                num_heads=readout_heads,
                dropout=dropout,
                prior_scale=prior_scale,
            )
        self.prototype_moe = PrototypeMoE(hidden_dim, visit_prototypes)
        self.graph_gru = GraphGRUCell(hidden_dim, dropout)
        self.visit_lookup = visit_lookup
        self.visit_bias = nn.Parameter(torch.zeros(hidden_dim))

    def _stage1_visit_embedding(self, patient_index: int, order_index: int, visit_bank: torch.Tensor) -> torch.Tensor:
        idx = self.visit_lookup.get((patient_index, order_index))
        if idx is None or idx >= visit_bank.size(0):
            return self.visit_bias
        return visit_bank[idx]

    def forward(self,
                batch_inputs: Dict[str, torch.Tensor],
                node_bank: torch.Tensor,
                visit_bank: torch.Tensor) -> EncoderOutputs:
        device = node_bank.device
        node_ids = batch_inputs['node_ids']
        node_masks = batch_inputs['node_masks']
        node_types = batch_inputs['node_types']
        node_lengths = batch_inputs['node_lengths']
        visit_masks = batch_inputs['visit_node_masks']
        visit_orders = batch_inputs['visit_orders']
        visit_lengths = batch_inputs['visit_lengths']
        patient_indices = batch_inputs['patient_index']
        prior_prob = batch_inputs['prior_prob']

        batch_size = node_ids.size(0)
        max_nodes = node_ids.size(1)
        max_visits = visit_masks.size(1)

        if max_nodes > 0:
            node_ids_clamped = torch.where(node_masks, node_ids, torch.zeros_like(node_ids))
            base_features = F.embedding(node_ids_clamped, node_bank)
            base_features = base_features * node_masks.unsqueeze(-1).float()
        else:
            base_features = torch.zeros(batch_size, 0, self.hidden_dim, device=device)


        stage_visit_embeds = torch.zeros(batch_size, max_visits, self.hidden_dim, device=device)
        stage_visit_embeds += self.visit_bias.view(1, 1, -1)
        visit_orders_list = visit_orders.cpu().tolist()
        patient_list = patient_indices.cpu().tolist()
        for b_idx, patient in enumerate(patient_list):
            limit = int(visit_lengths[b_idx].item())
            for v_idx in range(limit):
                order_index = visit_orders_list[b_idx][v_idx]
                visit_tensor = self._stage1_visit_embedding(patient, order_index, visit_bank)
                stage_visit_embeds[b_idx, v_idx] = visit_tensor.to(device)

        type_masks = [(node_types == t) & node_masks for t in range(3)] if max_nodes > 0 else [None, None, None]
        if self.readout_mode == 'windowed':
            modules = self.graph_readouts
            slot_count = self.window_size
            slot_offset = (self.window_size - visit_lengths).unsqueeze(1)
            idx_range = torch.arange(max_visits, device=device).unsqueeze(0)
            slot_indices = (slot_offset + idx_range).clamp(0, self.window_size - 1)
        else:
            modules = [self.graph_readout]
            slot_count = 1
            slot_indices = torch.zeros(batch_size, max_visits, dtype=torch.long, device=device)
        readout_states: List[List[Tuple[torch.Tensor, torch.Tensor]]] = [None] * slot_count

        h_t = self.graph_gru.init_hidden(batch_size, device)
        final_graph = torch.zeros(batch_size, self.hidden_dim, device=device)
        visit_repr_last = torch.zeros(batch_size, self.hidden_dim, device=device)
        prev_adj = torch.zeros(batch_size, max_nodes, max_nodes, device=device)
        has_prev = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for visit_idx in range(max_visits):
            visit_active = visit_idx < visit_lengths
            if not visit_active.any():
                continue
            code_mask = visit_masks[:, visit_idx, :] & node_masks
            visit_embed = stage_visit_embeds[:, visit_idx, :] * visit_active.unsqueeze(-1)
            visit_features = torch.cat([visit_embed.unsqueeze(1), base_features], dim=1)
            mask_ext = torch.cat([visit_active.unsqueeze(1), code_mask], dim=1)
            prior_prob_ext = torch.zeros(batch_size, max_nodes + 1, max_nodes + 1, device=device)
            if max_nodes > 0:
                prior_prob_ext[:, 1:, 1:] = prior_prob
                active_float = code_mask.float()
                denom = active_float.sum(dim=1, keepdim=True).clamp_min(1e-6)
                weights = torch.where(denom > 0, active_float / denom, torch.zeros_like(active_float))
                prior_prob_ext[:, 0, 1:] = weights
                prior_prob_ext[:, 1:, 0] = weights
            prior_prob_ext[:, 0, 0] = 1.0
            prior_ext = torch.log(prior_prob_ext.clamp_min(1e-6))
            #! History-aware Hidden Dependency Inference.
            graph_state = torch.zeros(batch_size, self.hidden_dim, device=device)
            for slot_idx, module in enumerate(modules):
                slot_active = visit_active & (slot_indices[:, visit_idx] == slot_idx)
                if not slot_active.any():
                    continue
                state, updated_state = module(
                    visit_features,
                    prior_ext,
                    mask_ext,
                    visit_active=slot_active,
                    prev_states=readout_states[slot_idx],
                )
                graph_state = torch.where(slot_active.unsqueeze(-1), state, graph_state)
                readout_states[slot_idx] = updated_state
            #! PDD GRU
            progress_vec = torch.zeros(batch_size, 3, device=device)
            if max_nodes > 0:
                active_mask = code_mask.float().unsqueeze(1) * code_mask.float().unsqueeze(2)
                adjacency = prior_prob * active_mask
                valid_progress = visit_active & has_prev
                delta = torch.abs(adjacency - prev_adj) * valid_progress.view(batch_size, 1, 1).float() # 
                degree_delta = delta.sum(dim=2)
                #! pooling
                for type_idx, t_mask in enumerate(type_masks):
                    if t_mask is None:
                        continue
                    mask_float = t_mask.float()
                    mask_sum = mask_float.sum(dim=1)
                    stats = torch.zeros(batch_size, device=device)
                    valid_type = mask_sum > 0
                    if valid_type.any():
                        masked_delta = (degree_delta * mask_float).sum(dim=1)
                        stats[valid_type] = masked_delta[valid_type] / mask_sum[valid_type]
                    progress_vec[:, type_idx] = stats
                progress_vec = progress_vec * visit_active.unsqueeze(-1).float()
                prev_adj = torch.where(visit_active.view(batch_size, 1, 1), adjacency, prev_adj)
                has_prev = has_prev | visit_active

            W_z, W_r, W_h, mixture = self.prototype_moe(stage_visit_embeds[:, visit_idx, :])
            h_t = self.graph_gru(
                graph_state=graph_state,
                prev_hidden=h_t,
                W_z=W_z,
                W_r=W_r,
                W_h=W_h,
                progress_features=progress_vec,
                visit_active=visit_active,
            )
            final_graph = torch.where(visit_active.unsqueeze(-1), graph_state, final_graph)
            visit_repr_last = torch.where(visit_active.unsqueeze(-1), graph_state, visit_repr_last)
        return EncoderOutputs(
            final_hidden=h_t,
            final_graph=final_graph,
            last_visit_repr=visit_repr_last,
        )


class PrescriptionDecoder(nn.Module):
    def __init__(self, hidden_dim: int, med_vocab_size: int, concat_graph: bool):
        super().__init__()
        input_dim = hidden_dim + (hidden_dim if concat_graph else 0)
        self.concat_graph = concat_graph
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, med_vocab_size),
        )

    def forward(self, hidden: torch.Tensor, graph_readout: torch.Tensor) -> torch.Tensor:
        if self.concat_graph:
            features = torch.cat([hidden, graph_readout], dim=-1)
        else:
            features = hidden
        return self.classifier(features)


class DrugModel(nn.Module):
    def __init__(self,
                 stage1_model: nn.Module,
                 stage1_data,
                 stage1_visit_lookup: Dict[Tuple[int, int], int],
                 visit_prototypes: torch.Tensor,
                 concept_count: int,
                 stage1_visit_count: int,
                 hidden_dim: int,
                 med_vocab_size: int,
                 num_experts: int = 4,
                 concat_graph: bool = True,
                 readout_mode: str = 'shared',
                 window: int = 3,
                 readout_layers: int = 2,
                 readout_heads: int = 2,
                 prior_scale: float = 1.0,
                 dropout: float = 0.1):
        super().__init__()
        if visit_prototypes.size(0) != num_experts:
            raise ValueError('visit_prototypes rows must equal num_experts.')
        if visit_prototypes.size(1) != hidden_dim:
            raise ValueError('Prototype dimension must equal seq-hidden-dim.')
        if concept_count <= 0:
            raise ValueError('concept_count must be positive.')
        if stage1_visit_count <= 0:
            raise ValueError('stage1_visit_count must be positive.')
        visit_prototypes = visit_prototypes.detach().clone()
        self.stage1_model = stage1_model
        self.stage1_data = stage1_data
        self.concept_count = concept_count
        self.stage1_visit_count = stage1_visit_count
        data_device = getattr(stage1_data, 'struct_feat').device if hasattr(stage1_data, 'struct_feat') else next(stage1_model.parameters()).device
        model_device = next(stage1_model.parameters()).device
        if data_device != model_device:
            raise ValueError('Stage-I encoder and data must reside on the same device.')
        self.encoder = SequenceEncoder(
            hidden_dim,
            stage1_visit_lookup,
            visit_prototypes,
            readout_mode=readout_mode,
            window_size=window,
            readout_layers=readout_layers,
            readout_heads=readout_heads,
            prior_scale=prior_scale,
            dropout=dropout,
        )
        self.decoder = PrescriptionDecoder(hidden_dim, med_vocab_size, concat_graph)
        self.concat_graph = concat_graph

    def _stage1_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        _, edge_feat, node_feat = self.stage1_model(self.stage1_data)
        node_bank = node_feat[:self.concept_count]
        visit_bank = edge_feat[:self.stage1_visit_count]
        return visit_bank, node_bank

    def forward(self, batch):
        visit_bank, node_bank = self._stage1_embeddings()
        outputs = self.encoder(batch, node_bank, visit_bank)
        logits = self.decoder(outputs.final_hidden, outputs.final_graph)
        return logits

    def stage1_parameters(self):
        return self.stage1_model.parameters()

    def downstream_parameters(self):
        stage1_param_ids = {id(p) for p in self.stage1_model.parameters()}
        for param in self.parameters():
            if id(param) not in stage1_param_ids:
                yield param