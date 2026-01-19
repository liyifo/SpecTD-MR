from __future__ import annotations
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data_builder import NodeRanges
from .models_ext import SetGNNExtended


def compute_initializer_output_dim(struct_dim: int,
                                   text_dim: int,
                                   logic_dim: int,
                                   projection_dim: int,
                                   use_text: bool,
                                   use_logic: bool,
                                   project_inputs: bool) -> int:
    has_text = use_text and text_dim > 0
    has_logic = use_logic and logic_dim > 0
    if project_inputs:
        struct_out = projection_dim
        sem_out = projection_dim if (has_text or has_logic) else 0
        return struct_out + sem_out
    output_dim = struct_dim
    if has_text:
        output_dim += text_dim
    if has_logic:
        output_dim += logic_dim
    return output_dim


class NodeFeatureInitializer(nn.Module):
    def __init__(self,
                 struct_dim: int,
                 text_dim: int,
                 logic_dim: int,
                 feature_dim: int,
                 use_text: bool,
                 use_logic: bool,
                 project_inputs: bool = True):
        super().__init__()
        self.use_text = use_text and text_dim > 0
        self.use_logic = use_logic and logic_dim > 0
        self.project_inputs = project_inputs
        self.has_sem = self.use_text or self.use_logic
        self.output_dim = compute_initializer_output_dim(
            struct_dim=struct_dim,
            text_dim=text_dim,
            logic_dim=logic_dim,
            projection_dim=feature_dim,
            use_text=use_text,
            use_logic=use_logic,
            project_inputs=project_inputs,
        )

        if self.project_inputs:
            self.struct_proj = nn.Linear(struct_dim, feature_dim)
            self.text_proj = nn.Linear(text_dim, feature_dim) if self.use_text else None
            self.logic_proj = nn.Linear(logic_dim, feature_dim) if self.use_logic else None
            self.gate = nn.Linear(feature_dim * 2, feature_dim) if (self.use_text and self.use_logic) else None
        else:
            self.struct_proj = nn.Identity()
            self.text_proj = nn.Identity() if self.use_text else None
            self.logic_proj = nn.Identity() if self.use_logic else None
            self.gate = None
        self.norm = nn.LayerNorm(self.output_dim)

    def forward(self, struct_feat: torch.Tensor,
                text_feat: torch.Tensor,
                logic_feat: torch.Tensor) -> torch.Tensor:
        struct_state = self.struct_proj(struct_feat)
        if not self.has_sem:
            return self.norm(struct_state)

        if self.project_inputs:
            text_state = self.text_proj(text_feat) if self.use_text else None
            logic_state = self.logic_proj(logic_feat) if self.use_logic else None

            if self.use_text and self.use_logic and self.gate is not None:
                gate = torch.sigmoid(self.gate(torch.cat([text_state, logic_state], dim=-1)))
                sem_state = gate * text_state + (1 - gate) * logic_state
            else:
                sem_state = text_state if self.use_text else logic_state
            fused = torch.cat([struct_state, sem_state], dim=-1)
            return self.norm(fused)

        parts = [struct_state]
        if self.use_text:
            parts.append(text_feat)
        if self.use_logic:
            parts.append(logic_feat)
        return self.norm(torch.cat(parts, dim=-1))


class RelativePositionEncoder(nn.Module):
    def __init__(self, num_buckets: int, heads: int):
        super().__init__()
        self.embedding = nn.Embedding(num_buckets, heads)

    def forward(self, edge_rel_pos: torch.Tensor) -> torch.Tensor:
        return self.embedding(edge_rel_pos)


class VisitPredictionHead(nn.Module):
    def __init__(self, hidden_dim: int, diag_size: int, proc_size: int, med_size: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.diag = nn.Linear(hidden_dim, diag_size)
        self.proc = nn.Linear(hidden_dim, proc_size)
        self.med = nn.Linear(hidden_dim, med_size)

    def forward(self, visit_emb: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.dropout(visit_emb)
        return {
            'diag': self.diag(h),
            'proc': self.proc(h),
            'med': self.med(h)
        }


class DrugHypergraphPretrainer(nn.Module):
    def __init__(self,
                 encoder: SetGNNExtended,
                 node_ranges: NodeRanges,
                 struct_dim: int,
                 text_dim: int,
                 logic_dim: int,
                 feature_dim: int,
                 hidden_dim: int,
                 diag_size: int,
                 proc_size: int,
                 med_size: int,
                 rel_pos_buckets: int,
                 heads: int,
                 dropout: float,
                 use_text: bool = True,
                 use_logic: bool = True,
                 project_node_embeds: bool = True,
                 embedding_init_range: Optional[float] = None):
        super().__init__()
        self.encoder = encoder
        self.node_ranges = node_ranges
        self.use_text = use_text
        self.use_logic = use_logic
        self.initializer = NodeFeatureInitializer(struct_dim, text_dim, logic_dim, feature_dim,
                                                  use_text=use_text, use_logic=use_logic,
                                                  project_inputs=project_node_embeds)
        self.visit_head = VisitPredictionHead(hidden_dim, diag_size, proc_size, med_size, dropout=dropout)
        self.relpos = RelativePositionEncoder(rel_pos_buckets, heads)
        self.struct_mask_token = nn.Parameter(torch.zeros(struct_dim))
        if self.use_text:
            self.text_mask_token = nn.Parameter(torch.zeros(text_dim))
        else:
            self.register_parameter('text_mask_token', None)
        if self.use_logic:
            self.logic_mask_token = nn.Parameter(torch.zeros(logic_dim))
        else:
            self.register_parameter('logic_mask_token', None)
        if embedding_init_range is not None:
            self._init_embedding_parameters(embedding_init_range)

    def _init_embedding_parameters(self, init_range: float) -> None:
        if init_range <= 0:
            return
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                module.weight.data.uniform_(-init_range, init_range)

    def forward(self, data, mask_plan: Optional[Dict[str, torch.Tensor]] = None):
        struct_feat = data.struct_feat
        text_feat = data.text_feat
        logic_feat = data.logic_feat
        if mask_plan is not None:
            random_targets = mask_plan.get('random_targets')
            mask_token_mask = mask_plan.get('mask_token')

            if random_targets is not None:
                random_mask = random_targets >= 0
                if random_mask.any():
                    struct_feat = struct_feat.clone()
                    source_idx = random_targets[random_mask]
                    struct_feat[random_mask] = data.struct_feat[source_idx]
                    if self.use_text:
                        if text_feat is data.text_feat:
                            text_feat = text_feat.clone()
                        text_feat[random_mask] = data.text_feat[source_idx]
                    if self.use_logic:
                        if logic_feat is data.logic_feat:
                            logic_feat = logic_feat.clone()
                        logic_feat[random_mask] = data.logic_feat[source_idx]

            if mask_token_mask is not None and mask_token_mask.any():
                mask_float = mask_token_mask.float().unsqueeze(-1)
                struct_feat = struct_feat * (1 - mask_float) + self.struct_mask_token * mask_float
                if self.use_text:
                    text_feat = text_feat * (1 - mask_float) + self.text_mask_token * mask_float
                if self.use_logic:
                    logic_feat = logic_feat * (1 - mask_float) + self.logic_mask_token * mask_float

        data.x = self.initializer(struct_feat, text_feat, logic_feat)
        rel_bias = self.relpos(data.edge_rel_pos.long())
        data.edge_rel_bias = rel_bias
        logits, edge_feat, node_feat, _ = self.encoder(data, edge_rel_bias=rel_bias)
        return logits, edge_feat, node_feat

    def masked_loss(self, visit_emb: torch.Tensor, batch_plans, device):
        logits = self.visit_head(visit_emb)
        losses: List[torch.Tensor] = []
        correct = 0
        total_targets = 0
        diag_offset = self.node_ranges.diag[0]
        proc_offset = self.node_ranges.proc[0]
        med_offset = self.node_ranges.med[0]
        for plan in batch_plans:
            vid = plan.visit_index
            if plan.diag_nodes:
                diag_vec = logits['diag'][vid]
                diag_logits = diag_vec.unsqueeze(0)
                diag_pred = diag_vec.argmax().item()
                for node_id in plan.diag_nodes:
                    target_idx = node_id - diag_offset
                    target = torch.tensor([target_idx], device=device)
                    losses.append(F.cross_entropy(diag_logits, target))
                    correct += int(diag_pred == target_idx)
                    total_targets += 1
            if plan.proc_nodes:
                proc_vec = logits['proc'][vid]
                proc_logits = proc_vec.unsqueeze(0)
                proc_pred = proc_vec.argmax().item()
                for node_id in plan.proc_nodes:
                    target_idx = node_id - proc_offset
                    target = torch.tensor([target_idx], device=device)
                    losses.append(F.cross_entropy(proc_logits, target))
                    correct += int(proc_pred == target_idx)
                    total_targets += 1
            if plan.med_nodes:
                med_vec = logits['med'][vid]
                med_logits = med_vec.unsqueeze(0)
                med_pred = med_vec.argmax().item()
                for node_id in plan.med_nodes:
                    target_idx = node_id - med_offset
                    target = torch.tensor([target_idx], device=device)
                    losses.append(F.cross_entropy(med_logits, target))
                    correct += int(med_pred == target_idx)
                    total_targets += 1
        if not losses:
            return torch.tensor(0.0, device=device), {'correct': 0, 'total': 0}
        return torch.stack(losses).mean(), {'correct': correct, 'total': total_targets}
