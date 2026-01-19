from __future__ import annotations

import dill
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import ipdb
import torch
from torch.utils.data import Dataset

from pretraining.data_builder import VisitSummary


@dataclass
class SequenceVisit:
    visit_index: int
    diag_nodes: Sequence[int]
    proc_nodes: Sequence[int]
    med_nodes: Sequence[int]
    order_index: int


@dataclass
class SequenceExample:
    patient_index: int
    visits: List[SequenceVisit]
    target_multi_hot: torch.Tensor
    visit_ids: List[int]
    node_ids: torch.Tensor
    node_types: torch.Tensor
    visit_node_masks: torch.Tensor
    prior_prob: torch.Tensor


def _load_records(records_path: str):
    with open(records_path, 'rb') as f:
        return dill.load(f)


def _split_patient_indices(num_patients: int):
    train_end = int(num_patients * 2 / 3)
    remaining = num_patients - train_end
    test_len = remaining // 2
    train_idx = list(range(train_end))
    test_idx = list(range(train_end, train_end + test_len))
    eval_idx = list(range(train_end + test_len, num_patients))
    return train_idx, eval_idx, test_idx


def _group_visit_summaries(visit_summaries: Sequence[VisitSummary]) -> Dict[int, List[VisitSummary]]:
    grouped: Dict[int, List[VisitSummary]] = {}
    for summary in visit_summaries:
        grouped.setdefault(summary.patient_index, []).append(summary)
    for patient in grouped.values():
        patient.sort(key=lambda v: v.visit_index)
    return grouped


class VisitSequenceDataset(Dataset):
    def __init__(self,
                 records_path: str,
                 artifacts,
                 split: str,
                 med_vocab_size: int,
                 cooccurrence_matrix):
        super().__init__()
        if not isinstance(cooccurrence_matrix, torch.Tensor):
            cooccurrence_matrix = torch.as_tensor(cooccurrence_matrix)
        self.coocc_matrix = cooccurrence_matrix.float()
        raw_records = _load_records(records_path)
        train_idx, eval_idx, test_idx = _split_patient_indices(len(raw_records))
        if split == 'train':
            selected = train_idx
        elif split == 'eval':
            selected = eval_idx
        elif split == 'test':
            selected = test_idx
        else:
            raise ValueError(f'Unknown split {split}')

        grouped = _group_visit_summaries(artifacts.visit_summaries)
        med_offset = artifacts.node_ranges.med[0]
        self.examples: List[SequenceExample] = []
        for patient_idx in selected:
            visits = grouped.get(patient_idx)
            if not visits or len(visits) < 2:
                continue
            target_visit = visits[-1]
            target_codes = [node - med_offset for node in target_visit.med_nodes]
            if not target_codes:
                continue
            target_vec = torch.zeros(med_vocab_size, dtype=torch.float32)
            target_vec[target_codes] = 1.0
            sequence_visits: List[SequenceVisit] = []
            for idx, visit in enumerate(visits):
                med_nodes = visit.med_nodes if idx < len(visits) - 1 else []
                sequence_visits.append(
                    SequenceVisit(
                        visit_index=visit.visit_index,
                        diag_nodes=visit.diag_nodes,
                        proc_nodes=visit.proc_nodes,
                        med_nodes=med_nodes,
                        order_index=visit.order_index,
                    )
                )
            node_ids, node_types, visit_masks = self._build_union(sequence_visits)
            prior_prob = self._build_prior(node_ids)
            self.examples.append(
                SequenceExample(
                    patient_index=patient_idx,
                    visits=sequence_visits,
                    target_multi_hot=target_vec,
                    visit_ids=[v.visit_index for v in sequence_visits],
                    node_ids=node_ids,
                    node_types=node_types,
                    visit_node_masks=visit_masks,
                    prior_prob=prior_prob,
                )
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> SequenceExample:
        return self.examples[idx]

    @staticmethod
    def _build_union(visits: Sequence[SequenceVisit]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ordered: Dict[int, int] = {}
        node_types: Dict[int, int] = {}
        last_index = len(visits) - 1 if visits else -1

        def _register(nodes: Sequence[int], kind: int, include: bool = True):
            if not include:
                return
            for nid in nodes:
                if nid not in ordered:
                    ordered[nid] = len(ordered)
                    node_types[nid] = kind

        for idx, visit in enumerate(visits):
            _register(visit.diag_nodes, 0)
            _register(visit.proc_nodes, 1)
            _register(visit.med_nodes, 2, include=(idx != last_index))

        if not ordered:
            return torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long), torch.zeros(0, 0, dtype=torch.bool)

        ids = list(ordered.keys())
        node_ids = torch.tensor(ids, dtype=torch.long)
        type_tensor = torch.tensor([node_types[nid] for nid in ids], dtype=torch.long)
        masks: List[torch.Tensor] = []
        for idx, visit in enumerate(visits):
            mask = torch.zeros(len(ids), dtype=torch.bool)
            for nid in set(list(visit.diag_nodes) + list(visit.proc_nodes)):
                mask[ordered[nid]] = True
            if idx != last_index:
                for nid in set(visit.med_nodes):
                    if nid in ordered:
                        mask[ordered[nid]] = True
            masks.append(mask)
        visit_masks = torch.stack(masks) if masks else torch.zeros(0, len(ids), dtype=torch.bool)
        return node_ids, type_tensor, visit_masks

    def _build_prior(self, node_ids: torch.Tensor) -> torch.Tensor:
        if node_ids.numel() == 0:
            return torch.zeros(0, 0, dtype=torch.float32)
        idx = node_ids.to(dtype=torch.long, device=self.coocc_matrix.device)
        sub = self.coocc_matrix.index_select(0, idx).index_select(1, idx)
        sub = sub + torch.eye(idx.size(0), device=self.coocc_matrix.device)
        row_sum = sub.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        prior = (sub / row_sum).to(dtype=torch.float32)
        return prior.cpu()


def collate_sequence_batch(batch: List[SequenceExample]):
    batch_size = len(batch)
    targets = torch.stack([item.target_multi_hot for item in batch])
    patient_index = torch.tensor([item.patient_index for item in batch], dtype=torch.long)
    visit_lengths = torch.tensor([len(item.visits) for item in batch], dtype=torch.long)
    node_lengths = torch.tensor([item.node_ids.numel() for item in batch], dtype=torch.long)
    max_visits = int(visit_lengths.max().item()) if batch else 0
    max_nodes = int(node_lengths.max().item()) if batch else 0

    node_ids = torch.zeros(batch_size, max_nodes, dtype=torch.long)
    node_masks = torch.zeros(batch_size, max_nodes, dtype=torch.bool)
    node_types = torch.zeros(batch_size, max_nodes, dtype=torch.long)
    visit_orders = torch.full((batch_size, max_visits), -1, dtype=torch.long)
    visit_node_masks = torch.zeros(batch_size, max_visits, max_nodes, dtype=torch.bool)
    prior_prob = torch.zeros(batch_size, max_nodes, max_nodes, dtype=torch.float32)

    for b_idx, example in enumerate(batch):
        v_len = len(example.visits)
        n_len = example.node_ids.numel()
        if n_len > 0:
            node_ids[b_idx, :n_len] = example.node_ids
            node_masks[b_idx, :n_len] = True
            node_types[b_idx, :n_len] = example.node_types
        if v_len > 0:
            orders = torch.tensor([visit.order_index for visit in example.visits], dtype=torch.long)
            visit_orders[b_idx, :v_len] = orders
            visit_node_masks[b_idx, :v_len, :n_len] = example.visit_node_masks[:v_len, :n_len]
        prior = example.prior_prob
        if prior.numel() > 0:
            size = prior.size(0)
            prior_prob[b_idx, :size, :size] = prior

    return {
        'targets': targets,
        'patient_index': patient_index,
        'visit_orders': visit_orders,
        'visit_node_masks': visit_node_masks,
        'visit_lengths': visit_lengths,
        'node_ids': node_ids,
        'node_masks': node_masks,
        'node_types': node_types,
        'node_lengths': node_lengths,
        'prior_prob': prior_prob,
    }