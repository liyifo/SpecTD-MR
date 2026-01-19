from __future__ import annotations

import dill
import torch
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from torch_geometric.data import Data

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from pretraining.preprocessing import norm_contruction  # type: ignore

from .embeddings import load_embedding_table, as_feature_matrix, reduce_with_pca
from .hierarchy import HierarchyEncoder, RelativePositionBucketizer, build_edge_relative_positions


@dataclass
class NodeRanges:
    diag: Tuple[int, int]
    proc: Tuple[int, int]
    med: Tuple[int, int]
    visit: Tuple[int, int]

    @property
    def concept_count(self) -> int:
        return self.visit[0]

    @property
    def total_nodes(self) -> int:
        return self.visit[0] + self.visit[1]


@dataclass
class VisitSummary:
    visit_index: int
    patient_index: int
    visit_node_id: int
    diag_nodes: List[int]
    proc_nodes: List[int]
    med_nodes: List[int]
    order_index: int
    is_terminal: bool = False
    split: str = 'train'


@dataclass
class DrugHypergraphArtifacts:
    data: Data
    visit_summaries: List[VisitSummary]
    node_ranges: NodeRanges
    membership_lookup: Dict[Tuple[int, int], int]
    vocabs: Dict[str, List[str]]
    split_usage: Dict[str, Dict[str, int]] = field(default_factory=dict)


def _load_pickle(path: str):
    with open(path, 'rb') as f:
        return dill.load(f)


def _build_node_metadata(voc) -> Tuple[List[str], List[str], List[str], NodeRanges]:
    diag_vocab = [voc['diag_voc'].idx2word[i] for i in range(len(voc['diag_voc'].idx2word))]
    proc_vocab = [voc['pro_voc'].idx2word[i] for i in range(len(voc['pro_voc'].idx2word))]
    med_vocab = [voc['med_voc'].idx2word[i] for i in range(len(voc['med_voc'].idx2word))]

    diag_range = (0, len(diag_vocab))
    proc_start = diag_range[0] + diag_range[1]
    med_start = proc_start + len(proc_vocab)
    visit_start = med_start + len(med_vocab)
    node_ranges = NodeRanges(
        diag=diag_range,
        proc=(proc_start, len(proc_vocab)),
        med=(med_start, len(med_vocab)),
        visit=(visit_start, 0),
    )
    return diag_vocab, proc_vocab, med_vocab, node_ranges


def _global_id(offset: int, local_idx: int) -> int:
    return offset + local_idx


def _attach_visit_range(node_ranges: NodeRanges, num_visits: int) -> NodeRanges:
    return NodeRanges(
        diag=node_ranges.diag,
        proc=node_ranges.proc,
        med=node_ranges.med,
        visit=(node_ranges.visit[0], num_visits)
    )


def _concept_type(global_id: int, ranges: NodeRanges) -> int:
    if ranges.diag[0] <= global_id < ranges.diag[0] + ranges.diag[1]:
        return 0
    if ranges.proc[0] <= global_id < ranges.proc[0] + ranges.proc[1]:
        return 1
    if ranges.med[0] <= global_id < ranges.med[0] + ranges.med[1]:
        return 2
    return 3


def _split_patient_sequences(records):
    total = len(records)
    train_end = int(total * 2 / 3)
    train_records = records[:train_end]
    remaining = records[train_end:]
    test_len = len(remaining) // 2
    test_records = records[train_end:train_end + test_len]
    eval_records = records[train_end + test_len:]
    return train_records, eval_records, test_records


def _prepare_pretraining_records(records):
    total = len(records)
    train_end = int(total * 2 / 3)
    remaining = total - train_end
    test_len = remaining // 2

    def _split_name(idx: int) -> str:
        if idx < train_end:
            return 'train'
        if idx < train_end + test_len:
            return 'test'
        return 'eval'

    def _empty_stats() -> Dict[str, int]:
        return {
            'patients_total': 0,
            'patients_used': 0,
            'visits_used': 0,
            'heldout_visits': 0,
        }

    split_usage = {
        'train': _empty_stats(),
        'eval': _empty_stats(),
        'test': _empty_stats(),
    }

    processed: List[List] = []
    terminal_lookup: Set[Tuple[int, int]] = set()
    patient_splits: List[str] = []

    for patient_idx, patient in enumerate(records):
        split = _split_name(patient_idx)
        patient_splits.append(split)
        stats = split_usage[split]
        stats['patients_total'] += 1
        if not patient:
            processed.append([])
            continue
        stats['patients_used'] += 1
        stats['visits_used'] += len(patient)
        last_visit_idx = len(patient) - 1
        cloned_patient: List[List] = []
        for visit_order, visit in enumerate(patient):
            diag_codes = list(visit[0])
            proc_codes = list(visit[1])
            med_codes = list(visit[2])
            if visit_order == last_visit_idx:
                terminal_lookup.add((patient_idx, visit_order))
                stats['heldout_visits'] += 1
            new_visit = [diag_codes, proc_codes, med_codes]
            if len(visit) > 3:
                new_visit.extend(visit[3:])
            cloned_patient.append(new_visit)
        processed.append(cloned_patient)
    return processed, split_usage, terminal_lookup, patient_splits


def _full_usage_stats(records) -> Dict[str, int]:
    non_empty = [p for p in records if p]
    return {
        'patients_total': len(records),
        'patients_used': len(non_empty),
        'visits_used': sum(len(p) for p in non_empty),
        'heldout_visits': 0,
    }


def build_drug_hypergraph(records_path: str,
                           voc_path: str,
                           struct_emb_path: Optional[str] = None,
                           text_emb_path: Optional[str] = None,
                           logic_emb_path: Optional[str] = None,
                           hierarchy_path: Optional[str] = None,
                           bucket_boundaries: Optional[Sequence[int]] = None,
                           usage: str = 'pretrain',
                           text_pca_dim: Optional[int] = None) -> DrugHypergraphArtifacts:
    raw_records = _load_pickle(records_path)
    voc = _load_pickle(voc_path)
    terminal_lookup: Set[Tuple[int, int]] = set()
    patient_splits: Optional[List[str]] = None
    if usage == 'pretrain':
        records, split_usage, terminal_lookup, patient_splits = _prepare_pretraining_records(raw_records)
    elif usage == 'full':
        records = raw_records
        train, val, test = _split_patient_sequences(raw_records)
        split_usage = {
            'train': _full_usage_stats(train),
            'eval': _full_usage_stats(val),
            'test': _full_usage_stats(test),
        }
        patient_splits = ['full'] * len(records)
    else:
        raise ValueError(f"Unknown usage mode: {usage}")

    if patient_splits is None:
        patient_splits = ['full'] * len(records)

    diag_vocab, proc_vocab, med_vocab, base_ranges = _build_node_metadata(voc)
    concept_strings = diag_vocab + proc_vocab + med_vocab

    visit_summaries: List[VisitSummary] = []
    membership_lookup: Dict[Tuple[int, int], int] = {}
    node_idx_list: List[int] = []
    visit_idx_list: List[int] = []
    edge_types: List[int] = []

    diag_offset = base_ranges.diag[0]
    proc_offset = base_ranges.proc[0]
    med_offset = base_ranges.med[0]

    visit_counter = 0
    visit_orders = defaultdict(int)
    for patient_idx, patient in enumerate(records):
        patient_split = patient_splits[patient_idx] if patient_idx < len(patient_splits) else 'full'
        for visit in patient:
            diag_codes, proc_codes, med_codes = visit[:3]
            visit_node_id = base_ranges.visit[0] + visit_counter
            diag_nodes = [_global_id(diag_offset, code) for code in diag_codes]
            proc_nodes = [_global_id(proc_offset, code) for code in proc_codes]
            med_nodes = [_global_id(med_offset, code) for code in med_codes]

            for nid in diag_nodes:
                membership_lookup[(nid, visit_node_id)] = len(node_idx_list)
                node_idx_list.append(nid)
                visit_idx_list.append(visit_node_id)
                edge_types.append(0)
            for nid in proc_nodes:
                membership_lookup[(nid, visit_node_id)] = len(node_idx_list)
                node_idx_list.append(nid)
                visit_idx_list.append(visit_node_id)
                edge_types.append(1)
            for nid in med_nodes:
                membership_lookup[(nid, visit_node_id)] = len(node_idx_list)
                node_idx_list.append(nid)
                visit_idx_list.append(visit_node_id)
                edge_types.append(2)

            order_index = visit_orders[patient_idx]
            visit_summaries.append(
                VisitSummary(
                    visit_index=visit_counter,
                    patient_index=patient_idx,
                    visit_node_id=visit_node_id,
                    diag_nodes=diag_nodes,
                    proc_nodes=proc_nodes,
                    med_nodes=med_nodes,
                    order_index=order_index,
                    is_terminal=(patient_idx, order_index) in terminal_lookup,
                    split=patient_split,
                )
            )
            visit_counter += 1
            visit_orders[patient_idx] += 1

    node_ranges = _attach_visit_range(base_ranges, visit_counter)
    total_nodes = node_ranges.total_nodes
    num_edges = len(node_idx_list)

    edge_index = torch.tensor([
        node_idx_list,
        visit_idx_list
    ], dtype=torch.long)

    # Build placeholder label tensor to keep SetGNN happy.
    data = Data(
        edge_index=edge_index,
        y=torch.zeros(visit_counter, 1, dtype=torch.float32)
    )
    data.n_x = torch.tensor([node_ranges.concept_count])
    data.num_hyperedges = torch.tensor([visit_counter])
    data.edge_type = torch.tensor(edge_types, dtype=torch.long)
    data.edge_concept = edge_index[0]
    data.edge_visit = torch.tensor([v - node_ranges.visit[0] for v in visit_idx_list], dtype=torch.long)

    struct_table = load_embedding_table(struct_emb_path)
    text_table = load_embedding_table(text_emb_path)
    logic_table = load_embedding_table(logic_emb_path)

    node_ids_for_struct = [idx for idx in range(total_nodes)]
    struct_feat, _ = as_feature_matrix(node_ids_for_struct, struct_table, None)
    text_keys = concept_strings + [f'visit_{i}' for i in range(visit_counter)]
    logic_keys = concept_strings + [f'visit_{i}' for i in range(visit_counter)]
    text_feat, _ = as_feature_matrix(text_keys, text_table, None)
    if text_pca_dim is not None and text_feat is not None and text_feat.size(1) > text_pca_dim:
        text_feat = reduce_with_pca(text_feat, text_pca_dim)
    logic_feat, _ = as_feature_matrix(logic_keys, logic_table, None)

    data.struct_feat = struct_feat
    data.text_feat = text_feat
    data.logic_feat = logic_feat

    data.node_type = torch.tensor([
        _concept_type(i, node_ranges) for i in range(total_nodes)
    ], dtype=torch.long)
    data.node_code = concept_strings + [f'visit_{i}' for i in range(visit_counter)]

    data.num_nodes = total_nodes
    data.x = data.struct_feat.clone()
    data.visit_node_offset = torch.tensor(node_ranges.visit[0])
    data.num_visits = torch.tensor([visit_counter])
    data.edge_mask_template = torch.ones(num_edges, dtype=torch.float32)

    # Norms for message passing
    data = norm_contruction(data, option='all_one')

    # Relative position encoding
    if hierarchy_path is not None:
        encoder = HierarchyEncoder(hierarchy_path)
        bucketizer = RelativePositionBucketizer(bucket_boundaries or [0, 1, 2, 3, 4, 6])
        data.edge_rel_pos = build_edge_relative_positions(
            visit_summaries=visit_summaries,
            membership_lookup=membership_lookup,
            node_codes=data.node_code,
            hierarchy=encoder,
            num_edges=num_edges,
            bucketizer=bucketizer
        )
    else:
        data.edge_rel_pos = torch.zeros(num_edges, dtype=torch.long)

    return DrugHypergraphArtifacts(
        data=data,
        visit_summaries=visit_summaries,
        node_ranges=node_ranges,
        membership_lookup=membership_lookup,
        vocabs={
            'diag': diag_vocab,
            'proc': proc_vocab,
            'med': med_vocab,
        },
        split_usage=split_usage,
    )
