import argparse
import copy
import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import ipdb
import numpy as np
import torch
import torch.nn.functional as F

from pretraining.data_builder import build_drug_hypergraph
from pretraining.masking import sample_visit_masks
from pretraining.model import DrugHypergraphPretrainer, compute_initializer_output_dim
from pretraining.clustering import DEC, PredictionHead, weighted_cluster_average
from pretraining.models_ext import SetGNNExtended
from recommender.metrics import ddi_rate_score, multi_label_metric


def build_gnn_args(cli_args, feature_dim: int, hidden_dim: int):
    return argparse.Namespace(
        All_num_layers=cli_args.num_layers,
        dropout=cli_args.dropout,
        aggregate=cli_args.aggregate,
        normalization=cli_args.normalization,
        LearnFeat=False,
        MLP_hidden=hidden_dim,
        MLP_num_layers=cli_args.mlp_layers,
        Classifier_hidden=hidden_dim,
        Classifier_num_layers=2,
        feature_dim=feature_dim,
        heads=cli_args.heads,
        PMA=True,
        num_features=feature_dim,
        num_labels=1,
    )


def mask_ratio_dict(cli_args) -> Dict[str, float]:
    return {
        'diag': cli_args.mask_ratio_diag,
        'proc': cli_args.mask_ratio_proc,
        'med': cli_args.mask_ratio_med,
    }


def build_terminal_supervision(visit_summaries, med_range: Tuple[int, int], device: torch.device):
    start, length = med_range
    indices: List[int] = []
    targets: List[torch.Tensor] = []
    for visit in visit_summaries:
        if not getattr(visit, 'is_terminal', False):
            continue
        med_ids = [node_id - start for node_id in visit.med_nodes if start <= node_id < start + length]
        if not med_ids:
            continue
        target = torch.zeros(length, device=device)
        target[med_ids] = 1.0
        indices.append(visit.visit_index)
        targets.append(target)
    if not targets:
        return torch.empty(0, dtype=torch.long, device=device), torch.empty(0, length, device=device)
    return torch.tensor(indices, dtype=torch.long, device=device), torch.stack(targets)


def build_global_mask_plan(plans,
                           num_nodes: int,
                           mask_token_prob: float,
                           mask_random_prob: float,
                           sample_same_type,
                           bernoulli_rng,
                           device: torch.device) -> Dict[str, torch.Tensor]:
    mask_tensor = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    random_targets = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
    for plan in plans:
        for node_id in plan.diag_nodes + plan.proc_nodes + plan.med_nodes:
            coin = bernoulli_rng()
            if coin < mask_token_prob:
                mask_tensor[node_id] = True
            elif coin < mask_token_prob + mask_random_prob:
                random_targets[node_id] = sample_same_type(node_id)
    return {
        'mask_token': mask_tensor,
        'random_targets': random_targets,
    }


def setup_logger(log_dir: Path) -> logging.Logger:
    logger = logging.getLogger('drug_pretraining')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(log_dir / 'run.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def log_data_snapshot(logger: logging.Logger, artifacts, data, fused_dim: int, project_features: bool):
    diag_start, diag_len = artifacts.node_ranges.diag
    proc_start, proc_len = artifacts.node_ranges.proc
    med_start, med_len = artifacts.node_ranges.med
    struct_dim = data.struct_feat.size(1) if data.struct_feat is not None else 0
    text_dim = data.text_feat.size(1) if data.text_feat is not None else 0
    logic_dim = data.logic_feat.size(1) if data.logic_feat is not None else 0
    stats = {
        'visits': len(artifacts.visit_summaries),
        'diag_nodes': diag_len,
        'proc_nodes': proc_len,
        'med_nodes': med_len,
        'total_nodes': data.num_nodes,
        'struct_dim': struct_dim,
        'text_dim': text_dim,
        'logic_dim': logic_dim,
        'fused_feature_dim': fused_dim,
        'project_node_embeds': project_features,
    }
    logger.info('Loaded hypergraph statistics: %s', stats)


def summarize_metrics(y_gt: List[np.ndarray],
                      y_pred: List[np.ndarray],
                      y_prob: List[np.ndarray],
                      smm_record: List[List[List[int]]],
                      ddi_path: str) -> Tuple[float, Dict[str, float]]:
    if not y_gt:
        zeros = {'ddi': 0.0, 'ja': 0.0, 'prauc': 0.0, 'avg_p': 0.0, 'avg_r': 0.0, 'avg_f1': 0.0}
        return 0.0, zeros
    y_gt_arr = np.concatenate(y_gt, axis=0)
    y_pred_arr = np.concatenate(y_pred, axis=0)
    y_prob_arr = np.concatenate(y_prob, axis=0)
    ja, prauc, avg_p, avg_r, avg_f1 = multi_label_metric(y_gt_arr, y_pred_arr, y_prob_arr)
    ddi = ddi_rate_score(smm_record, path=ddi_path)
    metrics = {
        'ddi': ddi,
        'ja': ja,
        'prauc': prauc,
        'avg_p': avg_p,
        'avg_r': avg_r,
        'avg_f1': avg_f1,
    }
    return ja, metrics


def build_med_label_matrix(visits, med_range: Tuple[int, int]) -> np.ndarray:
    start, length = med_range
    if not visits:
        return np.zeros((0, length), dtype=np.float32)
    labels = []
    for visit in visits:
        target = np.zeros(length, dtype=np.float32)
        for node_id in getattr(visit, 'med_nodes', []):
            rel_idx = node_id - start
            if 0 <= rel_idx < length:
                target[rel_idx] = 1.0
        labels.append(target)
    return np.stack(labels, axis=0) if labels else np.zeros((0, length), dtype=np.float32)


def prepare_terminal_split(visits, med_range: Tuple[int, int], device: torch.device):
    if not visits:
        return None
    terminal_visits = [visit for visit in visits if getattr(visit, 'is_terminal', False)]
    if not terminal_visits:
        return None
    indices = torch.tensor([visit.visit_index for visit in terminal_visits], dtype=torch.long, device=device)
    targets = build_med_label_matrix(terminal_visits, med_range)
    return {'indices': indices, 'targets': targets}


def evaluate_medication_split(model: DrugHypergraphPretrainer,
                              visit_emb: torch.Tensor,
                              visit_indices: torch.Tensor,
                              target_matrix: np.ndarray,
                              ddi_path: str) -> Tuple[float, Dict[str, float]]:
    if visit_indices is None or visit_indices.numel() == 0 or target_matrix.size == 0:
        zeros = {'ddi': 0.0, 'ja': 0.0, 'prauc': 0.0, 'avg_p': 0.0, 'avg_r': 0.0, 'avg_f1': 0.0}
        return 0.0, zeros
    with torch.no_grad():
        subset = visit_emb.index_select(0, visit_indices)
        logits = model.visit_head.med(subset)
        prob = torch.sigmoid(logits).cpu().numpy()
    pred = (prob >= 0.5).astype(np.float32)
    smm_record = [[np.where(sample == 1)[0].tolist()] for sample in pred]
    return summarize_metrics([target_matrix], [pred], [prob], smm_record, ddi_path)


def compute_split_metrics(model: DrugHypergraphPretrainer,
                          data,
                          num_visits: int,
                          eval_splits: Dict[str, Dict[str, Any]],
                          ddi_path: str) -> Dict[str, Dict[str, float]]:
    if not eval_splits:
        return {}
    was_training = model.training
    metrics = {}
    with torch.no_grad():
        model.eval()
        _, edge_feat_eval, _ = model(data)
        visit_emb_eval = edge_feat_eval[:num_visits]
        for split_name, info in eval_splits.items():
            _, split_metrics = evaluate_medication_split(
                model,
                visit_emb_eval,
                info['indices'],
                info['targets'],
                ddi_path,
            )
            metrics[split_name] = split_metrics
    if was_training:
        model.train()
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Drug recommendation masked-visit pretraining')
    parser.add_argument('--records-path', default='data/MIMIC-III/records_final.pkl')
    parser.add_argument('--voc-path', default='data/MIMIC-III/voc_final.pkl')
    parser.add_argument('--struct-emb-path', required=True)
    parser.add_argument('--text-emb-path')
    parser.add_argument('--logic-emb-path')
    parser.add_argument('--hierarchy-path')
    parser.add_argument('--warmup1-epochs', type=int, default=100,
                        help='Number of epochs for masked visit warmup (non-terminal visits).')
    parser.add_argument('--warmup2-epochs', type=int, default=50,
                        help='Number of epochs for terminal-visit medication supervision before clustering kicks in.')
    parser.add_argument('--epochs', type=int, default=150,
                        help='Number of epochs for the full objective (medication + clustering + contrast).')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--feature-dim', type=int, default=64,
                        help='Dimension of linear projections for struct/text/logic embeddings when projection is enabled')
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--mlp-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--aggregate', default='mean', choices=['mean', 'sum'])
    parser.add_argument('--normalization', default='ln', choices=['ln', 'bn', 'None'])
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--num-clusters', type=int, default=25)
    parser.add_argument('--mask-ratio-diag', type=float, default=0.2)
    parser.add_argument('--mask-ratio-proc', type=float, default=0.2)
    parser.add_argument('--mask-ratio-med', type=float, default=0.2)
    parser.add_argument('--mask-mode', choices=['vanilla', 'bert'], default='bert')
    parser.add_argument('--mask-token-prob', type=float, default=0.8)
    parser.add_argument('--mask-random-prob', type=float, default=0.1)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--log-dir', default='pretrain_logs')
    parser.add_argument('--ddi-path', default='data/MIMIC-III/ddi_A_final.pkl',
                        help='Path to the precomputed DDI adjacency matrix for metric reporting.')
    parser.add_argument('--save-checkpoint', action='store_true')
    parser.add_argument('--project-node-embeds', dest='project_node_embeds', action='store_true',
                        help='Apply linear projections to node embeddings before fusion (default)')
    parser.add_argument('--no-project-node-embeds', dest='project_node_embeds', action='store_false',
                        help='Skip linear projections and concatenate raw embeddings directly')
    parser.add_argument('--text-pca-dim', type=int, default=None,
                        help='Reduce text embeddings to this dimension via PCA before training (defaults to feature-dim when text embeddings provided)')
    parser.add_argument('--embedding-init-range', type=float, default=None,
                        help='If set, reinitialize every nn.Embedding weight uniformly within [-value, value].')
    parser.add_argument('--cluster-weight', type=float, default=1.0,
                        help='Weight applied to the DEC clustering losses during the full objective.')
    parser.add_argument('--alignment-weight', type=float, default=0.1,
                        help='Weight applied to the node-visit alignment/contrastive head during the full objective.')
    parser.add_argument('--tsne-interval', type=int, default=0,
                        help='If > 0, run t-SNE visualization every N epochs during the full objective stage.')
    parser.add_argument('--tsne-max-samples', type=int, default=2000,
                        help='Maximum number of samples per t-SNE plot (set <=0 to disable subsampling).')
    parser.add_argument('--tsne-output-dir', type=str, default=None,
                        help='Directory to store t-SNE plots (defaults to <log-dir>/tsne).')
    parser.add_argument('--warmup1-best-path', type=str, default=None,
                        help='Path to store the best Warmup1 checkpoint (defaults to <log-dir>/warmup1_best.pt).')
    parser.add_argument('--warmup2-patience', type=int, default=5,
                        help='Early-stopping patience (in epochs) for Warmup2 based on eval JA; <=0 disables early stop.')
    parser.add_argument('--warmup2-best-path', type=str, default=None,
                        help='Path to store the best Warmup2 checkpoint (defaults to <log-dir>/warmup2_best.pt).')
    parser.add_argument('--warmup2-init-checkpoint', type=str, default=None,
                        help='Optional checkpoint to load before Phase 3 (overrides freshly saved Warmup2 weights).')
    parser.add_argument('--full-patience', type=int, default=0,
                        help='Early-stopping patience (in epochs) for the full objective based on eval JA; <=0 disables.')
    parser.add_argument('--full-best-path', type=str, default=None,
                        help='Path to store the best full-objective checkpoint (defaults to <log-dir>/full_best.pt).')
    parser.add_argument('--full-init-checkpoint', type=str, default=None,
                        help='Checkpoint to load before starting the full objective (useful for hyper-parameter sweeps).')
    parser.set_defaults(project_node_embeds=True)
    args = parser.parse_args()

    if args.text_pca_dim is None and args.text_emb_path is not None:
        args.text_pca_dim = args.feature_dim

    base_log_dir = Path(args.log_dir)
    run_name = f"nc{args.num_clusters}_h{args.heads}_fd{args.feature_dim}_clw{args.cluster_weight}_alw{args.alignment_weight}"
    log_dir = base_log_dir / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_dir)
    logger.info('Starting pretraining with args: %s', vars(args))
    tsne_dir = Path(args.tsne_output_dir) if args.tsne_output_dir else log_dir / 'tsne'
    warmup2_best_path = Path(args.warmup2_best_path) if args.warmup2_best_path else log_dir / 'warmup2_best.pt'
    warmup1_best_path = Path(args.warmup1_best_path) if args.warmup1_best_path else log_dir / 'warmup1_best.pt'

    full_best_path = Path(args.full_best_path) if args.full_best_path else log_dir / 'full_best.pt'

    artifacts = build_drug_hypergraph(
        records_path=args.records_path,
        voc_path=args.voc_path,
        struct_emb_path=args.struct_emb_path,
        text_emb_path=args.text_emb_path,
        logic_emb_path=args.logic_emb_path,
        hierarchy_path=args.hierarchy_path,
        text_pca_dim=args.text_pca_dim,
    )
    data = artifacts.data
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu')
    data = data.to(device)
    use_text = args.text_emb_path is not None
    use_logic = args.logic_emb_path is not None
    fused_dim = compute_initializer_output_dim(
        struct_dim=data.struct_feat.size(1),
        text_dim=data.text_feat.size(1),
        logic_dim=data.logic_feat.size(1),
        projection_dim=args.feature_dim,
        use_text=use_text,
        use_logic=use_logic,
        project_inputs=args.project_node_embeds,
    )
    log_data_snapshot(logger, artifacts, data, fused_dim, args.project_node_embeds)

    split_to_visits: Dict[str, List] = {}
    for visit in artifacts.visit_summaries:
        split_to_visits.setdefault(visit.split, []).append(visit)
    train_visit_summaries = split_to_visits.get('train')
    if not train_visit_summaries:
        train_visit_summaries = split_to_visits.get('full', artifacts.visit_summaries)
        if train_visit_summaries is artifacts.visit_summaries:
            logger.warning('No explicit train split detected; defaulting supervised phases to all visits.')
    eval_visit_summaries = split_to_visits.get('eval')

    train_visit_indices = [visit.visit_index for visit in train_visit_summaries]
    train_visit_index_tensor = torch.tensor(train_visit_indices, dtype=torch.long, device=device) if train_visit_summaries else torch.empty(0, dtype=torch.long, device=device)

    train_concept_ids: Set[int] = set()
    for visit in train_visit_summaries:
        train_concept_ids.update(visit.diag_nodes)
        train_concept_ids.update(visit.proc_nodes)
        train_concept_ids.update(visit.med_nodes)
    if train_concept_ids:
        train_concept_index_tensor = torch.tensor(sorted(train_concept_ids), dtype=torch.long, device=device)
    else:
        train_concept_index_tensor = torch.empty(0, dtype=torch.long, device=device)

    logger.info(
        'Supervised phases use %d train visits and %d unique concept nodes',
        len(train_visit_summaries),
        len(train_concept_ids),
    )

    eval_splits: Dict[str, Dict[str, Any]] = {}
    train_terminal_info = prepare_terminal_split(train_visit_summaries, artifacts.node_ranges.med, device)
    if train_terminal_info:
        eval_splits['train'] = train_terminal_info
    eval_terminal_info = prepare_terminal_split(eval_visit_summaries, artifacts.node_ranges.med, device)
    if eval_terminal_info:
        eval_splits['eval'] = eval_terminal_info
    if eval_splits:
        terminal_counts = {split: int(info['indices'].numel()) for split, info in eval_splits.items()}
        logger.info('Terminal evaluation splits prepared: %s', terminal_counts)
    else:
        logger.info('No terminal visit splits available for metric reporting.')
    has_eval_split = 'eval' in eval_splits

    stage1_config = {
        'version': 1,
        'feature_dim': args.feature_dim,
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'mlp_layers': args.mlp_layers,
        'dropout': args.dropout,
        'aggregate': args.aggregate,
        'normalization': args.normalization,
        'heads': args.heads,
        'text_pca_dim': args.text_pca_dim,
        'project_node_embeds': args.project_node_embeds,
        'use_text': use_text,
        'use_logic': use_logic,
        'warmup1_epochs': args.warmup1_epochs,
        'warmup2_epochs': args.warmup2_epochs,
        'full_epochs': args.epochs,
        'num_clusters': args.num_clusters,
        'cluster_weight': args.cluster_weight,
        'alignment_weight': args.alignment_weight,
    }
    with open(log_dir / 'stage1_config.json', 'w', encoding='utf-8') as f:
        json.dump(stage1_config, f, indent=2)

    diag_size = len(artifacts.vocabs['diag'])
    proc_size = len(artifacts.vocabs['proc'])
    med_size = len(artifacts.vocabs['med'])
    rel_pos_buckets = int(data.edge_rel_pos.max().item()) + 1

    gnn_args = build_gnn_args(args, fused_dim, args.hidden_dim)
    encoder = SetGNNExtended(gnn_args, data)
    model = DrugHypergraphPretrainer(
        encoder=encoder,
        node_ranges=artifacts.node_ranges,
        struct_dim=data.struct_feat.size(1),
        text_dim=data.text_feat.size(1),
        logic_dim=data.logic_feat.size(1),
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        diag_size=diag_size,
        proc_size=proc_size,
        med_size=med_size,
        rel_pos_buckets=rel_pos_buckets,
        heads=args.heads,
        dropout=args.dropout,
        use_text=use_text,
        use_logic=use_logic,
        project_node_embeds=args.project_node_embeds,
        embedding_init_range=args.embedding_init_range,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history_path = log_dir / 'history.jsonl'

    num_visits = len(artifacts.visit_summaries)
    num_nodes = data.num_nodes

    diag_start, diag_len = artifacts.node_ranges.diag
    proc_start, proc_len = artifacts.node_ranges.proc
    med_start, med_len = artifacts.node_ranges.med
    diag_ids = list(range(diag_start, diag_start + diag_len))
    proc_ids = list(range(proc_start, proc_start + proc_len))
    med_ids = list(range(med_start, med_start + med_len))
    type_pools = {
        0: diag_ids,
        1: proc_ids,
        2: med_ids,
    }

    mask_token_prob = args.mask_token_prob if args.mask_mode == 'bert' else 1.0
    mask_random_prob = args.mask_random_prob if args.mask_mode == 'bert' else 0.0
    if mask_token_prob + mask_random_prob > 1.0:
        raise ValueError('mask-token-prob + mask-random-prob must be <= 1.0')

    def sample_same_type(node_id: int) -> int:
        node_type = int(data.node_type[node_id].item())
        pool = type_pools.get(node_type, [])
        if len(pool) <= 1:
            return node_id
        candidate = random.choice(pool)
        while candidate == node_id:
            candidate = random.choice(pool)
        return candidate

    terminal_indices, terminal_targets = build_terminal_supervision(
        train_visit_summaries,
        artifacts.node_ranges.med,
        device,
    )

    def medication_loss(visit_emb: torch.Tensor) -> torch.Tensor:
        if terminal_indices.numel() == 0:
            return visit_emb.new_zeros(())
        med_logits = model.visit_head.med(visit_emb[terminal_indices])
        return F.binary_cross_entropy_with_logits(med_logits, terminal_targets)

    def log_history(stage: str, epoch: int, metrics: Dict[str, float]) -> None:
        entry = {'stage': stage, 'epoch': epoch}
        entry.update(metrics)
        with open(history_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry) + '\n')

    warmup2_patience = max(0, args.warmup2_patience)
    warmup2_best_eval = float('-inf')
    warmup2_best_state: Optional[Dict[str, torch.Tensor]] = None
    warmup2_best_epoch = 0
    warmup2_epochs_no_gain = 0
    if warmup2_patience > 0 and not has_eval_split:
        logger.warning('Warmup2 early stopping disabled because eval split is unavailable.')

    full_patience = max(0, args.full_patience)
    full_best_eval = float('-inf')
    full_best_state: Optional[Dict[str, torch.Tensor]] = None
    full_best_epoch = 0
    full_epochs_no_gain = 0
    if full_patience > 0 and not has_eval_split:
        logger.warning('Full objective early stopping disabled because eval split is unavailable.')

    def load_weights_into_model(path: Path, description: str) -> bool:
        if path is None or not path.exists():
            return False
        payload = torch.load(path, map_location=device)
        state_dict = payload.get('model', payload)
        model.load_state_dict(state_dict)
        logger.info('Loaded %s weights from %s', description, path)
        return True
    
    # ipdb.set_trace()
    if os.path.exists(warmup1_best_path):
        print("No Warmup1 best path found, skipping weight loading.")
        load_weights_into_model(warmup1_best_path, 'Warmup1 best')

    # Phase 1: masked modeling on non-terminal visits
    for epoch in range(1, args.warmup1_epochs + 1):
        model.eval()
        plans = sample_visit_masks(artifacts.visit_summaries, mask_ratio_dict(args), seed=epoch)
        if not plans:
            continue
        optimizer.zero_grad()
        rng = random.Random(epoch)
        mask_plan = build_global_mask_plan(
            plans,
            num_nodes,
            mask_token_prob,
            mask_random_prob,
            sample_same_type,
            rng.random,
            device,
        )
        _, edge_feat, _ = model(data, mask_plan=mask_plan)
        visit_emb = edge_feat[:num_visits]
        loss, stats = model.masked_loss(visit_emb, plans, device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        acc = (stats['correct'] / stats['total']) if stats['total'] > 0 else 0.0
        log_history('warmup1', epoch, {'loss': loss.item(), 'mask_acc': acc})
        logger.info('[Warmup1][Epoch %d] loss=%.4f mask-acc=%.4f', epoch, loss.item(), acc)

    # Phase 2: terminal-visit medication supervision
    for epoch in range(1, args.warmup2_epochs + 1):
        if terminal_indices.numel() == 0:
            break
        model.train()
        optimizer.zero_grad()
        _, edge_feat, _ = model(data)
        visit_emb = edge_feat[:num_visits]
        loss = medication_loss(visit_emb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        split_metrics = compute_split_metrics(model, data, num_visits, eval_splits, args.ddi_path)
        history_payload = {'med_loss': loss.item()}
        log_components = [f'[Warmup2][Epoch {epoch}] med-loss={loss.item():.4f}']
        for split_name, metrics_dict in split_metrics.items():
            history_payload[f'{split_name}_ja'] = metrics_dict['ja']
            history_payload[f'{split_name}_ddi'] = metrics_dict['ddi']
            log_components.append(f"{split_name}-ja={metrics_dict['ja']:.4f}")
            log_components.append(f"{split_name}-ddi={metrics_dict['ddi']:.4f}")
        log_history('warmup2', epoch, history_payload)
        logger.info(' '.join(log_components))
        if has_eval_split and warmup2_patience > 0:
            eval_metrics = split_metrics.get('eval')
            eval_ja = eval_metrics.get('ja') if eval_metrics else None
            if eval_ja is not None and eval_ja > warmup2_best_eval + 1e-6:
                warmup2_best_eval = eval_ja
                warmup2_best_epoch = epoch
                warmup2_epochs_no_gain = 0
                warmup2_best_state = copy.deepcopy(model.state_dict())
                torch.save({'model': warmup2_best_state, 'epoch': epoch, 'eval_ja': eval_ja}, warmup2_best_path)
                logger.info('Warmup2 eval JA improved to %.4f at epoch %d. Saved %s', eval_ja, epoch, warmup2_best_path)
            elif eval_ja is not None:
                warmup2_epochs_no_gain += 1
                if warmup2_epochs_no_gain >= warmup2_patience:
                    logger.info('Warmup2 early stopping triggered after %d epochs without eval JA improvement (best %.4f @ epoch %d).',
                                warmup2_patience, warmup2_best_eval, warmup2_best_epoch)
                    break
        
    # ipdb.set_trace()
    init_ckpt_path = Path(args.warmup2_init_checkpoint) if args.warmup2_init_checkpoint else None
    if init_ckpt_path and load_weights_into_model(init_ckpt_path, 'Warmup2 init checkpoint'):
        pass
    elif warmup2_best_state is not None:
        model.load_state_dict(warmup2_best_state)
        logger.info('Loaded in-memory Warmup2 best state (eval JA=%.4f @ epoch %d).', warmup2_best_eval, warmup2_best_epoch)
    else:
        if not os.path.exists(warmup2_best_path):
            print("No Warmup2 best path found, skipping weight loading.")
            exit(1)
        load_weights_into_model(warmup2_best_path, 'Warmup2 best')

    full_init_ckpt_path = Path(args.full_init_checkpoint) if args.full_init_checkpoint else None
    if full_init_ckpt_path:
        load_weights_into_model(full_init_ckpt_path, 'Full init checkpoint')


    # Phase 3: full objective (medication + clustering + alignment)
    node_cluster = DEC(num_cluster=args.num_clusters, feat_dim=args.hidden_dim).to(device)
    visit_cluster = DEC(num_cluster=args.num_clusters, feat_dim=args.hidden_dim).to(device)
    alignment_head = PredictionHead(
        input_dim=args.hidden_dim,
        hidden_dim=args.hidden_dim * 4,
        output_dim=args.hidden_dim,
    ).to(device)
    cluster_params = list(node_cluster.parameters()) + list(visit_cluster.parameters()) + list(alignment_head.parameters())
    cluster_optimizer = torch.optim.Adam(cluster_params, lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        cluster_optimizer.zero_grad()
        _, edge_feat, node_feat = model(data)
        visit_emb = edge_feat[:num_visits]
        concept_emb = node_feat[:artifacts.node_ranges.concept_count]

        med_loss = medication_loss(visit_emb)

        if train_concept_index_tensor.numel() > 0:
            concept_subset = concept_emb.index_select(0, train_concept_index_tensor)
            node_cluster_loss = node_cluster.loss(concept_subset, epoch - 1)
            node_Q = node_cluster.get_Q()
        else:
            concept_subset = concept_emb.new_zeros((0, concept_emb.size(1)))
            node_cluster_loss = concept_emb.new_zeros(())
            node_Q = None

        if train_visit_index_tensor.numel() > 0:
            visit_subset = visit_emb.index_select(0, train_visit_index_tensor)
            visit_cluster_loss = visit_cluster.loss(visit_subset, epoch - 1)
            visit_Q = visit_cluster.get_Q()
        else:
            visit_subset = visit_emb.new_zeros((0, visit_emb.size(1)))
            visit_cluster_loss = visit_emb.new_zeros(())
            visit_Q = None

        node_assignments = torch.argmax(node_Q, dim=1) if node_Q is not None else None
        visit_assignments = torch.argmax(visit_Q, dim=1) if visit_Q is not None else None

        if node_Q is not None and visit_Q is not None:
            node_cluster_feat, visit_cluster_feat = weighted_cluster_average(node_Q, concept_subset, visit_Q, visit_subset)
            align_loss = alignment_head.build_loss(node_cluster_feat, visit_cluster_feat)
        else:
            align_loss = visit_emb.new_zeros(())

        total_loss = med_loss
        total_loss = total_loss + args.cluster_weight * (node_cluster_loss + visit_cluster_loss)
        total_loss = total_loss + args.alignment_weight * align_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        cluster_optimizer.step()
        split_metrics = compute_split_metrics(model, data, num_visits, eval_splits, args.ddi_path)
        history_payload = {
            'total_loss': total_loss.item(),
            'med_loss': med_loss.item(),
            'node_cluster_loss': node_cluster_loss.item(),
            'visit_cluster_loss': visit_cluster_loss.item(),
            'align_loss': align_loss.item(),
        }
        log_components = [
            f'[Full][Epoch {epoch}] total={total_loss.item():.4f}',
            f'med={med_loss.item():.4f}',
            f'nodeC={node_cluster_loss.item():.4f}',
            f'visitC={visit_cluster_loss.item():.4f}',
            f'align={align_loss.item():.4f}',
        ]
        for split_name, metrics_dict in split_metrics.items():
            history_payload[f'{split_name}_ja'] = metrics_dict['ja']
            history_payload[f'{split_name}_ddi'] = metrics_dict['ddi']
            log_components.append(f"{split_name}-ja={metrics_dict['ja']:.4f}")
            log_components.append(f"{split_name}-ddi={metrics_dict['ddi']:.4f}")
        log_history('full', epoch, history_payload)
        logger.info(' '.join(log_components))

        if has_eval_split and full_patience > 0:
            eval_metrics = split_metrics.get('eval')
            eval_ja = eval_metrics.get('ja') if eval_metrics else None
            if eval_ja is not None and eval_ja > full_best_eval + 1e-6:
                full_best_eval = eval_ja
                full_best_epoch = epoch
                full_epochs_no_gain = 0
                # full_best_state = copy.deepcopy(model.state_dict())
                # torch.save({'model': full_best_state, 'epoch': epoch, 'eval_ja': eval_ja}, full_best_path)
                # logger.info('Full objective eval JA improved to %.4f at epoch %d. Saved %s', eval_ja, epoch, full_best_path)
            elif eval_ja is not None:
                full_epochs_no_gain += 1
                if full_epochs_no_gain >= full_patience:
                    logger.info('Full objective early stopping triggered after %d epochs without eval JA improvement (best %.4f @ epoch %d).',
                                full_patience, full_best_eval, full_best_epoch)
                    break


    if full_best_state is not None:
        model.load_state_dict(full_best_state)
        logger.info('Loaded best full objective state (eval JA=%.4f @ epoch %d).', full_best_eval, full_best_epoch)
    else:
        load_weights_into_model(full_best_path, 'Full best')

    if args.save_checkpoint:
        ckpt = log_dir / 'stage1_final.pt'
        payload = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'config': stage1_config,
            'visit_centroids': visit_cluster.mean.detach().cpu(),
            'node_centroids': node_cluster.mean.detach().cpu(),
        }
        torch.save(payload, ckpt)


if __name__ == '__main__':
    main()
