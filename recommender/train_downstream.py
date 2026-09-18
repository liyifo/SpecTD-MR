from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import dill
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from metrics import ddi_rate_score, multi_label_metric
from pretrain.data_builder import build_drug_hypergraph
from recommender.data import (
    VisitSequenceDataset,
    collate_sequence_batch,
)
from recommender.model import (
    DrugTGCTModel,
    SEQUENCE_ENCODER_CHOICES,
    TGCT_ENCODER,
)
from recommender.pretrain_utils import build_stage1_encoder


DECISION_THRESHOLD = 0.5


STAGE1_CONFIG_KEYS = {
    'feature_dim': 'feature_dim',
    'hidden_dim': 'hidden_dim',
    'num_layers': 'num_layers',
    'mlp_layers': 'mlp_layers',
    'dropout': 'dropout',
    'aggregate': 'aggregate',
    'normalization': 'normalization',
    'heads': 'heads',
    'text_pca_dim': 'text_pca_dim',
    'project_node_embeds': 'project_node_embeds',
}


def load_stage1_checkpoint(checkpoint_path: str) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, Any]], Dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(checkpoint, dict):
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        metadata = checkpoint.get('config')
        extras = {k: v for k, v in checkpoint.items() if k not in {'model', 'config'}}
    else:
        state_dict = checkpoint
        metadata = None
        extras = {}
    return state_dict, metadata, extras


def populate_stage1_args_from_metadata(args: argparse.Namespace,
                                       metadata: Optional[Dict[str, Any]]) -> None:
    if metadata is None:
        raise ValueError('Stage-I checkpoint does not include config metadata; please export a newer checkpoint.')

    missing: List[str] = []
    mismatched: List[str] = []
    for attr, meta_key in STAGE1_CONFIG_KEYS.items():
        if meta_key not in metadata:
            missing.append(f'--{attr.replace("_", "-")}')
            continue

        meta_value = metadata[meta_key]
        current = getattr(args, attr, None)
        if current is not None and current != meta_value:
            mismatched.append(f'{attr}={current} -> {meta_value}')

        setattr(args, attr, meta_value)

    if missing:
        raise ValueError(
            'Stage-I checkpoint is missing config for: ' + ', '.join(missing) +
            '. Please re-save the checkpoint with metadata before running Stage-II.'
        )
    if mismatched:
        print('Warning: overriding CLI Stage-I params with checkpoint metadata: ' + ', '.join(mismatched))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='TGCT downstream finetuning for drug recommendation',
    )
    # data paths
    parser.add_argument('--records-path', default='data/MIMIC-III/records_final.pkl')
    parser.add_argument('--voc-path', default='data/MIMIC-III/voc_final.pkl')
    parser.add_argument('--struct-emb-path', required=True)
    parser.add_argument('--text-emb-path', required=True)
    parser.add_argument('--logic-emb-path', required=True)
    parser.add_argument('--hierarchy-path', default=None,
                        help='Optional hierarchy JSON/PKL; omit to disable relative position bias.')
    parser.add_argument('--pretrain-checkpoint', required=True)
    parser.add_argument('--ddi-path', default='data/MIMIC-III/ddi_A_final.pkl')
    parser.add_argument('--concept-cooccurrence-path', default='data/MIMIC-III/concept_cooccurrence.pkl')
    parser.add_argument('--text-pca-dim', type=int, default=None,
                        help='Optional PCA dimension for text embeddings; defaults to feature-dim when text embeddings are provided.')

    # Stage-I hypergraph params
    parser.add_argument('--feature-dim', type=int, default=None)
    parser.add_argument('--hidden-dim', type=int, default=None)
    parser.add_argument('--num-layers', type=int, default=None)
    parser.add_argument('--mlp-layers', type=int, default=None)
    parser.add_argument('--dropout', type=float, default=None)
    parser.add_argument('--aggregate', default=None, choices=['mean', 'sum'])
    parser.add_argument('--normalization', default=None, choices=['ln', 'bn', 'None'])
    parser.add_argument('--heads', type=int, default=None)
    parser.add_argument('--project-node-embeds', dest='project_node_embeds', action='store_true',
                        help='Force-enable Stage-I projection layers (overrides checkpoint metadata).')
    parser.add_argument('--no-project-node-embeds', dest='project_node_embeds', action='store_false',
                        help='Force-disable Stage-I projection layers.')
    parser.set_defaults(project_node_embeds=None)

    # Sequence encoder params
    parser.add_argument('--seq-hidden-dim', type=int, default=128)
    parser.add_argument(
        '--sequence-encoder',
        choices=SEQUENCE_ENCODER_CHOICES,
        default=TGCT_ENCODER,
        help='Stage-II TGCT recommendation encoder.',
    )
    parser.add_argument('--seq-dropout', type=float, default=0.1)

    # optimization
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--lr-stage1', type=float, default=1e-5,
                        help='Learning rate applied to the embedded Stage-I encoder; set to 0 to freeze.')
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--early-stopping-patience', type=int, default=0,
                        help='Stop after this many epochs without validation Jaccard improvement; <=0 disables.')
    parser.add_argument('--lr-scheduler-patience', type=int, default=0,
                        help='Reduce learning rates after this many stagnant validation epochs; <=0 disables.')
    parser.add_argument('--lr-scheduler-factor', type=float, default=0.5,
                        help='Multiplicative learning-rate reduction factor used by the validation scheduler.')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=42,
                        help='Fallback seed used when an independent seed is omitted.')
    parser.add_argument('--python-seed', type=int, default=None)
    parser.add_argument('--numpy-seed', type=int, default=None)
    parser.add_argument('--torch-seed', type=int, default=None)
    parser.add_argument('--log-dir', default='downstream_logs')
    parser.add_argument('--bootstrap-rounds', type=int, default=10)
    parser.add_argument('--bootstrap-ratio', type=float, default=0.8)
    return parser


def to_device(batch: Dict, device: torch.device) -> Dict:
    tensor_keys = [
        'targets',
        'patient_index',
        'visit_orders',
        'visit_times',
        'visit_node_masks',
        'visit_lengths',
        'node_ids',
        'node_masks',
        'node_types',
        'node_lengths',
        'prior_prob',
        'history_medications',
    ]
    for key in tensor_keys:
        if key in batch:
            batch[key] = batch[key].to(device)
    return batch


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


def compute_multilabel_loss(logits: torch.Tensor,
                            targets: torch.Tensor) -> torch.Tensor:
    pos_weight = logits.new_tensor(1.75)
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)


def evaluate(model: DrugTGCTModel,
             dataloader: DataLoader,
             device: torch.device,
             ddi_path: str) -> Tuple[float, Dict[str, float]]:
    model.eval()
    y_gt, y_pred, y_prob = [], [], []
    smm_record: List[List[List[int]]] = []
    with torch.no_grad():
        for batch in dataloader:
            batch = to_device(batch, device)
            logits = model(batch)
            prob = torch.sigmoid(logits).cpu().numpy()
            pred = (prob >= DECISION_THRESHOLD).astype(np.float32)
            y_gt.append(batch['targets'].cpu().numpy())
            y_pred.append(pred)
            y_prob.append(prob)
            for sample in pred:
                meds = np.where(sample == 1)[0].tolist()
                smm_record.append([meds])
    return summarize_metrics(y_gt, y_pred, y_prob, smm_record, ddi_path)


def bootstrap_evaluate(model: DrugTGCTModel,
                       dataset: VisitSequenceDataset,
                       args,
                       device: torch.device) -> List[Dict[str, float]]:
    results = []
    total = len(dataset)
    sample_size = max(1, int(total * args.bootstrap_ratio))
    for _ in range(args.bootstrap_rounds):
        indices = np.random.choice(total, sample_size, replace=True)
        loader = DataLoader(Subset(dataset, indices.tolist()),
                            batch_size=args.batch_size,
                            shuffle=False,
                            collate_fn=collate_sequence_batch)
        _, metrics = evaluate(model, loader, device, args.ddi_path)
        results.append(metrics)
    return results


def aggregate_bootstrap_metrics(test_metrics: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    aggregated = {}
    for key in test_metrics[0].keys():
        values = np.array([m[key] for m in test_metrics])
        aggregated[key] = {'mean': float(values.mean()), 'std': float(values.std())}
    return aggregated


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    args.python_seed = args.seed if args.python_seed is None else args.python_seed
    args.numpy_seed = args.seed if args.numpy_seed is None else args.numpy_seed
    args.torch_seed = args.seed if args.torch_seed is None else args.torch_seed
    random.seed(args.python_seed)
    np.random.seed(args.numpy_seed)
    torch.manual_seed(args.torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.torch_seed)
    stage1_state_dict, stage1_metadata, _ = load_stage1_checkpoint(args.pretrain_checkpoint)
    populate_stage1_args_from_metadata(args, stage1_metadata)
    stage1_use_text = stage1_metadata.get('use_text') if stage1_metadata else None
    stage1_use_logic = stage1_metadata.get('use_logic') if stage1_metadata else None
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu')
    if args.hierarchy_path is None:
        print('Hierarchy path not provided; relative position bias will be disabled for Stage-I/II graphs.')
    if args.text_pca_dim is None and args.text_emb_path:
        if args.feature_dim is None:
            raise ValueError('feature_dim 未解析，无法推断 text_pca_dim。请检查 Stage-I 配置。')
        args.text_pca_dim = args.feature_dim
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    run_config = vars(args).copy()
    run_config['decision_threshold'] = DECISION_THRESHOLD
    with open(log_dir / 'run_config.json', 'w', encoding='utf-8') as f:
        json.dump(run_config, f, indent=2)

    stage1_model, pretrain_artifacts, visit_lookup = build_stage1_encoder(
        records_path=args.records_path,
        voc_path=args.voc_path,
        struct_emb_path=args.struct_emb_path,
        text_emb_path=args.text_emb_path,
        logic_emb_path=args.logic_emb_path,
        hierarchy_path=args.hierarchy_path,
        text_pca_dim=args.text_pca_dim,
        checkpoint_path=args.pretrain_checkpoint,
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        aggregate=args.aggregate,
        normalization=args.normalization,
        mlp_layers=args.mlp_layers,
        heads=args.heads,
        project_node_embeds=args.project_node_embeds if args.project_node_embeds is not None else True,
        state_dict=stage1_state_dict,
        # state_dict=None,
        use_text=stage1_use_text,
        use_logic=stage1_use_logic,
        device=device,
    )

    stage1_data = pretrain_artifacts.data.to(device)
    stage1_visit_count = len(pretrain_artifacts.visit_summaries)
    concept_count = pretrain_artifacts.node_ranges.concept_count

    with open(args.concept_cooccurrence_path, 'rb') as f:
        concept_graph = dill.load(f)

    full_artifacts = build_drug_hypergraph(
        records_path=args.records_path,
        voc_path=args.voc_path,
        struct_emb_path=args.struct_emb_path,
        text_emb_path=args.text_emb_path,
        logic_emb_path=args.logic_emb_path,
        hierarchy_path=args.hierarchy_path,
        text_pca_dim=args.text_pca_dim,
        usage='full'
    )

    med_vocab_size = full_artifacts.node_ranges.med[1]
    cooccurrence_matrix = concept_graph['matrix']
    train_dataset = VisitSequenceDataset(
        args.records_path, full_artifacts, 'train', med_vocab_size, cooccurrence_matrix,
    )
    eval_dataset = VisitSequenceDataset(
        args.records_path, full_artifacts, 'eval', med_vocab_size, cooccurrence_matrix,
    )
    test_dataset = VisitSequenceDataset(
        args.records_path, full_artifacts, 'test', med_vocab_size, cooccurrence_matrix,
    )
    print(
        f'Dataset sizes: train={len(train_dataset)} '
        f'eval={len(eval_dataset)} test={len(test_dataset)}'
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_sequence_batch)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_sequence_batch)
    model = DrugTGCTModel(
        stage1_model=stage1_model,
        stage1_data=stage1_data,
        stage1_visit_lookup=visit_lookup,
        concept_count=concept_count,
        stage1_visit_count=stage1_visit_count,
        hidden_dim=args.seq_hidden_dim,
        med_vocab_size=med_vocab_size,
        med_node_start=full_artifacts.node_ranges.med[0],
        concat_graph=False,
        dropout=args.seq_dropout,
        sequence_encoder=args.sequence_encoder,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model parameters: total={total_params:,} trainable={trainable_params:,}')

    stage1_params = list(model.stage1_parameters())
    downstream_params = list(model.downstream_parameters())
    stage1_frozen = args.lr_stage1 <= 0 or not stage1_params
    optimizer_groups = []
    stage1_group_idx: Optional[int]
    downstream_group_idx: int
    if stage1_frozen:
        model.stage1_model.requires_grad_(False)
        stage1_group_idx = None
        downstream_group_idx = 0
        print('Stage-I encoder frozen: gradients disabled and parameters excluded from optimizer.')
    else:
        model.stage1_model.requires_grad_(True)
        optimizer_groups.append({'params': stage1_params, 'lr': args.lr_stage1, 'weight_decay': args.weight_decay})
        stage1_group_idx = 0
        downstream_group_idx = 1
    optimizer_groups.append({'params': downstream_params, 'lr': args.lr, 'weight_decay': args.weight_decay})
    optimizer = torch.optim.AdamW(optimizer_groups)
    if not 0.0 < args.lr_scheduler_factor < 1.0:
        raise ValueError('--lr-scheduler-factor must be between 0 and 1.')
    scheduler = None
    if args.lr_scheduler_patience > 0:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=args.lr_scheduler_factor,
            patience=args.lr_scheduler_patience,
        )
    best_ja = float('-inf')
    epochs_without_improvement = 0
    best_path = log_dir / 'best.pt'
    history_path = log_dir / 'history.jsonl'
    if history_path.exists():
        history_path.unlink()
    for epoch in range(1, args.epochs + 1):
        epoch_train_start = time.time()
        model.train()
        if stage1_frozen:
            model.stage1_model.eval()
        epoch_loss = 0.0
        batches = 0
        train_gt, train_pred, train_prob = [], [], []
        train_smm: List[List[List[int]]] = []
        progress = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}', leave=True)
        for batch in progress:
            batch = to_device(batch, device)
            logits = model(batch)
            bce = compute_multilabel_loss(logits, batch['targets'])
            loss = bce
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            batches += 1
            progress.set_postfix({'loss': loss.item()})

            with torch.no_grad():
                prob = torch.sigmoid(logits).detach().cpu().numpy()
                pred = (prob >= DECISION_THRESHOLD).astype(np.float32)
                train_prob.append(prob)
                train_pred.append(pred)
                train_gt.append(batch['targets'].detach().cpu().numpy())
                for sample in pred:
                    meds = np.where(sample == 1)[0].tolist()
                    train_smm.append([meds])
            train_elapsed = time.time() - epoch_train_start
        avg_loss = epoch_loss / max(1, batches)

        train_ja, train_metrics = summarize_metrics(train_gt, train_pred, train_prob, train_smm, args.ddi_path)
        eval_start = time.time()
        val_ja, metrics = evaluate(model, eval_loader, device, args.ddi_path)
        eval_elapsed = time.time() - eval_start
        stage1_lr = 0.0 if stage1_group_idx is None else optimizer.param_groups[stage1_group_idx]['lr']
        downstream_lr = optimizer.param_groups[downstream_group_idx]['lr']
        history_entry = {
            'epoch': epoch,
            'lr_stage1': stage1_lr,
            'lr_downstream': downstream_lr,
            'train_loss': avg_loss,
                'train_time_sec': train_elapsed,
            'eval_time_sec': eval_elapsed,
        }
        history_entry.update({f'train_{k}': v for k, v in train_metrics.items()})
        history_entry.update({f'val_{k}': v for k, v in metrics.items()})
        with open(history_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(history_entry) + '\n')
        print(
                f"Epoch {epoch}/{args.epochs} | lr_stage1={stage1_lr:.2e} | lr_downstream={downstream_lr:.2e} | "
                f"train_ja={train_metrics['ja']:.4f} | val_ja={metrics['ja']:.4f} | val_ddi={metrics['ddi']:.4f} | "
                f"train_time={train_elapsed:.1f}s | eval_time={eval_elapsed:.1f}s"
        )
        if val_ja > best_ja + 1e-6:
            best_ja = val_ja
            epochs_without_improvement = 0
            torch.save({'model': model.state_dict(), 'epoch': epoch}, best_path)
        else:
            epochs_without_improvement += 1
        if scheduler is not None:
            scheduler.step(val_ja)
        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            print(
                f'Early stopping after {args.early_stopping_patience} epochs without validation Jaccard improvement.'
            )
            break

    print(f'Best validation Jaccard: {best_ja:.4f} (checkpoint={best_path})')
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    test_metrics = bootstrap_evaluate(model, test_dataset, args, device)
    aggregated = aggregate_bootstrap_metrics(test_metrics)
    result = {
        'rounds': test_metrics,
        'aggregate': aggregated,
        'size': len(test_dataset),
    }
    print(f'Bootstrap test metrics (all, n={len(test_dataset)}):')
    for key, value in aggregated.items():
        print(f'{key}: {value["mean"]:.4f} +/- {value["std"]:.4f}')
    with open(log_dir / 'test_bootstrap.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)


if __name__ == '__main__':
    main()
