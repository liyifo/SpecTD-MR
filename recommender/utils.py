from __future__ import annotations
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import torch

from pretraining.data_builder import build_drug_hypergraph
from pretraining.model import (
    DrugHypergraphPretrainer,
    compute_initializer_output_dim,
)
from pretraining.models_ext import SetGNNExtended


def _build_gnn_args(feature_dim: int,
                    hidden_dim: int,
                    num_layers: int,
                    dropout: float,
                    aggregate: str,
                    normalization: str,
                    mlp_layers: int,
                    heads: int):
    return SimpleNamespace(
        All_num_layers=num_layers,
        dropout=dropout,
        aggregate=aggregate,
        normalization=normalization,
        LearnFeat=False,
        MLP_hidden=hidden_dim,
        MLP_num_layers=mlp_layers,
        Classifier_hidden=hidden_dim,
        Classifier_num_layers=2,
        feature_dim=feature_dim,
        heads=heads,
        PMA=True,
        num_features=feature_dim,
        num_labels=1,
    )


def _instantiate_stage1_model(records_path: str,
                              voc_path: str,
                              struct_emb_path: str,
                              text_emb_path: str,
                              logic_emb_path: str,
                              hierarchy_path: Optional[str],
                              text_pca_dim: Optional[int],
                              checkpoint_path: str,
                              feature_dim: int,
                              hidden_dim: int,
                              num_layers: int,
                              dropout: float,
                              aggregate: str,
                              normalization: str,
                              mlp_layers: int,
                              heads: int,
                              project_node_embeds: bool,
                              state_dict: Optional[Dict[str, torch.Tensor]],
                              use_text_override: Optional[bool],
                              use_logic_override: Optional[bool],
                              device: torch.device):
    artifacts = build_drug_hypergraph(
        records_path=records_path,
        voc_path=voc_path,
        struct_emb_path=struct_emb_path,
        text_emb_path=text_emb_path,
        logic_emb_path=logic_emb_path,
        hierarchy_path=hierarchy_path,
        text_pca_dim=text_pca_dim,
        usage='pretrain'
    )
    use_text = use_text_override if use_text_override is not None else text_emb_path is not None
    use_logic = use_logic_override if use_logic_override is not None else logic_emb_path is not None
    gnn_input_dim = compute_initializer_output_dim(
        struct_dim=artifacts.data.struct_feat.size(1),
        text_dim=artifacts.data.text_feat.size(1),
        logic_dim=artifacts.data.logic_feat.size(1),
        projection_dim=feature_dim,
        use_text=use_text,
        use_logic=use_logic,
        project_inputs=project_node_embeds,
    )
    gnn_args = _build_gnn_args(gnn_input_dim, hidden_dim, num_layers, dropout, aggregate, normalization, mlp_layers, heads)
    encoder = SetGNNExtended(gnn_args, artifacts.data)
    model = DrugHypergraphPretrainer(
        encoder=encoder,
        node_ranges=artifacts.node_ranges,
        struct_dim=artifacts.data.struct_feat.size(1),
        text_dim=artifacts.data.text_feat.size(1),
        logic_dim=artifacts.data.logic_feat.size(1),
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        diag_size=len(artifacts.vocabs['diag']),
        proc_size=len(artifacts.vocabs['proc']),
        med_size=len(artifacts.vocabs['med']),
        rel_pos_buckets=int(artifacts.data.edge_rel_pos.max().item()) + 1,
        heads=heads,
        dropout=dropout,
        use_text=use_text,
        use_logic=use_logic,
        project_node_embeds=project_node_embeds,
    ).to(device)
    if state_dict is None:
        checkpoint_state: Any = torch.load(checkpoint_path, map_location=device)
        state_to_load = checkpoint_state['model'] if isinstance(checkpoint_state, dict) and 'model' in checkpoint_state else checkpoint_state
    else:
        state_to_load = state_dict
    model.load_state_dict(state_to_load)
    visit_lookup = {
        (summary.patient_index, summary.order_index): summary.visit_index
        for summary in artifacts.visit_summaries
    }
    return model, artifacts, visit_lookup


def load_stage1_embeddings(records_path: str,
                           voc_path: str,
                           struct_emb_path: str,
                           text_emb_path: str,
                           logic_emb_path: str,
                           hierarchy_path: Optional[str],
                           text_pca_dim: Optional[int],
                           checkpoint_path: str,
                           feature_dim: int,
                           hidden_dim: int,
                           num_layers: int,
                           dropout: float,
                           aggregate: str,
                           normalization: str,
                           mlp_layers: int,
                           heads: int,
                           project_node_embeds: bool,
                           state_dict: Optional[Dict[str, torch.Tensor]],
                           use_text: Optional[bool],
                           use_logic: Optional[bool],
                           device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, Dict[Tuple[int, int], int], object]:
    model, artifacts, visit_lookup = _instantiate_stage1_model(
        records_path=records_path,
        voc_path=voc_path,
        struct_emb_path=struct_emb_path,
        text_emb_path=text_emb_path,
        logic_emb_path=logic_emb_path,
        hierarchy_path=hierarchy_path,
        text_pca_dim=text_pca_dim,
        checkpoint_path=checkpoint_path,
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        aggregate=aggregate,
        normalization=normalization,
        mlp_layers=mlp_layers,
        heads=heads,
        project_node_embeds=project_node_embeds,
        state_dict=state_dict,
        use_text_override=use_text,
        use_logic_override=use_logic,
        device=device,
    )
    model.eval()
    with torch.no_grad():
        _, edge_feat, node_feat = model(artifacts.data.to(device))
    visit_embeddings = edge_feat[:len(artifacts.visit_summaries)].detach().cpu()
    return node_feat.detach().cpu(), visit_embeddings, visit_lookup, artifacts


def build_stage1_encoder(records_path: str,
                         voc_path: str,
                         struct_emb_path: str,
                         text_emb_path: str,
                         logic_emb_path: str,
                         hierarchy_path: Optional[str],
                         text_pca_dim: Optional[int],
                         checkpoint_path: str,
                         feature_dim: int,
                         hidden_dim: int,
                         num_layers: int,
                         dropout: float,
                         aggregate: str,
                         normalization: str,
                         mlp_layers: int,
                         heads: int,
                         project_node_embeds: bool,
                         state_dict: Optional[Dict[str, torch.Tensor]],
                         use_text: Optional[bool],
                         use_logic: Optional[bool],
                         device: torch.device):
    return _instantiate_stage1_model(
        records_path=records_path,
        voc_path=voc_path,
        struct_emb_path=struct_emb_path,
        text_emb_path=text_emb_path,
        logic_emb_path=logic_emb_path,
        hierarchy_path=hierarchy_path,
        text_pca_dim=text_pca_dim,
        checkpoint_path=checkpoint_path,
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        aggregate=aggregate,
        normalization=normalization,
        mlp_layers=mlp_layers,
        heads=heads,
        project_node_embeds=project_node_embeds,
        state_dict=state_dict,
        use_text_override=use_text,
        use_logic_override=use_logic,
        device=device,
    )