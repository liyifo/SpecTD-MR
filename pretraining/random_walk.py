import argparse
import json
from pathlib import Path
from typing import Optional

import torch
from torch_geometric.nn import Node2Vec

from .data_builder import build_drug_hypergraph


def generate_deepwalk_embeddings(records_path: str,
                                 voc_path: str,
                                 output_path: str,
                                 embed_dim: int = 128,
                                 walk_length: int = 40,
                                 context_size: int = 5,
                                 walks_per_node: int = 10,
                                 num_negative_samples: int = 1,
                                 batch_size: int = 128,
                                 lr: float = 0.01,
                                 epochs: int = 10,
                                 device: Optional[str] = None,
                                 workers: int = 4) -> None:
    artifacts = build_drug_hypergraph(records_path=records_path,
                                      voc_path=voc_path,
                                      struct_emb_path=None,
                                      text_emb_path=None,
                                      logic_emb_path=None,
                                      hierarchy_path=None)
    data = artifacts.data
    edge_index = data.edge_index
    num_nodes = data.num_nodes

    device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))

    node2vec = Node2Vec(edge_index=edge_index,
                        embedding_dim=embed_dim,
                        walk_length=walk_length,
                        context_size=context_size,
                        walks_per_node=walks_per_node,
                        num_negative_samples=num_negative_samples,
                        p=1.0,
                        q=1.0,
                        sparse=True).to(device)

    optimizer = torch.optim.SparseAdam(list(node2vec.parameters()), lr=lr)
    loader = node2vec.loader(batch_size=batch_size, shuffle=True, num_workers=workers)

    node2vec.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = node2vec.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        print(f'Epoch {epoch}/{epochs} - loss: {avg_loss:.4f}')

    node2vec.eval()
    embeddings = node2vec.forward().detach().cpu()
    ids = list(range(num_nodes))
    payload = {
        'ids': ids,
        'embeddings': embeddings,
        'metadata': {
            'records_path': str(records_path),
            'voc_path': str(voc_path),
            'embed_dim': embed_dim,
            'walk_length': walk_length,
            'context_size': context_size,
            'walks_per_node': walks_per_node,
            'epochs': epochs,
        }
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print(f'Saved embeddings to {output_path}')


def main():
    parser = argparse.ArgumentParser(description='Generate DeepWalk embeddings for concept-visit hypergraph.')
    parser.add_argument('--records-path', default='data/MIMIC-III/records_final.pkl')
    parser.add_argument('--voc-path', default='data/MIMIC-III/voc_final.pkl')
    parser.add_argument('--output-path', default='data/emb/struct_deepwalk.pt')
    parser.add_argument('--embed-dim', type=int, default=128)
    parser.add_argument('--walk-length', type=int, default=40)
    parser.add_argument('--context-size', type=int, default=5)
    parser.add_argument('--walks-per-node', type=int, default=10)
    parser.add_argument('--num-negative-samples', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--device', default=None)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    generate_deepwalk_embeddings(records_path=args.records_path,
                                 voc_path=args.voc_path,
                                 output_path=args.output_path,
                                 embed_dim=args.embed_dim,
                                 walk_length=args.walk_length,
                                 context_size=args.context_size,
                                 walks_per_node=args.walks_per_node,
                                 num_negative_samples=args.num_negative_samples,
                                 batch_size=args.batch_size,
                                 lr=args.lr,
                                 epochs=args.epochs,
                                 device=args.device,
                                 workers=args.workers)


if __name__ == '__main__':
    main()
