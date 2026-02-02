import torch
import numpy as np
from typing import Tuple


def prepare_edge_features(
    edge_index: torch.Tensor,
    node_features: torch.Tensor,
    edge_type: str = "ddi"
) -> torch.Tensor:
    src_features = node_features[edge_index[0]]
    tgt_features = node_features[edge_index[1]]
    edge_features = torch.cat([src_features, tgt_features], dim=-1)
    return edge_features


def prepare_query_edges(node_pairs: list, num_nodes: int) -> torch.Tensor:
    edges = []
    for a, b in node_pairs:
        a = min(max(a, 0), num_nodes - 1)
        b = min(max(b, 0), num_nodes - 1)
        edges.append([a, b])
    return torch.tensor(edges, dtype=torch.long)


def negative_sampling(edge_index: torch.Tensor, num_nodes: int, num_negatives: int = 1) -> torch.Tensor:
    num_positives = edge_index.shape[1]
    negative_edges = []
    for i in range(num_positives):
        src = edge_index[0, i].item()
        for _ in range(num_negatives):
            while True:
                tgt = np.random.randint(0, num_nodes)
                if tgt != src:
                    break
            negative_edges.append([src, tgt])
    return torch.tensor(negative_edges, dtype=torch.long)
