import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv


class BioKGGNN(nn.Module):
    def __init__(self, input_dim: int = 768, hidden_dim: int = 256, output_dim: int = 256, num_layers: int = 3, gnn_type: str = "gcn", dropout: float = 0.3):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        if gnn_type == "gcn":
            self.conv_layers.append(GCNConv(input_dim, hidden_dim))
        else:
            self.conv_layers.append(SAGEConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            if gnn_type == "gcn":
                self.conv_layers.append(GCNConv(hidden_dim, hidden_dim))
            else:
                self.conv_layers.append(SAGEConv(hidden_dim, hidden_dim))
        self.dropout = dropout
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv in self.conv_layers[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv_layers[-1](x, edge_index)
        return x
