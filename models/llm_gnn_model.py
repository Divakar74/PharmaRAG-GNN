import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from .heads import ProjectionHead, FeatureFusion


class LLMDDIModel(nn.Module):
    """
    LLM-GNN hybrid model for Drug-Drug Interaction prediction.
    """
    
    def __init__(
        self,
        llm_embedding_dim: int = 768,
        gnn_hidden_dim: int = 256,
        gnn_output_dim: int = 256,
        num_gnn_layers: int = 3,
        gnn_type: str = "gcn",
        dropout: float = 0.3,
        fusion_type: str = "concat"
    ):
        super().__init__()
        
        self.llm_embedding_dim = llm_embedding_dim
        self.gnn_hidden_dim = gnn_hidden_dim
        self.gnn_output_dim = gnn_output_dim
        
        # GNN layers
        self.gnn_type = gnn_type
        self.conv_layers = nn.ModuleList()
        
        # First GNN layer
        if gnn_type == "gcn":
            self.conv_layers.append(GCNConv(llm_embedding_dim, gnn_hidden_dim))
        elif gnn_type == "gat":
            self.conv_layers.append(GATConv(llm_embedding_dim, gnn_hidden_dim // 4, heads=4))
        elif gnn_type == "sage":
            self.conv_layers.append(SAGEConv(llm_embedding_dim, gnn_hidden_dim))
        
        # Additional GNN layers
        for _ in range(num_gnn_layers - 1):
            if gnn_type == "gcn":
                self.conv_layers.append(GCNConv(gnn_hidden_dim, gnn_hidden_dim))
            elif gnn_type == "gat":
                self.conv_layers.append(GATConv(gnn_hidden_dim, gnn_hidden_dim // 4, heads=4))
            elif gnn_type == "sage":
                self.conv_layers.append(SAGEConv(gnn_hidden_dim, gnn_hidden_dim))
        
        # Feature fusion
        self.fusion = FeatureFusion(
            hidden_dim=gnn_hidden_dim,
            fusion_type=fusion_type,
            dropout=dropout
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(gnn_hidden_dim * 2, gnn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gnn_hidden_dim, gnn_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gnn_hidden_dim // 2, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        query_edges: torch.Tensor
    ) -> torch.Tensor:
        x = node_features
        for conv in self.conv_layers:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.3, training=self.training)
        
        src_idx = query_edges[:, 0]
        tgt_idx = query_edges[:, 1]
        
        src_emb = x[src_idx]
        tgt_emb = x[tgt_idx]
        
        fused = self.fusion(src_emb, tgt_emb)
        output = self.classifier(fused)
        return output.squeeze(-1)
