import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionHead(nn.Module):
    """
    Single attention head for multi-head attention mechanism.
    Used for node feature attention in the GNN model.
    """
    def __init__(self, hidden_dim: int, head_dim: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, head_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention for combining features from multiple sources.
    """
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        self.num_heads = num_heads
        
        self.heads = nn.ModuleList([
            AttentionHead(hidden_dim, self.head_dim) for _ in range(num_heads)
        ])
        self.output_linear = nn.Linear(num_heads * self.head_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        head_outputs = [head(x) for head in self.heads]
        concat = torch.cat(head_outputs, dim=-1)
        output = self.output_linear(concat)
        output = self.dropout(output)
        return output


class ProjectionHead(nn.Module):
    """
    Projection head for downstream tasks.
    """
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, input_dim // 2)
        self.linear2 = nn.Linear(input_dim // 2, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(F.relu(self.linear1(x)))
        x = self.linear2(x)
        return x


class FeatureFusion(nn.Module):
    """
    Module for fusing features from different sources.
    """
    def __init__(self, hidden_dim: int, fusion_type: str = "concat", dropout: float = 0.1):
        super().__init__()
        self.fusion_type = fusion_type
        self.hidden_dim = hidden_dim
        
        if fusion_type == "concat":
            self.fusion_linear = nn.Linear(hidden_dim * 2, hidden_dim)
        elif fusion_type == "attention":
            self.attention = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        if self.fusion_type == "concat":
            fused = torch.cat([x1, x2], dim=-1)
            fused = self.fusion_linear(fused)
            fused = F.relu(fused)
        elif self.fusion_type == "add":
            fused = x1 + x2
        elif self.fusion_type == "attention":
            scores = torch.sigmoid(self.attention(torch.cat([x1, x2], dim=-1)))
            fused = scores * x1 + (1 - scores) * x2
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")
        
        return self.dropout(fused)
