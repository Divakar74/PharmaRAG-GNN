import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionHead(nn.Module):
    def __init__(self, hidden_dim: int, head_dim: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, head_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class ProjectionHead(nn.Module):
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
    def __init__(self, hidden_dim: int, fusion_type: str = "concat", dropout: float = 0.1):
        super().__init__()
        self.fusion_type = fusion_type
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
        return self.dropout(fused)
