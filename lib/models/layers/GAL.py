
import torch.nn as nn
from einops import rearrange, repeat, einsum
import torch.nn.functional as F
from timm.models.layers import Mlp

class GraphAttentionLayer(nn.Module):
    """
    单层图注意力层
    """
    
    def __init__(self, d_model=768, num_heads=12, dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = d_model // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        
        # 投影层
        # self.q_proj = nn.Linear(d_model, d_model)
        # self.k_proj = nn.Linear(d_model, d_model)
        # self.v_proj = nn.Linear(d_model, d_model)

        # 边注意力投影
        self.edge_attention = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_heads)
        )

        self.attn_drop = nn.Dropout(dropout)
        
        # 输出投影
        self.out_proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(dropout)
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # 前馈网络
        # self.ffn = nn.Sequential(
        #     nn.Linear(d_model, d_model * 4),
        #     nn.ReLU(),
        #     nn.Linear(d_model * 4, d_model),
        #     nn.Dropout(dropout)
        # )
        self.ffn = Mlp(in_features=d_model, hidden_features=d_model * 4, act_layer=nn.GELU, drop=dropout)

        
    def forward(self, x, edge_features=None, causal_adj=None):
        """
        x: [B, N, D] - 节点特征
        edge_features: [B, N, N, 2D] - 边特征
        返回: [B, N, D] - 更新后的节点特征
        """
        B, N, C = x.shape
        residual = x

        x = self.norm1(x)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        
        # # 1. 多头注意力
        # Q = self.q_proj(x)  # [B, N, D]
        # K = self.k_proj(x)  # [B, N, D]
        # V = self.v_proj(x)  # [B, N, D]
        #
        # # 重排为多头
        # Q = rearrange(Q, 'b n (h d) -> b h n d', h=self.num_heads)
        # K = rearrange(K, 'b n (h d) -> b h n d', h=self.num_heads)
        # V = rearrange(V, 'b n (h d) -> b h n d', h=self.num_heads)
        #
        # # 计算注意力分数
        # scale = self.head_dim ** -0.5
        # attn_scores = einsum(Q, K, 'b h i d, b h j d -> b h i j') * scale
        
        # 添加边注意力（如果提供）
        if edge_features is not None:
            # 计算边注意力权重
            edge_attn = self.edge_attention(edge_features)  # [B, N, N, H]
            edge_attn = rearrange(edge_attn, 'b i j h -> b h i j')  # [B, H, N, N]

            if causal_adj is not None:
                # causal_adj: [B, N, N] -> [B, 1, N, N]
                causal_adj = causal_adj.unsqueeze(1)  # [B, 1, N, N]
                # 结合方式：乘法（可尝试加法，但乘法更符合权重调制）
                edge_attn = edge_attn + causal_adj
            
            # 将边注意力乘到注意力分数上
            attn_scores = attn_scores + edge_attn
        
        # 应用softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_drop(attn_weights)

        attended = (attn_weights @ v).transpose(1, 2).reshape(B, N, C)
        attended = self.out_proj(attended)
        attended = self.proj_drop(attended)
        
        # # 应用注意力
        # attended = einsum(attn_weights, V, 'b h i j, b h j d -> b h i d')
        # attended = rearrange(attended, 'b h n d -> b n (h d)')
        # attended = self.out_proj(attended)
        
        # 残差连接
        x = residual + attended
        
        # 2. 前馈网络
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x
        
        return x, attn_weights