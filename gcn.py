import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self, num_users, num_items, inner_dim, user_emb, item_emb, n_layers=3):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.inner_dim = inner_dim
        self.n_layers = n_layers

        # 外部传入的嵌入层（包含 padding_idx=0）
        self.user_embedding = user_emb
        self.item_embedding = item_emb

        # 使用 PyG 的 GCNConv 层（每层保持嵌入维度不变）
        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(GCNConv(inner_dim, inner_dim, bias=False))

        self.dropout = nn.Dropout(0.1)

    def forward(self, edge_index):
        """
        edge_index: [2, E] 用户-物品边（0-based，用户节点 0..num_users-1，物品节点 num_users..num_users+num_items-1）
        返回: 用户最终嵌入 [num_users, D], 物品最终嵌入 [num_items, D]
        """
        # 排除 padding 索引 0，取真实节点嵌入
        user_emb = self.user_embedding.weight[1:]  # [num_users, D]
        item_emb = self.item_embedding.weight[1:]  # [num_items, D]
        x = torch.cat([user_emb, item_emb], dim=0)  # [num_users+num_items, D]

        # 多层 GCN
        for conv in self.convs:
            x = conv(x, edge_index)          # 消息传递 + 线性变换
            x = F.relu(x)
            x = self.dropout(x)

        user_final, item_final = torch.split(x, [self.num_users, self.num_items], dim=0)
        return user_final, item_final