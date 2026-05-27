import torch
import torch.nn as nn
from towers import UserTower, ItemTower
from config import config

class TwoTowerModel(nn.Module):
    def __init__(self, user_embedding, item_embedding, edge_index=None):
        super().__init__()
        self.user_embedding = user_embedding
        self.item_embedding = item_embedding
        if edge_index is not None:
            self.register_buffer('edge_index', edge_index)
        else:
            self.edge_index = None

        if config.use_gnn:
            from gcn import GCN
            self.gnn = GCN(
                num_users=config.num_users,
                num_items=config.num_items,
                inner_dim=config.inner_dim,
                user_emb=user_embedding,
                item_emb=item_embedding,
                n_layers=config.gnn_layers
            )
            # 融合层
            if config.gnn_fusion == 'concat':
                self.user_fusion = nn.Linear(config.inner_dim * 2, config.inner_dim)
                self.item_fusion = nn.Linear(config.inner_dim * 2, config.inner_dim)
            elif config.gnn_fusion == 'sum':
                self.user_fusion = lambda x_id, x_gnn: x_id + x_gnn
                self.item_fusion = lambda x_id, x_gnn: x_id + x_gnn
            else:
                raise ValueError(f"Unknown gnn_fusion: {config.gnn_fusion}")
        else:
            self.gnn = None

        self.user_tower = UserTower()
        self.item_tower = ItemTower()

    def get_fused_user_emb(self, user_ids):
        """
        根据用户ID获取融合后的用户嵌入
        user_ids: [B] 或标量，1-based（0为padding）
        返回: [B, D]
        """
        u_id = self.user_embedding(user_ids)  # [B, D]
        if self.gnn is None:
            return u_id

        u_gnn_all, _ = self.gnn(self.edge_index)
        # 将user_ids转为0-based索引，排除padding（user_ids=0）
        user_idx = user_ids - 1
        # 处理padding的情况：user_idx为-1的位置，需要置零
        u_gnn = torch.zeros_like(u_id)
        non_zero = user_ids != 0
        if non_zero.any():
            u_gnn[non_zero] = u_gnn_all[user_idx[non_zero]]

        if config.gnn_fusion == 'concat':
            return self.user_fusion(torch.cat([u_id, u_gnn], dim=-1))
        else:  # 'sum'
            return self.user_fusion(u_id, u_gnn)

    def get_fused_item_emb(self, item_ids):
        """
        根据物品ID获取融合后的物品嵌入
        item_ids: [B] 或标量，1-based（0为padding）
        返回: [B, D]
        """
        i_id = self.item_embedding(item_ids)  # [B, D]
        if self.gnn is None:
            return i_id

        _, i_gnn_all = self.gnn(self.edge_index)
        item_idx = item_ids - 1
        i_gnn = torch.zeros_like(i_id)
        non_zero = item_ids != 0
        if non_zero.any():
            i_gnn[non_zero] = i_gnn_all[item_idx[non_zero]]

        if config.gnn_fusion == 'concat':
            return self.item_fusion(torch.cat([i_id, i_gnn], dim=-1))
        else:
            return self.item_fusion(i_id, i_gnn)

    def forward(self, user_ids, click_seq, like_seq, pos_item_ids, neg_item_ids,
                pos_item_genres, neg_item_genres):
        # 基础 ID 嵌入（包含 padding）
        u_id = self.user_embedding(user_ids)          # [B, D]
        pos_i_id = self.item_embedding(pos_item_ids)  # [B, D]
        neg_i_id = self.item_embedding(neg_item_ids)  # [B, D]

        if self.gnn is not None:
            # GNN 嵌入（真实节点，索引 0..N-1）
            u_gnn_all, i_gnn_all = self.gnn(self.edge_index)

            # 索引转换（用户/物品 ID 从 1 开始 → 0-based）
            user_idx = user_ids - 1
            pos_item_idx = pos_item_ids - 1
            neg_item_idx = neg_item_ids - 1

            u_gnn = torch.zeros_like(u_id)
            non_zero_user = user_ids != 0
            if non_zero_user.any():
                u_gnn[non_zero_user] = u_gnn_all[user_idx[non_zero_user]]

            pos_i_gnn = torch.zeros_like(pos_i_id)
            non_zero_pos = pos_item_ids != 0
            if non_zero_pos.any():
                pos_i_gnn[non_zero_pos] = i_gnn_all[pos_item_idx[non_zero_pos]]

            neg_i_gnn = torch.zeros_like(neg_i_id)
            non_zero_neg = neg_item_ids != 0
            if non_zero_neg.any():
                neg_i_gnn[non_zero_neg] = i_gnn_all[neg_item_idx[non_zero_neg]]

            # 融合
            if config.gnn_fusion == 'concat':
                u_emb = self.user_fusion(torch.cat([u_id, u_gnn], dim=-1))
                pos_i_emb = self.item_fusion(torch.cat([pos_i_id, pos_i_gnn], dim=-1))
                neg_i_emb = self.item_fusion(torch.cat([neg_i_id, neg_i_gnn], dim=-1))
            else:  # sum
                u_emb = self.user_fusion(u_id, u_gnn)
                pos_i_emb = self.item_fusion(pos_i_id, pos_i_gnn)
                neg_i_emb = self.item_fusion(neg_i_id, neg_i_gnn)

            # 序列物品嵌入
            click_flat = click_seq.view(-1)  # [B*L]
            like_flat = like_seq.view(-1)    # [B*L]

            # ID 嵌入
            click_id_emb = self.item_embedding(click_flat)  # [B*L, D]
            like_id_emb = self.item_embedding(like_flat)    # [B*L, D]

            # GNN 嵌入（需减1索引，padding 0 对应 0 向量）
            click_gnn_emb = torch.zeros_like(click_id_emb)
            like_gnn_emb = torch.zeros_like(like_id_emb)
            non_zero_click = click_flat != 0
            non_zero_like = like_flat != 0
            if non_zero_click.any():
                click_gnn_emb[non_zero_click] = i_gnn_all[click_flat[non_zero_click] - 1]
            if non_zero_like.any():
                like_gnn_emb[non_zero_like] = i_gnn_all[like_flat[non_zero_like] - 1]

            if config.gnn_fusion == 'concat':
                click_seq_emb_flat = self.item_fusion(torch.cat([click_id_emb, click_gnn_emb], dim=-1))
                like_seq_emb_flat = self.item_fusion(torch.cat([like_id_emb, like_gnn_emb], dim=-1))
            else:
                click_seq_emb_flat = self.item_fusion(click_id_emb, click_gnn_emb)
                like_seq_emb_flat = self.item_fusion(like_id_emb, like_gnn_emb)

            click_seq_emb = click_seq_emb_flat.view(click_seq.size(0), click_seq.size(1), -1)
            like_seq_emb = like_seq_emb_flat.view(like_seq.size(0), like_seq.size(1), -1)

        else:
            # 无 GNN，直接使用 ID 嵌入
            u_emb = u_id
            pos_i_emb = pos_i_id
            neg_i_emb = neg_i_id
            click_seq_emb = self.item_embedding(click_seq)  # [B, L, D]
            like_seq_emb = self.item_embedding(like_seq)    # [B, L, D]

        # 用户塔和物品塔
        a1, a2 = self.user_tower(u_emb, click_seq_emb, like_seq_emb)
        b_pos = self.item_tower(pos_i_emb, pos_item_genres)
        b_neg = self.item_tower(neg_i_emb, neg_item_genres)

        return a1, a2, b_pos, b_neg