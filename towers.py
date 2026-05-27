import torch
import torch.nn as nn
import torch.nn.functional as F
from dcn import DCN
from config import config

class UserTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner_dim = config.inner_dim
        self.out_dim = config.out_dim

        self.seq_fc = nn.Linear(self.inner_dim, self.inner_dim)
        self.dcn_click = DCN(self.inner_dim * 2, config.dcn_hidden, config.cross_layers, config.dropout)
        self.dcn_like = DCN(self.inner_dim * 2, config.dcn_hidden, config.cross_layers, config.dropout)
        self.fc_click = nn.Linear(self.inner_dim * 2, self.out_dim)   # 输出 out_dim
        self.fc_like = nn.Linear(self.inner_dim * 2, self.out_dim)     # 输出 out_dim

    def forward(self, u_emb, click_seq_emb, like_seq_emb):
        """
        u_emb: [B, inner_dim]
        click_seq_emb: [B, L, inner_dim]
        like_seq_emb: [B, L, inner_dim]
        """
        # 对序列求平均（考虑 mask）
        mask_click = (click_seq_emb.abs().sum(dim=-1) != 0).float().unsqueeze(-1)  # [B, L, 1]
        mask_like = (like_seq_emb.abs().sum(dim=-1) != 0).float().unsqueeze(-1)

        click_agg = (click_seq_emb * mask_click).sum(dim=1) / mask_click.sum(dim=1).clamp(min=1)  # [B, inner_dim]
        like_agg = (like_seq_emb * mask_like).sum(dim=1) / mask_like.sum(dim=1).clamp(min=1)

        click_agg = self.seq_fc(click_agg)
        like_agg = self.seq_fc(like_agg)

        # 拼接用户嵌入和序列聚合
        click_concat = torch.cat([u_emb, click_agg], dim=-1)  # [B, 2*inner_dim]
        like_concat = torch.cat([u_emb, like_agg], dim=-1)

        # 经过 DCN 和最终投影
        click_out = self.dcn_click(click_concat)              # [B, 2*inner_dim]
        like_out = self.dcn_like(like_concat)                 # [B, 2*inner_dim]

        a1 = self.fc_click(click_out)                         # [B, out_dim]
        a2 = self.fc_like(like_out)                           # [B, out_dim]
        return a1, a2


class ItemTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner_dim = config.inner_dim
        self.out_dim = config.out_dim

        if config.use_item_features:
            self.genre_net = nn.Sequential(
                nn.Linear(config.num_genres, 32),
                nn.ReLU(),
                nn.Linear(32, self.inner_dim)
            )
        dcn_input_dim = self.inner_dim * (2 if config.use_item_features else 1)
        self.dcn = DCN(dcn_input_dim, config.dcn_hidden, config.cross_layers, config.dropout)
        self.final_proj = nn.Linear(dcn_input_dim, self.out_dim)   # 输出 out_dim

    def forward(self, i_emb, item_genres=None):
        """
        i_emb: [B, inner_dim]
        item_genres: [B, num_genres] (可选)
        返回: [B, out_dim]
        """
        concat_list = [i_emb]
        if config.use_item_features and item_genres is not None:
            genre_emb = self.genre_net(item_genres)                # [B, inner_dim]
            concat_list.append(genre_emb)
        concat = torch.cat(concat_list, dim=-1)                    # [B, dcn_input_dim]
        out = self.dcn(concat)                                     # [B, dcn_input_dim]
        b = self.final_proj(out)                                   # [B, out_dim]
        return b