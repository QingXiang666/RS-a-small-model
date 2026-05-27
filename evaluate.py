import torch
import torch.nn.functional as F
import numpy as np
from config import config

def recall_at_k(pred_items, true_item, k):
    pred = set(pred_items[:k])
    true = set([true_item])
    return len(pred & true) / len(true) if true else 0

def evaluate_full(model, test_pairs, user_seq_dict, item_genre_tensor, device,
                  k_list=[10,20], train_user_items=None):
    """
    返回: results, recommendations（与之前一致）
    """
    model.eval()
    recalls_click = {k: [] for k in k_list}
    aucs_click = []
    recalls_like = {k: [] for k in k_list}
    aucs_like = []
    recommendations = {}

    num_users = config.num_users
    num_items = config.num_items
    max_seq_len = config.max_seq_len
    inner_dim = config.inner_dim

    all_items = torch.arange(1, num_items+1, device=device)
    all_users = torch.arange(1, num_users+1, device=device)

    with torch.no_grad():
        # 1. 所有物品的融合 ID 嵌入（用于序列）
        all_item_fused = model.get_fused_item_emb(all_items)  # [num_items, inner_dim]

        # 2. 所有物品的最终嵌入（用于相似度计算）
        all_item_final = model.item_tower(all_item_fused, item_genre_tensor[all_items])
        all_item_final_norm = F.normalize(all_item_final, p=2, dim=-1)  # [num_items, out_dim]

        # 3. 所有用户的融合嵌入
        all_user_fused = model.get_fused_user_emb(all_users)  # [num_users, inner_dim]

    # 开始逐用户评估
    for user, true_item in test_pairs:
        # 获取该用户的融合嵌入
        u_fused = all_user_fused[user-1].unsqueeze(0)  # [1, inner_dim]

        seq = user_seq_dict.get(user, {'click': [0]*max_seq_len, 'like': [0]*max_seq_len})
        click_seq_ids = torch.tensor([seq['click']], device=device)  # [1, L]
        like_seq_ids = torch.tensor([seq['like']], device=device)

        # 获取序列物品的嵌入（从预计算的 all_item_fused 中索引）
        click_seq_emb = torch.zeros(1, max_seq_len, inner_dim, device=device)
        like_seq_emb = torch.zeros(1, max_seq_len, inner_dim, device=device)

        non_zero_click = click_seq_ids != 0
        non_zero_like = like_seq_ids != 0
        if non_zero_click.any():
            # 注意：物品 ID 从 1 开始，索引时需要减 1
            idx_click = click_seq_ids[non_zero_click] - 1
            click_seq_emb[non_zero_click] = all_item_fused[idx_click]
        if non_zero_like.any():
            idx_like = like_seq_ids[non_zero_like] - 1
            like_seq_emb[non_zero_like] = all_item_fused[idx_like]

        # 用户塔
        a1, a2 = model.user_tower(u_fused, click_seq_emb, like_seq_emb)  # [1, out_dim]
        a1_norm = F.normalize(a1, p=2, dim=-1)
        a2_norm = F.normalize(a2, p=2, dim=-1)

        scores_click = torch.mm(a1_norm, all_item_final_norm.t()).squeeze(0)  # [num_items]
        scores_like = torch.mm(a2_norm, all_item_final_norm.t()).squeeze(0)

        # 排除训练集中已交互的物品
        if train_user_items is not None and user in train_user_items:
            interacted = train_user_items[user] - 1
            scores_click[interacted] = -float('inf')
            scores_like[interacted] = -float('inf')

        # 排序与指标计算（与原来相同）
        sorted_click = torch.argsort(scores_click, descending=True)
        sorted_like = torch.argsort(scores_like, descending=True)

        top_items_click = (sorted_click + 1).cpu().numpy()
        top_items_like = (sorted_like + 1).cpu().numpy()

        recommendations[user] = {
            'click_top20': top_items_click[:20].tolist(),
            'like_top20': top_items_like[:20].tolist()
        }

        num_items_total = all_item_final.size(0)
        rank_click = (sorted_click == (true_item - 1)).nonzero(as_tuple=True)[0].item() + 1
        rank_like = (sorted_like == (true_item - 1)).nonzero(as_tuple=True)[0].item() + 1
        auc_click = (num_items_total - rank_click) / (num_items_total - 1)
        auc_like = (num_items_total - rank_like) / (num_items_total - 1)
        aucs_click.append(auc_click)
        aucs_like.append(auc_like)

        for k in k_list:
            recalls_click[k].append(recall_at_k(top_items_click, true_item, k))
            recalls_like[k].append(recall_at_k(top_items_like, true_item, k))

    # 汇总结果（不变）
    recall_results_click = {f'recall@{k}_click': np.mean(recalls_click[k]) for k in k_list}
    auc_result_click = {'auc_click': np.mean(aucs_click)}
    recall_results_like = {f'recall@{k}_like': np.mean(recalls_like[k]) for k in k_list}
    auc_result_like = {'auc_like': np.mean(aucs_like)}

    results = {**recall_results_click, **auc_result_click,
               **recall_results_like, **auc_result_like}
    return results, recommendations