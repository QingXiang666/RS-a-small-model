import torch
import torch.nn.functional as F
import numpy as np
from config import config

def train_epoch(model, dataloader, optimizer, device, item_log_tensor, item_genre_tensor, num_items):
    model.train()
    total_loss = 0
    for batch in dataloader:
        user, pos_item, click_seq, like_seq, pos_genres = [x.to(device) for x in batch]
        B = user.size(0)

        a1, a2, b_pos, _ = model(user, click_seq, like_seq, pos_item, pos_item, pos_genres, pos_genres)

        # 归一化 -> 余弦相似度
        a1_norm = F.normalize(a1, p=2, dim=-1)
        a2_norm = F.normalize(a2, p=2, dim=-1)
        b_pos_norm = F.normalize(b_pos, p=2, dim=-1)

        sim_click = torch.mm(a1_norm, b_pos_norm.t())   # [B, B]
        sim_like = torch.mm(a2_norm, b_pos_norm.t())

        pos_scores_click = sim_click.diag()
        pos_scores_like = sim_like.diag()

        if config.use_logQ_correction:
            item_counts = torch.bincount(pos_item, minlength=num_items+1)
            p = item_counts[pos_item] / B
            log_p = torch.log(p + 1e-12)

            pos_scores_click = pos_scores_click - log_p
            pos_scores_like = pos_scores_like - log_p

            log_p_all = torch.log(item_counts / B + 1e-12)
            log_p_matrix = log_p_all[pos_item]
            sim_click_corrected = sim_click - log_p_matrix
            sim_like_corrected = sim_like - log_p_matrix
        else:
            sim_click_corrected = sim_click
            sim_like_corrected = sim_like
            pos_scores_click = sim_click.diag()
            pos_scores_like = sim_like.diag()

        mask = ~torch.eye(B, dtype=torch.bool, device=device)
        neg_scores_click = sim_click_corrected[mask].view(B, B-1)
        neg_scores_like = sim_like_corrected[mask].view(B, B-1)

        pos_click = pos_scores_click[:, None]
        pos_like = pos_scores_like[:, None]

        loss_click = torch.clamp(neg_scores_click + config.margin - pos_click, min=0).mean()
        loss_like = torch.clamp(neg_scores_like + config.margin - pos_like, min=0).mean()

        loss = config.alpha1 * loss_click + config.alpha2 * loss_like

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device, item_genre_tensor, k=10, train_user_items=None):
    """验证集评估：返回 Recall@k 和 AUC（仅点击任务）"""
    model.eval()
    recalls = []
    aucs = []
    with torch.no_grad():
        all_items = torch.arange(1, config.num_items+1, device=device)
        all_item_emb_base = model.get_fused_item_emb(all_items)
        all_item_emb = model.item_tower(all_item_emb_base, item_genre_tensor[all_items])
        all_item_emb_norm = F.normalize(all_item_emb, p=2, dim=-1)   # 归一化

        for batch in dataloader:
            user, pos_item, click_seq, like_seq, _ = [x.to(device) for x in batch]
            u_emb = model.get_fused_user_emb(user)
            click_seq_emb = model.get_fused_item_emb(click_seq.view(-1)).view(click_seq.size(0), click_seq.size(1), -1)
            like_seq_emb = model.get_fused_item_emb(like_seq.view(-1)).view(like_seq.size(0), like_seq.size(1), -1)
            a1, _ = model.user_tower(u_emb, click_seq_emb, like_seq_emb)
            a1_norm = F.normalize(a1, p=2, dim=-1)   # 归一化

            scores = torch.mm(a1_norm, all_item_emb_norm.t())   # [B, num_items]

            if train_user_items is not None:
                for i, u in enumerate(user.cpu().numpy()):
                    if u in train_user_items:
                        interacted = train_user_items[u] - 1
                        scores[i, interacted] = -float('inf')

            for i in range(len(user)):
                user_scores = scores[i]
                true_id = pos_item[i].item()
                sorted_indices = torch.argsort(user_scores, descending=True)
                rank = (sorted_indices == (true_id - 1)).nonzero(as_tuple=True)[0].item() + 1
                num_items = all_item_emb.size(0)
                auc = (num_items - rank) / (num_items - 1)
                aucs.append(auc)

                if (true_id - 1) in sorted_indices[:k]:
                    recalls.append(1)
                else:
                    recalls.append(0)

    return np.mean(recalls), np.mean(aucs)


def train(model, train_loader, val_loader, item_genre_tensor, num_items, device, train_user_items):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5)
    best_val_auc = 0
    early_stop_counter = 0

    for epoch in range(config.epochs):
        loss = train_epoch(model, train_loader, optimizer, device, None, item_genre_tensor, num_items)
        val_recall, val_auc = evaluate(model, val_loader, device, item_genre_tensor,
                                        k=config.top_k[0], train_user_items=train_user_items)
        print(f'Epoch {epoch+1}, Loss: {loss:.4f}, Val Recall@{config.top_k[0]}: {val_recall:.4f}, Val AUC: {val_auc:.4f}')

        scheduler.step(val_auc)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), 'best_model.pth')
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= 10:
                print("Early stopping")
                break