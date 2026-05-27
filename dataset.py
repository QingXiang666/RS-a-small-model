# dataset.py
import torch
from torch.utils.data import Dataset
# 不再导入 config

class PairwiseDataset(Dataset):
    def __init__(self, pairs, user_seq, item_features, num_genres):
        """
        pairs: list of (user_id, item_id) 正样本
        user_seq: dict user_id -> {'click': list, 'like': list}
        item_features: dict item_id -> genre_vector
        num_genres: int, 类型总数（用于填充默认向量）
        """
        self.pairs = pairs
        self.user_seq = user_seq
        self.item_features = item_features
        self.num_genres = num_genres
        self.max_seq_len = 10  # 可以从 config 获取，但为避免导入，也可以作为参数传入

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        user, pos_item = self.pairs[idx]
        seq_val = self.user_seq.get(user)

        # 兼容处理：如果 seq_val 是列表（旧缓存），则将其作为点击序列，点赞序列置空
        if seq_val is None:
            click_seq = [0] * self.max_seq_len
            like_seq = [0] * self.max_seq_len
        elif isinstance(seq_val, list):
            click_seq = seq_val
            like_seq = [0] * self.max_seq_len
        else:
            click_seq = seq_val['click']
            like_seq = seq_val['like']

        # 使用 self.num_genres 而不是 config.num_genres
        pos_genres = self.item_features.get(pos_item, [0] * self.num_genres)

        return (
            torch.tensor(user, dtype=torch.long),
            torch.tensor(pos_item, dtype=torch.long),
            torch.tensor(click_seq, dtype=torch.long),
            torch.tensor(like_seq, dtype=torch.long),
            torch.tensor(pos_genres, dtype=torch.float)
        )