import os
import torch
import pandas as pd
import numpy as np
from collections import defaultdict
import pickle
from config import config

def load_ratings_1m():
    """加载 ratings.dat，返回 DataFrame 并重新映射 user_id, item_id 为从 1 开始的连续值（0 留作 padding）"""
    file_path = os.path.join(config.data_dir, 'ratings.dat')
    df = pd.read_csv(file_path, sep='::', names=['user_id', 'item_id', 'rating', 'timestamp'],
                     engine='python', encoding='latin-1')

    unique_users = df['user_id'].unique()
    user_map = {old: new + 1 for new, old in enumerate(unique_users)}
    df['user_id'] = df['user_id'].map(user_map)

    unique_items = df['item_id'].unique()
    item_map = {old: new + 1 for new, old in enumerate(unique_items)}
    df['item_id'] = df['item_id'].map(item_map)

    return df, user_map, item_map

def load_movies_10m(item_map):
    """加载 movies.dat，提取类型多热向量，返回 item_id -> genre_vector 的字典"""
    file_path = os.path.join(config.data_dir, 'movies.dat')
    movies_df = pd.read_csv(file_path, sep='::', names=['item_id', 'title', 'genres'],
                            engine='python', encoding='latin-1')

    all_genres = set()
    for g in movies_df['genres'].str.split('|'):
        all_genres.update(g)
    genre_list = sorted(all_genres)
    genre_to_idx = {g: i for i, g in enumerate(genre_list)}
    num_genres = len(genre_list)

    item_features = {}
    for _, row in movies_df.iterrows():
        old_id = row['item_id']
        if old_id not in item_map:
            continue
        new_id = item_map[old_id]
        vec = [0] * num_genres
        for g in row['genres'].split('|'):
            vec[genre_to_idx[g]] = 1
        item_features[new_id] = vec

    return item_features, num_genres

def split_by_time(df, val_ratio=0.1, test_ratio=0.1):
    """按时间顺序为每个用户划分训练/验证/测试集"""
    df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
    train_list, val_list, test_list = [], [], []

    for user, group in df.groupby('user_id'):
        interactions = group.values
        n = len(interactions)
        if n < 3:
            train_list.extend(interactions)
            continue
        n_train = int(n * (1 - val_ratio - test_ratio))
        n_val = int(n * val_ratio)
        train_list.extend(interactions[:n_train])
        val_list.extend(interactions[n_train:n_train + n_val])
        test_list.extend(interactions[n_train + n_val:])

    train_df = pd.DataFrame(train_list, columns=df.columns)
    val_df = pd.DataFrame(val_list, columns=df.columns)
    test_df = pd.DataFrame(test_list, columns=df.columns)
    return train_df, val_df, test_df

def build_user_sequences(train_df, max_len, like_threshold=4):
    """
    根据训练集构建每个用户的点击序列和点赞序列（按时间排序，取最近 max_len 个物品）
    返回字典：user_id -> {'click': list, 'like': list}
    """
    train_df = train_df.sort_values(['user_id', 'timestamp'])
    user_seq = {}
    for user, group in train_df.groupby('user_id'):
        click_seq = group['item_id'].tolist()[-max_len:]
        like_seq = group[group['rating'] >= like_threshold]['item_id'].tolist()[-max_len:]
        # 左侧填充0至max_len
        click_seq = [0] * (max_len - len(click_seq)) + click_seq
        like_seq = [0] * (max_len - len(like_seq)) + like_seq
        user_seq[user] = {'click': click_seq, 'like': like_seq}
    return user_seq

def compute_item_popularity_power(train_df, power=0.75):
    """
    计算物品流行度：count^power，归一化为概率，并返回 log(p)
    """
    counts = train_df['item_id'].value_counts()
    counts_pow = counts ** power
    probs = counts_pow / counts_pow.sum()
    log_probs = np.log(probs + 1e-12)   # 避免 log(0)
    p_dict = probs.to_dict()
    log_p_dict = log_probs.to_dict()
    return p_dict, log_p_dict

def build_interaction_graph(train_df, num_users, num_items):
    """
    构建用户-物品交互图的边索引（0-based），用于 GNN
    """
    users = train_df['user_id'].values - 1   # 转为0-based
    items = train_df['item_id'].values - 1 + num_users   # 物品节点偏移
    edge_index = np.vstack([np.concatenate([users, items]),
                            np.concatenate([items, users])])   # 无向边
    return torch.tensor(edge_index, dtype=torch.long)

def save_cache(obj, name):
    os.makedirs(config.cache_dir, exist_ok=True)
    with open(os.path.join(config.cache_dir, name), 'wb') as f:
        pickle.dump(obj, f)

def load_cache(name):
    path = os.path.join(config.cache_dir, name)
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

def prepare_data():
    """加载所有数据，返回所需字典和DataFrame，并更新 config"""
    data = load_cache('all_data_1m.pkl')
    if data is not None:
        config.num_users = data['num_users']
        config.num_items = data['num_items']
        config.num_genres = data['num_genres']
        return data

    ratings_df, user_map, item_map = load_ratings_1m()
    item_features, num_genres = load_movies_10m(item_map)

    train_df, val_df, test_df = split_by_time(ratings_df)

    user_seq = build_user_sequences(train_df, config.max_seq_len)

    item_prob, item_log = compute_item_popularity_power(train_df, power=0.75)

    # 正样本对
    train_pairs = train_df[['user_id', 'item_id']].values.tolist()
    val_pairs = val_df[['user_id', 'item_id']].values.tolist()
    test_pairs = test_df[['user_id', 'item_id']].values.tolist()

    # 训练集用户交互字典（用于评估排除已交互物品）
    train_user_items = {}
    for user, group in train_df.groupby('user_id'):
        train_user_items[user] = group['item_id'].values

    num_users = ratings_df['user_id'].nunique()
    num_items = ratings_df['item_id'].nunique()

    config.num_users = num_users
    config.num_items = num_items
    config.num_genres = num_genres

    # 构建图边（如果需要 GNN）
    edge_index = None
    if config.use_gnn:
        edge_index = build_interaction_graph(train_df, num_users, num_items)

    data = {
        'item_features': item_features,
        'user_seq': user_seq,                # dict: {user: {'click': [...], 'like': [...]}}
        'item_log': item_log,                 # dict: item_id -> log(p)
        'train_pairs': train_pairs,
        'val_pairs': val_pairs,
        'test_pairs': test_pairs,
        'train_user_items': train_user_items,
        'num_users': num_users,
        'num_items': num_items,
        'num_genres': num_genres,
        'edge_index': edge_index,              # 用于 GNN
    }
    save_cache(data, 'all_data_1m.pkl')
    return data