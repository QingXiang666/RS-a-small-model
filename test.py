import torch
import numpy as np
from config import config
from data_utils import prepare_data
from model import TwoTowerModel
from evaluate import evaluate_full
import torch.nn as nn
from utils import set_seed
import csv

def main():
    set_seed(42)
    print("配置:", vars(config))

    # 1. 准备数据（与训练时相同）
    print("加载数据...")
    data = prepare_data()
    item_features = data['item_features']
    user_seq = data['user_seq']
    test_pairs = data['test_pairs']
    train_user_items = data['train_user_items']
    num_users = config.num_users
    num_items = config.num_items
    num_genres = config.num_genres
    edge_index = data.get('edge_index')

    print(f"数据集统计: 用户数={num_users}, 物品数={num_items}, 类型数={num_genres}")

    # 2. 创建嵌入层（必须与训练时相同结构）
    user_embedding = nn.Embedding(num_users + 1, config.inner_dim, padding_idx=0)
    item_embedding = nn.Embedding(num_items + 1, config.inner_dim, padding_idx=0)
    nn.init.normal_(user_embedding.weight, std=0.1)
    nn.init.normal_(item_embedding.weight, std=0.1)

    user_embedding = user_embedding.to(config.device)
    item_embedding = item_embedding.to(config.device)

    # 3. 构建模型
    model = TwoTowerModel(user_embedding, item_embedding, edge_index).to(config.device)

    # 4. 准备辅助张量
    item_genre_tensor = torch.zeros(num_items + 1, num_genres, device=config.device)
    for item_id, genres in item_features.items():
        item_genre_tensor[item_id] = torch.tensor(genres, device=config.device)

    # 5. 加载最佳模型权重
    model.load_state_dict(torch.load('best_model.pth'))
    print("成功加载最佳模型 best_model.pth")

    # 6. 评估测试集
    results, recommendations = evaluate_full(
        model, test_pairs, user_seq, item_genre_tensor, config.device,
        k_list=config.top_k, train_user_items=train_user_items
    )
    print("\n测试集结果:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    # 打印前5个用户的推荐示例
    print("\n推荐示例 (前5个用户):")
    for i, (user, rec) in enumerate(recommendations.items()):
        if i >= 5:
            break
        print(f"用户 {user}: 点击推荐 Top20 = {rec['click_top20']}")
        print(f"       点赞推荐 Top20 = {rec['like_top20']}")

    # ====== 保存所有用户的推荐结果到 CSV 文件 ======
    output_file = 'recommendations.csv'
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['user_id', 'click_top20', 'like_top20'])
        for user_id, rec in recommendations.items():
            click_str = ','.join(map(str, rec['click_top20']))
            like_str = ','.join(map(str, rec['like_top20']))
            writer.writerow([user_id, click_str, like_str])
    print(f"\n所有用户的推荐结果已保存至 {output_file}")


if __name__ == '__main__':
    main()