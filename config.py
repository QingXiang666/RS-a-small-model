import torch

class Config:
    # 路径
    data_dir = './data/ml-1m/'
    cache_dir = './cache/'

    # 数据集信息（动态更新）
    num_users = None
    num_items = None
    num_genres = None
    num_occupations = 21    # ml-1m数据中没有职业特征，仅占位不使用

    # 模型参数
    inner_dim = 128  # 内部嵌入维度（ID嵌入、GNN、序列聚合等）
    out_dim = 32  # 最终输出维度（用于相似度计算）
    max_seq_len = 10    # 用户行为序列的最大长度
    dcn_hidden = [64, 32]   # DCN深度交叉网络的隐藏层维度
    cross_layers = 3    # DCN的层数
    dropout = 0.1

    # 是否使用 GNN（GCN）
    use_gnn = True
    gnn_layers = 3
    gnn_fusion = 'concat'          # 'concat' 或 'sum'，如何融合GNN与ID嵌入

    # 特征相关
    use_user_features = False   # ml-1m无用户特征
    use_item_features = True    # 使用电影类型特征

    # 用户特征维度（保留，实际不用）
    age_buckets = 10
    gender_emb_dim = 2
    occupation_emb_dim = 8

    # 损失函数参数
    alpha1 = 1.0        # 点击损失权重
    alpha2 = 1.0        # 点赞损失权重
    margin = 0.3        # hinge loss 的间隔

    # 训练参数
    batch_size = 1024
    epochs = 100
    lr = 0.001
    weight_decay = 1e-5    # 权重衰减
    use_logQ_correction = True   # 训练时是否使用采样概率修正

    # 评估
    top_k = [10, 20]
    eval_exclude_interacted = True    # 评估时是否排除训练集中已交互的物品

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Config()