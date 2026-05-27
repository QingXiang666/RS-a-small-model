import os
import random
import numpy as np
import torch

# 设置随机种子
# 设置完全确定性
def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)