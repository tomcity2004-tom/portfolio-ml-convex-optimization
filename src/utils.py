import numpy as np
import random

def set_seed(seed: int):
    """
    固定所有隨機種子，確保整個專案結果完全可重現
    """
    np.random.seed(seed)
    random.seed(seed)
    # 如果之後使用 torch 等其他套件，也可以在這裡固定
    print(f"隨機種子已設定為: {seed} (確保可重現性)")
