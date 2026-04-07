import cvxpy as cp
import pandas as pd
import numpy as np

def optimize_portfolio(mu: pd.Series, Sigma: pd.DataFrame, config):
    """
    使用凸優化求解單期 Mean-Variance 投資組合配置問題
    目標：最小化風險同時最大化風險調整後報酬
    """
    assets = mu.index
    n = len(assets)
    
    # 定義變數：權重 w
    w = cp.Variable(n)
    
    # 風險厭惡係數
    lambda_risk = config['optimization']['risk_aversion']
    
    # 目標函數：0.5 * w^T Σ w - λ * μ^T w   （凸二次規劃）
    objective = cp.Minimize(0.5 * cp.quad_form(w, Sigma.values) 
                           - lambda_risk * w.T @ mu.values)
    
    # 約束條件
    constraints = [cp.sum(w) == 1]   # 權重總和為 1
    
    if config['optimization'].get('long_only', True):
        constraints.append(w >= 0)   # long-only（不做空）
    
    # 建立並求解問題
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS, verbose=False)
    
    if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        weights = pd.Series(w.value, index=assets)
        print(f"  ✓ 凸優化成功 | 最大權重: {weights.max():.4f} | 最小權重: {weights.min():.4f}")
        return weights
    else:
        print(f"  ⚠️  優化失敗 (狀態: {prob.status})，改用等權重配置")
        return pd.Series(1.0 / n, index=assets)
