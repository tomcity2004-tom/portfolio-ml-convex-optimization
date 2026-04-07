import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from src.convex_optimizer import optimize_portfolio

def evaluate_and_save(returns, predictions, config):
    """
    進行滾動回測 + 凸優化配置，並計算績效指標與繪圖
    """
    print("開始進行回測評估...")

    rebalance_step = config['evaluation']['rebalance_freq']
    rebalance_dates = returns.index[::rebalance_step]
    
    portfolio_returns = pd.Series(index=returns.index, dtype=float)
    weights_history = []

    for i in range(len(rebalance_dates) - 1):
        t = rebalance_dates[i]
        past_returns = returns.loc[:t]
        
        # 樣本不足則跳過
        if len(past_returns) < config['ml']['lookback_windows']:
            continue
            
        # 計算共變異數矩陣
        Sigma = past_returns.cov()
        
        # 使用 ML 預測作為預期報酬
        mu = predictions.loc[t]
        
        # 凸優化求解權重
        weights = optimize_portfolio(mu, Sigma, config)
        weights_history.append(weights)
        
        # 計算下一期投資組合實際報酬
        next_period = returns.loc[t:rebalance_dates[i+1]].iloc[1:]
        if len(next_period) > 0:
            port_ret = (weights * next_period).sum(axis=1)
            portfolio_returns.loc[port_ret.index] = port_ret

    # 基準策略：等權重
    n_assets = len(returns.columns)
    eq_weights = pd.Series(1.0 / n_assets, index=returns.columns)
    eq_returns = (eq_weights * returns).sum(axis=1)

    # 計算績效指標
    def calc_metrics(r: pd.Series):
        if len(r.dropna()) == 0:
            return {'Annual Return': 0, 'Annual Vol': 0, 'Sharpe': 0}
        ann_ret = r.mean() * 52
        ann_vol = r.std() * np.sqrt(52)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        max_dd = (r.cumsum().cummax() - r.cumsum()).min()
        return {
            'Annual Return (%)': round(ann_ret * 100, 2),
            'Annual Vol (%)': round(ann_vol * 100, 2),
            'Sharpe Ratio': round(sharpe, 3),
            'Max Drawdown': round(max_dd, 4)
        }

    metrics = pd.DataFrame({
        'ML_Convex_Opt': calc_metrics(portfolio_returns.dropna()),
        'Equal_Weight': calc_metrics(eq_returns)
    })

    # 儲存結果
    metrics.to_csv('results/performance/metrics.csv')
    portfolio_returns.dropna().to_csv('results/performance/portfolio_returns.csv')

    # 繪製累積報酬曲線
    plt.figure(figsize=(12, 6))
    portfolio_returns.dropna().cumsum().plot(label='ML + Convex Optimization', linewidth=2)
    eq_returns.cumsum().plot(label='Equal Weight Benchmark', linestyle='--')
    plt.title('Cumulative Portfolio Returns (Out-of-Sample Backtest)')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Log Return')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('results/performance/equity_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 輸出總結
    print("\n🎉 回測評估完成！")
    print(f"ML + Convex Optimization Sharpe Ratio: {metrics.loc['Sharpe Ratio', 'ML_Convex_Opt']}")
    print(f"Equal Weight Sharpe Ratio: {metrics.loc['Sharpe Ratio', 'Equal_Weight']}")
    print(f"Sharpe 提升幅度: {(metrics.loc['Sharpe Ratio', 'ML_Convex_Opt'] - metrics.loc['Sharpe Ratio', 'Equal_Weight']):+.3f}")
    print("\n結果檔案已儲存：")
    print("   - results/performance/metrics.csv")
    print("   - results/performance/equity_curve.png")
    print("   - results/performance/portfolio_returns.csv")

    return metrics
