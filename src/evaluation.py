import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from src.convex_optimizer import optimize_portfolio

def evaluate_and_save(returns, predictions, config):
    """
    完整版回測評估函數
    - 比較三種策略：ML + Convex、Equal Weight、Sample Mean-Variance
    - 輸出完整績效指標表格
    - 自動儲存 csv 和圖表
    """
    print("開始進行回測評估...")

    rebalance_step = config['evaluation']['rebalance_freq']
    rebalance_dates = returns.index[::rebalance_step]
    
    portfolio_returns = pd.Series(index=returns.index, dtype=float)

    # === ML + Convex Optimization 回測 ===
    for i in range(len(rebalance_dates) - 1):
        t = rebalance_dates[i]
        past_returns = returns.loc[:t]
        
        if len(past_returns) < config['ml']['lookback_windows']:
            continue
            
        Sigma = past_returns.cov()
        mu = predictions.loc[t]
        
        weights = optimize_portfolio(mu, Sigma, config)
        
        # 計算下一期實際報酬
        next_period = returns.loc[t:rebalance_dates[i+1]].iloc[1:]
        if len(next_period) > 0:
            port_ret = (weights * next_period).sum(axis=1)
            portfolio_returns.loc[port_ret.index] = port_ret

    # === 基準策略 1: Equal Weight ===
    n_assets = len(returns.columns)
    eq_weights = pd.Series(1.0 / n_assets, index=returns.columns)
    eq_returns = (eq_weights * returns).sum(axis=1)

    # === 基準策略 2: Sample Mean-Variance ===
    sample_mu = returns.mean()
    sample_Sigma = returns.cov()
    sample_weights = optimize_portfolio(sample_mu, sample_Sigma, config)
    sample_returns = (sample_weights * returns).sum(axis=1)

    # === 計算績效指標 ===
    strategies = {
        "ML + Convex": portfolio_returns.dropna(),
        "Equal Weight": eq_returns,
        "Sample MV": sample_returns
    }

    metrics_list = []
    for name, ret_series in strategies.items():
        if len(ret_series) == 0:
            continue
            
        ann_ret = ret_series.mean() * 52
        ann_vol = ret_series.std() * np.sqrt(52)
        sharpe = ann_ret / ann_vol if ann_vol != 0 else 0
        
        # 最大回撤
        cum_ret = ret_series.cumsum()
        max_dd = (cum_ret - cum_ret.cummax()).min()
        
        # Calmar Ratio
        calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0
        
        metrics_list.append({
            'Strategy': name,
            'Annual Return (%)': round(ann_ret * 100, 2),
            'Annual Volatility (%)': round(ann_vol * 100, 2),
            'Sharpe Ratio': round(sharpe, 3),
            'Max Drawdown': round(max_dd, 4),
            'Calmar Ratio': round(calmar, 3)
        })

    # 轉成 DataFrame（Strategy 作為欄位）
    metrics = pd.DataFrame(metrics_list).set_index('Strategy').T

    # 儲存檔案
    metrics.to_csv('results/performance/metrics.csv')
    portfolio_returns.dropna().to_csv('results/performance/portfolio_returns.csv')

    # === 繪製累積報酬曲線 ===
    plt.figure(figsize=(12, 7))
    portfolio_returns.dropna().cumsum().plot(label='ML + Convex', linewidth=2.5)
    eq_returns.cumsum().plot(label='Equal Weight', linestyle='--', alpha=0.8)
    sample_returns.cumsum().plot(label='Sample MV', linestyle='-.', alpha=0.8)
    
    plt.title('Cumulative Portfolio Returns (Out-of-Sample Backtest)', fontsize=14, pad=20)
    plt.xlabel('Date')
    plt.ylabel('Cumulative Log Return')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/performance/equity_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

    # === 輸出專業總結 ===
    print("\n" + "="*75)
    print("                  投資組合回測績效總結")
    print("="*75)
    print(metrics.round(3))
    
    # 安全取值（避免 KeyError）
    ml_sharpe = metrics.loc['Sharpe Ratio'].iloc[0]
    eq_sharpe = metrics.loc['Sharpe Ratio'].iloc[1]
    mv_sharpe = metrics.loc['Sharpe Ratio'].iloc[2]
    
    print("\nSharpe Ratio 提升幅度：")
    print(f"   ML+Convex vs Equal Weight : +{ml_sharpe - eq_sharpe:.3f}")
    print(f"   ML+Convex vs Sample MV    : +{ml_sharpe - mv_sharpe:.3f}")
    
    print("\n📊 結果檔案已儲存：")
    print("   • results/performance/metrics.csv          → 完整績效表格")
    print("   • results/performance/equity_curve.png     → 累積報酬曲線圖")
    print("   • results/performance/portfolio_returns.csv → 投資組合報酬序列")

    return metrics
