import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from src.convex_optimizer import optimize_portfolio

def calculate_metrics(returns: pd.Series, name: str):
    """計算完整的績效指標"""
    if len(returns.dropna()) == 0:
        return {name: [0, 0, 0, 0, 0]}
    
    # 年化報酬與波動率（假設週頻率）
    ann_ret = returns.mean() * 52
    ann_vol = returns.std() * np.sqrt(52)
    sharpe = ann_ret / ann_vol if ann_vol != 0 else 0
    
    # 最大回撤
    cum_ret = returns.cumsum()
    cum_max = cum_ret.cummax()
    drawdown = cum_ret - cum_max
    max_dd = drawdown.min()
    
    # Calmar Ratio = 年化報酬 / 最大回撤（絕對值）
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0
    
    return {
        'Strategy': [name],
        'Annual Return (%)': [round(ann_ret * 100, 2)],
        'Annual Volatility (%)': [round(ann_vol * 100, 2)],
        'Sharpe Ratio': [round(sharpe, 3)],
        'Max Drawdown': [round(max_dd, 4)],
        'Calmar Ratio': [round(calmar, 3)]
    }

def evaluate_and_save(returns, predictions, config):
    """
    改進版：完整回測 + 多策略比較 + 專業表格輸出
    """
    print("開始進行回測評估...")

    rebalance_step = config['evaluation']['rebalance_freq']
    rebalance_dates = returns.index[::rebalance_step]
    
    portfolio_returns = pd.Series(index=returns.index, dtype=float)
    weights_history = []

    for i in range(len(rebalance_dates) - 1):
        t = rebalance_dates[i]
        past_returns = returns.loc[:t]
        
        if len(past_returns) < config['ml']['lookback_windows']:
            continue
            
        Sigma = past_returns.cov()
        mu = predictions.loc[t]
        
        weights = optimize_portfolio(mu, Sigma, config)
        weights_history.append(weights)
        
        # 計算下一期實際報酬
        next_period = returns.loc[t:rebalance_dates[i+1]].iloc[1:]
        if len(next_period) > 0:
            port_ret = (weights * next_period).sum(axis=1)
            portfolio_returns.loc[port_ret.index] = port_ret

    # === 基準策略 ===
    n_assets = len(returns.columns)
    eq_weights = pd.Series(1.0 / n_assets, index=returns.columns)
    eq_returns = (eq_weights * returns).sum(axis=1)

    # 傳統樣本均值-變異數 (Sample MV)
    sample_mu = returns.mean()
    sample_Sigma = returns.cov()
    sample_weights = optimize_portfolio(sample_mu, sample_Sigma, config)
    sample_returns = (sample_weights * returns).sum(axis=1)

    # === 計算各策略績效 ===
    metrics_dict = {}
    metrics_dict.update(calculate_metrics(portfolio_returns.dropna(), "ML + Convex Optimization"))
    metrics_dict.update(calculate_metrics(eq_returns, "Equal Weight"))
    metrics_dict.update(calculate_metrics(sample_returns, "Sample Mean-Variance"))

    # 轉成 DataFrame 並轉置
    metrics = pd.DataFrame(metrics_dict).set_index('Strategy').T

    # 儲存結果
    metrics.to_csv('results/performance/metrics.csv')
    portfolio_returns.dropna().to_csv('results/performance/portfolio_returns.csv')

    # === 繪製累積報酬曲線 ===
    plt.figure(figsize=(12, 7))
    portfolio_returns.dropna().cumsum().plot(label='ML + Convex Optimization', linewidth=2.5)
    eq_returns.cumsum().plot(label='Equal Weight', linestyle='--', alpha=0.8)
    sample_returns.cumsum().plot(label='Sample Mean-Variance', linestyle='-.', alpha=0.8)
    
    plt.title('Cumulative Portfolio Returns (Out-of-Sample Backtest)', fontsize=14, pad=20)
    plt.xlabel('Date')
    plt.ylabel('Cumulative Log Return')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/performance/equity_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

    # === 輸出專業總結 ===
    print("\n" + "="*60)
    print("                  投資組合回測績效總結")
    print("="*60)
    print(metrics.round(3))
    print("\nSharpe Ratio 提升幅度：")
    print(f"   ML+Convex vs Equal Weight : +{metrics.loc['Sharpe Ratio', 'ML + Convex Optimization'] - metrics.loc['Sharpe Ratio', 'Equal Weight']:.3f}")
    print(f"   ML+Convex vs Sample MV    : +{metrics.loc['Sharpe Ratio', 'ML + Convex Optimization'] - metrics.loc['Sharpe Ratio', 'Sample Mean-Variance']:.3f}")
    
    print("\n📊 結果檔案已儲存：")
    print("   • results/performance/metrics.csv          → 完整績效表格")
    print("   • results/performance/equity_curve.png     → 累積報酬曲線")
    print("   • results/performance/portfolio_returns.csv → 投資組合報酬序列")

    return metrics
