import pandas as pd

def build_features(returns, config):
    """
    為每檔股票建構機器學習所需的特徵
    包含：動能指標、波動率、偏態等
    """
    feature_list = []
    lookback = config['ml']['lookback_windows']
    
    print(f"正在建構特徵 (回顧期: {lookback} 週)...")
    
    for ticker in returns.columns:
        ret = returns[ticker]
        df = pd.DataFrame(index=returns.index)
        
        # 動能指標 (Momentum)
        df[f'{ticker}_mom_4w'] = ret.rolling(window=4).sum()
        df[f'{ticker}_mom_12w'] = ret.rolling(window=12).sum()
        
        # 風險相關特徵
        df[f'{ticker}_vol_12w'] = ret.rolling(window=12).std()
        df[f'{ticker}_vol_52w'] = ret.rolling(window=52).std()
        df[f'{ticker}_skew'] = ret.rolling(window=lookback).skew()
        
        feature_list.append(df)
    
    # 合併所有股票的特徵
    features = pd.concat(feature_list, axis=1).dropna()
    
    print(f"特徵建構完成：{features.shape[0]} 期 × {features.shape[1]} 個特徵")
    
    return features
