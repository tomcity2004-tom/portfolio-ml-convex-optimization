import pandas as pd
import xgboost as xgb

def get_ml_predictions(returns, features, config):
    """
    使用 XGBoost 為每檔股票建構下一期報酬預測訊號
    目前使用簡化版全樣本訓練（研究型專案建議後續改成 walk-forward）
    """
    print("正在訓練 XGBoost 模型並產生預測訊號...")
    
    predictions = pd.DataFrame(index=features.index, columns=returns.columns)
    
    for ticker in returns.columns:
        # 取出該股票對應的特徵
        asset_features = features[[col for col in features.columns if col.startswith(ticker)]]
        
        # 目標變數：下一期實際報酬（shift(-1)）
        y = returns[ticker].shift(-1).loc[asset_features.index].dropna()
        X = asset_features.loc[y.index]
        
        if len(X) < 30:  # 樣本太少則跳過
            predictions[ticker] = 0.0
            continue
        
        # 訓練 XGBoost 迴歸模型
        model = xgb.XGBRegressor(
            n_estimators=config['ml']['n_estimators'],
            random_state=config['ml']['random_seed'],
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8
        )
        
        model.fit(X, y)
        
        # 產生預測值
        pred = model.predict(X)
        predictions.loc[X.index, ticker] = pred
        
        print(f"  ✓ {ticker} 模型訓練完成 (樣本數: {len(X)})")
    
    print(f"XGBoost 預測訊號產生完成！形狀: {predictions.shape}")
    return predictions
