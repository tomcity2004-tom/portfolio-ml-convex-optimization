import yfinance as yf
import pandas as pd
import numpy as np

def load_data(config):
    """
    下載股票資料並計算對數報酬 - 加強版錯誤處理
    """
    tickers = config['data']['tickers']
    start = config['data']['start_date']
    end = config['data']['end_date']
    
    print(f"正在下載 {len(tickers)} 檔股票資料 (結束日期: {end})...")
    
    # 優先使用日頻率（最穩定）
    data = yf.download(tickers, start=start, end=end, 
                      interval="1d", progress=False, auto_adjust=True)['Close']
    
    if data.empty or len(data) < 20:
        print("⚠️ 日頻率資料不足，嘗試週頻率...")
        data = yf.download(tickers, start=start, end=end, 
                          interval="1wk", progress=False, auto_adjust=True)['Close']
    
    if len(tickers) == 1:
        data = data.to_frame(tickers[0])
    
    # 計算對數報酬
    returns = np.log(data / data.shift(1)).dropna()
    
    # 重要防護：如果還是空的，明確報錯並給建議
    if returns.empty or len(returns) < 10:
        print("❌ 錯誤：下載後的報酬資料為空或過少！")
        print(f"   下載到的原始資料形狀: {data.shape}")
        print(f"   建議：把 config.yaml 中的 end_date 改成 '2025-12-31' 或更早")
        raise ValueError("資料下載失敗，請調整 config.yaml 中的 end_date 後重試。")
    
    print(f"✅ 資料載入成功：{returns.shape[0]} 期 × {returns.shape[1]} 檔資產")
    print(f"   時間範圍：{returns.index[0].date()} 到 {returns.index[-1].date()}")
    
    return returns
