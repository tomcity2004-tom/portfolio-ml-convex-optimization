import yfinance as yf
import pandas as pd
import numpy as np

def load_data(config):
    """
    使用 yfinance 同時下載多檔股票（美股版本）
    """
    tickers = config['data']['tickers']
    start = config['data']['start_date']
    end = config['data']['end_date']
    
    print(f"正在下載 {len(tickers)} 檔美股資料...")
    
    # 同時下載所有股票的調整後收盤價
    data = yf.download(tickers, start=start, end=end, 
                      interval="1d", progress=False, auto_adjust=True)['Close']
    
    # 如果只有一檔，轉成 DataFrame
    if len(tickers) == 1:
        data = data.to_frame(tickers[0])
    
    # 計算對數報酬並去除 NaN
    returns = np.log(data / data.shift(1)).dropna()
    
    if returns.empty or len(returns) < 20:
        raise ValueError(f"❌ 資料不足！僅有 {len(returns)} 期有效報酬，請檢查日期範圍。")
    
    print(f"✅ 資料載入成功：{returns.shape[0]} 期 × {returns.shape[1]} 檔資產")
    print(f"   時間範圍：{returns.index[0].date()} 到 {returns.index[-1].date()}")
    print(f"   股票代碼：{list(returns.columns)}")
    
    return returns
