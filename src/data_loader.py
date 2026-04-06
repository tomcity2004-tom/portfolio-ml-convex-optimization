import yfinance as yf
import pandas as pd
import numpy as np

def load_data(config):
    """
    使用 yfinance 下載股票調整後收盤價，並計算對數報酬
    """
    tickers = config['data']['tickers']
    start = config['data']['start_date']
    end = config['data']['end_date']
    freq = config['data'].get('freq', '1wk')
    
    print(f"正在下載 {len(tickers)} 檔香港股票資料 (頻率: {freq})...")
    
    # 下載調整後收盤價
    data = yf.download(tickers, start=start, end=end, interval=freq, 
                      progress=False, auto_adjust=True)['Close']
    
    # 如果只有一檔股票，轉成 DataFrame
    if len(tickers) == 1:
        data = data.to_frame(tickers[0])
    
    # 計算對數報酬並去除缺失值
    returns = np.log(data / data.shift(1)).dropna()
    
    print(f"資料載入完成：{returns.shape[0]} 期 × {returns.shape[1]} 檔資產")
    print(f"時間範圍：{returns.index[0].date()} 到 {returns.index[-1].date()}")
    
    return returns
