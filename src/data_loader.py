import yfinance as yf
import pandas as pd
import numpy as np

def load_data(config):
    """
    使用 yfinance 下載股票資料，並計算對數報酬
    已加入錯誤處理與日頻率備援
    """
    tickers = config['data']['tickers']
    start = config['data']['start_date']
    end = config['data']['end_date']
    
    print(f"正在下載 {len(tickers)} 檔香港股票資料...")
    
    # 先嘗試下載日頻率資料（較穩定）
    data = yf.download(tickers, start=start, end=end, 
                      interval="1d", progress=False, auto_adjust=True)['Close']
    
    if data.empty or data.shape[0] < 10:
        print("⚠️ 日頻率資料下載失敗，嘗試週頻率...")
        data = yf.download(tickers, start=start, end=end, 
                          interval="1wk", progress=False, auto_adjust=True)['Close']
    
    # 如果只有一檔，轉成 DataFrame
    if len(tickers) == 1:
        data = data.to_frame(tickers[0])
    
    # 計算對數報酬
    returns = np.log(data / data.shift(1)).dropna()
    
    if returns.empty:
        raise ValueError("❌ 下載的資料過少或完全為空，請檢查網路或調整日期範圍！")
    
    print(f"資料載入完成：{returns.shape[0]} 期 × {returns.shape[1]} 檔資產")
    print(f"時間範圍：{returns.index[0].date()} 到 {returns.index[-1].date()}")
    
    return returns
