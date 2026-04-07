import yfinance as yf
import pandas as pd
import numpy as np
import time

def load_data(config):
    """
    下載股票資料 - 防 rate limit 加強版（每次下載一檔 + 等待）
    """
    tickers = config['data']['tickers']
    start = config['data']['start_date']
    end = config['data']['end_date']
    
    print(f"正在下載 {len(tickers)} 檔股票資料 (結束日期: {end})...")
    print("⚠️  因 yfinance rate limit，採用單檔下載 + 等待策略...")
    
    data_list = []
    
    for i, ticker in enumerate(tickers):
        print(f"  下載 [{i+1}/{len(tickers)}] {ticker} ...")
        try:
            # 單檔下載，使用日頻率（最穩定）
            df = yf.download(ticker, start=start, end=end, 
                           interval="1d", progress=False, auto_adjust=True)
            
            if not df.empty:
                data_list.append(df['Close'].rename(ticker))
                print(f"    ✓ {ticker} 下載成功 ({len(df)} 筆資料)")
            else:
                print(f"    ⚠️ {ticker} 無資料")
            
            # 每下載一檔等待 2 秒，避免被封
            time.sleep(2)
            
        except Exception as e:
            print(f"    ❌ {ticker} 下載失敗: {e}")
            time.sleep(5)
    
    if not data_list:
        raise ValueError("所有股票下載失敗！請稍後再試或減少股票數量。")
    
    # 合併所有股票的收盤價
    data = pd.concat(data_list, axis=1)
    
    # 計算對數報酬
    returns = np.log(data / data.shift(1)).dropna()
    
    if returns.empty or len(returns) < 20:
        raise ValueError(f"報酬資料不足 (僅 {len(returns)} 期)，請調整日期範圍後重試。")
    
    print(f"\n✅ 資料載入成功：{returns.shape[0]} 期 × {returns.shape[1]} 檔資產")
    print(f"   時間範圍：{returns.index[0].date()} 到 {returns.index[-1].date()}")
    
    return returns
