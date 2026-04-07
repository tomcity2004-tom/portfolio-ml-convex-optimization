import yfinance as yf
import pandas as pd
import numpy as np
import time

def load_data(config):
    """加強版資料載入：單檔下載 + 長等待 + 美股備援"""
    tickers = config['data']['tickers']
    start = config['data']['start_date']
    end = config['data']['end_date']
    
    print(f"正在下載 {len(tickers)} 檔股票資料...")
    print("使用單檔下載策略避免 rate limit...")
    
    data_list = []
    
    for i, ticker in enumerate(tickers):
        print(f"  [{i+1}/{len(tickers)}] 下載 {ticker} ...")
        try:
            df = yf.download(ticker, start=start, end=end, interval="1d", 
                           progress=False, auto_adjust=True, timeout=10)
            
            if not df.empty and len(df) > 50:
                data_list.append(df['Close'].rename(ticker))
                print(f"    ✓ 成功 ({len(df)} 筆資料)")
            else:
                print(f"    ⚠️ 資料不足或失敗")
                
            time.sleep(3)  # 每檔等待3秒，降低被封風險
            
        except Exception as e:
            print(f"    ❌ 下載失敗: {e}")
            time.sleep(5)
    
    if not data_list:
        print("\n❌ 所有香港股票下載失敗！切換到美股測試模式...")
        # 備援：改用美股（穩定很多）
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
        print(f"改用美股測試：{tickers}")
        for ticker in tickers:
            df = yf.download(ticker, start=start, end=end, interval="1d", progress=False, auto_adjust=True)
            if not df.empty:
                data_list.append(df['Close'].rename(ticker))
            time.sleep(2)
    
    if not data_list:
        raise ValueError("無法下載任何資料，請稍後再試或聯絡我。")
    
    data = pd.concat(data_list, axis=1).dropna(how='all')
    returns = np.log(data / data.shift(1)).dropna()
    
    if len(returns) < 20:
        raise ValueError(f"有效報酬資料太少 (僅 {len(returns)} 期)")
    
    print(f"\n✅ 資料載入成功！{returns.shape[0]} 期 × {returns.shape[1]} 檔資產")
    print(f"時間範圍：{returns.index[0].date()} 到 {returns.index[-1].date()}")
    
    return returns
