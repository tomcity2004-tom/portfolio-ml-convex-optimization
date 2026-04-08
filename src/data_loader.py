import yfinance as yf
import pandas as pd
import numpy as np
import time

def load_data(config):
    """單檔逐一下載 + 等待，降低 rate limit 風險"""
    tickers = config['data']['tickers']
    start = config['data']['start_date']
    end = config['data']['end_date']
    
    print(f"正在下載 {len(tickers)} 檔股票資料（使用單檔 + 等待策略）...")
    print(f"股票：{tickers}")
    
    data_list = []
    
    for i, ticker in enumerate(tickers):
        print(f"  [{i+1}/{len(tickers)}] 下載 {ticker} ...")
        try:
            df = yf.download(ticker, start=start, end=end, interval="1d", 
                           progress=False, auto_adjust=True, timeout=15)
            
            if not df.empty and len(df) > 100:
                data_list.append(df['Close'].rename(ticker))
                print(f"    ✓ 成功 ({len(df)} 筆資料)")
            else:
                print(f"    ⚠️ 資料不足")
                
            time.sleep(3)   # 每檔等待 3 秒，避免被封
            
        except Exception as e:
            print(f"    ❌ 下載失敗: {e}")
            time.sleep(5)
    
    if not data_list:
        raise ValueError("❌ 所有股票都下載失敗！請等待 30 分鐘後再試，或減少股票數量。")
    
    # 合併資料
    data = pd.concat(data_list, axis=1)
    returns = np.log(data / data.shift(1)).dropna()
    
    if len(returns) < 50:
        raise ValueError(f"❌ 有效報酬資料太少 (僅 {len(returns)} 期)")
    
    print(f"\n✅ 資料載入成功！{returns.shape[0]} 期 × {returns.shape[1]} 檔資產")
    print(f"   時間範圍：{returns.index[0].date()} 到 {returns.index[-1].date()}")
    
    return returns
