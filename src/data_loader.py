import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime

def load_data(config):
    """
    使用 EODHD API 下載歷史股價（5 檔同時下載版本）
    """
    tickers = config['data']['tickers']
    start = config['data']['start_date']
    end = config['data']['end_date']
    api_token = config['data']['api_token']

    print(f"正在使用 EODHD API 下載 {len(tickers)} 檔美股資料...")
    print(f"股票：{tickers}")

    data_list = []
    
    for i, ticker in enumerate(tickers):
        print(f"  [{i+1}/{len(tickers)}] 下載 {ticker} ...")
        
        url = f"https://eodhd.com/api/eod/{ticker}"
        params = {
            "api_token": api_token,
            "from": start,
            "to": end,
            "period": "d",      # 日頻率
            "fmt": "json"
        }
        
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            json_data = resp.json()
            
            if not json_data:
                print(f"    ⚠️ {ticker} 無資料")
                time.sleep(2)
                continue
                
            df = pd.DataFrame(json_data)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df = df[['close']].rename(columns={'close': ticker})
            
            data_list.append(df[ticker])
            print(f"    ✓ 成功 ({len(df)} 筆資料)")
            
            time.sleep(1.5)   # 避免太快被限制（免費方案建議）
            
        except Exception as e:
            print(f"    ❌ {ticker} 下載失敗: {e}")
            time.sleep(3)
    
    if not data_list:
        raise ValueError("❌ 所有股票下載失敗！請檢查 API Key 是否正確、額度是否足夠。")
    
    # 合併所有股票
    data = pd.concat(data_list, axis=1)
    returns = np.log(data / data.shift(1)).dropna()
    
    if len(returns) < 50:
        raise ValueError(f"❌ 有效報酬資料太少 (僅 {len(returns)} 期)")
    
    print(f"\n✅ EODHD 資料載入成功！{returns.shape[0]} 期 × {returns.shape[1]} 檔資產")
    print(f"   時間範圍：{returns.index[0].date()} 到 {returns.index[-1].date()}")
    
    return returns
