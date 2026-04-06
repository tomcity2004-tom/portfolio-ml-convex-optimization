# 使用機器學習與凸優化的投資組合優化

## 專案概述
本專案使用 **XGBoost** 建構預測訊號（預測下一期報酬），再以 **CVXPY 凸優化** 進行投資組合配置。  
完全符合「金融算法與建模」課程研究型專案要求，包含完整 pipeline 與可重現性。

## 快速開始（GitHub Codespaces 或本地皆可）

### 1. 安裝環境
```bash
pip install -r requirements.txt
```

### 只需執行以下單一指令，即可跑完整個專案流程：
```Bash
python main_pipeline.py
```
### 完整執行流程
資料取得/載入 (src/data_loader.py)
使用 yfinance 下載香港股市股票歷史價格資料（週頻率）
清理 + 特徵工程 (src/preprocessing.py)
計算對數報酬、動能指標、波動率、偏態等特徵
ML 估計/訓練 (src/ml_model.py)
使用 XGBoost 建構下一期報酬預測訊號
凸優化配置 (src/convex_optimizer.py)
將 ML 預測結果轉為預期報酬 μ，結合共變異數矩陣 Σ 進行 mean-variance 凸優化
評估與結果輸出 (src/evaluation.py)
進行滾動窗回測，計算 Sharpe Ratio、年化報酬、最大回撤等指標，並與等權重基準比較，輸出圖表與報告

### 主要參數設定（config.yaml）

資料：香港市場股票（0700.HK、9988.HK、0005.HK、1299.HK、0388.HK）
時間範圍：2018 年至 2026 年
ML 模型：XGBoost（可調整 n_estimators、lookback window）
優化參數：風險厭惡係數 (risk_aversion)、是否 long-only
再平衡頻率：每 4 週一次

### 預期輸出結果（執行後會出現在 results/ 資料夾）

results/performance/metrics.csv：各模型的績效指標（Sharpe Ratio、年化報酬、波動率等）
results/performance/equity_curve.png：投資組合累積報酬曲線圖（與等權重基準比較）
results/portfolios/：每期最佳投資組合權重
Console 輸出：ML + Convex 優化後的 Sharpe Ratio 提升幅度

### 可重現性保證
所有隨機種子 (random_seed) 已固定
所有參數集中於 config.yaml
使用固定版本的 requirements.txt
單一指令 python main_pipeline.py 即可完整重現所有結果

### 專案結構
```
portfolio-ml-convex-optimization/
├── config.yaml
├── requirements.txt
├── README.md
├── main_pipeline.py
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── ml_model.py
│   ├── convex_optimizer.py
│   ├── evaluation.py
│   └── utils.py
├── results/          # 執行後自動產生
└── data/             # 原始與處理後資料
```
