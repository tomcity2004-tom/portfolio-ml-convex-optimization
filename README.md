# 使用機器學習與凸優化的投資組合優化

## 專案概述
本專案使用 **XGBoost** 建構預測訊號（預測下一期報酬），再以 **CVXPY 凸優化** 進行投資組合配置。  
完全符合「金融算法與建模」課程研究型專案要求，包含完整 pipeline 與可重現性。

## 快速開始（GitHub Codespaces 或本地皆可）

### 1. 安裝環境
```bash
pip install -r requirements.txt
```

### 2. 執行環境
```bash
python main_pipeline.py
```

### 專案結構
```
portfolio-ml-convex-optimization/
├── config.yaml                  # 集中管理所有參數
├── requirements.txt             # 環境依賴套件與版本
├── README.md                    # 專案說明與執行指南
├── main_pipeline.py             # 單一入口主程式
├── src/                         # 核心功能模組
│   ├── data_loader.py           # 資料取得與載入
│   ├── preprocessing.py         # 資料清理與特徵工程
│   ├── ml_model.py              # XGBoost 預測模型
│   ├── convex_optimizer.py      # 凸優化配置求解
│   ├── evaluation.py            # 回測評估與績效指標
│   └── utils.py                 # 共用工具函數（如固定隨機種子）
├── results/                     # 執行後自動產生（不納入版本控制）
│   └── performance/             # 指標表格、圖表、權重檔案
└── .gitignore                   # 忽略暫存檔案與結果資料夾
```
