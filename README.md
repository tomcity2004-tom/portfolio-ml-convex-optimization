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
