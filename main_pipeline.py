import yaml
import os
from src.data_loader import load_data
from src.preprocessing import build_features
from src.ml_model import get_ml_predictions
from src.evaluation import evaluate_and_save
from src.utils import set_seed

def main():
    # 建立結果資料夾（如果不存在就自動建立）
    os.makedirs("results/performance", exist_ok=True)
    os.makedirs("results/portfolios", exist_ok=True)

    # 載入設定檔
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 固定隨機種子，確保結果可重現
    set_seed(config["ml"]["random_seed"])

    print("=== 1. 資料取得/載入 ===")
    returns = load_data(config)

    print("=== 2. 清理與特徵工程 ===")
    features = build_features(returns, config)

    print("=== 3. ML 估計預期報酬 / 建構訊號 ===")
    predictions = get_ml_predictions(returns, features, config)

    print("=== 4. 凸優化 + 回測評估 ===")
    evaluate_and_save(returns, predictions, config)

    print("\n✅ 完整流程執行完畢！")
    print("   所有結果已儲存至 results/ 資料夾")
    print("   你可以在 GitHub 左側檔案列表點擊 results/performance/ 查看圖表與指標")

if __name__ == "__main__":
    main()
