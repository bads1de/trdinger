import joblib
import json
import pandas as pd
import sys
from pathlib import Path

# 結果ディレクトリのパス
RESULTS_DIR = Path(
    r"C:\Users\buti3\trading\backend\results\ml_pipeline\run_20251205_100857"
)


def main():
    # ファイルパス
    model_path = RESULTS_DIR / "model_xgb.joblib"
    feature_names_path = RESULTS_DIR / "feature_names.json"

    if not model_path.exists():
        print(f"Model file not found: {model_path}")
        return
    if not feature_names_path.exists():
        print(f"Feature names file not found: {feature_names_path}")
        return

    # ロード
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)

    print(f"Loading feature names from {feature_names_path}...")
    with open(feature_names_path, "r", encoding="utf-8") as f:
        feature_names = json.load(f)

    # 重要度取得
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        print("Model does not have feature_importances_ attribute.")
        return

    # DataFrame作成
    # 特徴量名と重要度の数が一致するか確認
    if len(feature_names) != len(importances):
        print(
            f"Warning: Number of feature names ({len(feature_names)}) does not match number of importances ({len(importances)})."
        )
        # XGBoostの場合、学習時に使われなかった特徴量が除外されることがあるが、
        # joblibで保存されたsklearnラッパーなら通常は一致するはず。
        # 一致しない場合は、model.get_booster().feature_names などで確認が必要だが、
        # ここでは単純に長さを合わせる（またはエラーにする）
        min_len = min(len(feature_names), len(importances))
        feature_names = feature_names[:min_len]
        importances = importances[:min_len]

    df_imp = pd.DataFrame({"feature": feature_names, "importance": importances})

    # ソート
    df_imp = df_imp.sort_values("importance", ascending=False)

    # 表示
    print("\n=== Feature Importance (Top 30) ===")
    print(df_imp.head(30).to_string(index=False))

    # 寄与度0の特徴量を確認
    zero_imp_features = df_imp[df_imp["importance"] == 0]
    num_zero = len(zero_imp_features)

    print(f"\n=== Zero Importance Features (Count: {num_zero}) ===")
    if num_zero > 0:
        print(zero_imp_features["feature"].tolist())
    else:
        print("None")

    # 全保存
    output_path = RESULTS_DIR / "feature_importance.csv"
    df_imp.to_csv(output_path, index=False)
    print(f"\nFull feature importance saved to {output_path}")


if __name__ == "__main__":
    main()
