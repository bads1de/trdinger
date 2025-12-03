"""
特徴量重要度分析スクリプト

最新のMLパイプライン実行結果から特徴量重要度を抽出し、
上位/下位特徴量をレポートします。
"""

import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def analyze_feature_importance(results_dir: str):
    """特徴量重要度を分析"""
    results_path = Path(results_dir)

    # モデルファイルを探す
    model_files = list(results_path.glob("model_*.joblib"))

    if not model_files:
        print("モデルファイルが見つかりません")
        return

    # 最初のモデルを読み込み（通常は最良モデル）
    model_file = model_files[0]
    print(f"\n分析対象: {model_file.name}")

    model = joblib.load(model_file)

    # 特徴量重要度を取得
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feature_names = (
            model.feature_name_
            if hasattr(model, "feature_name_")
            else [f"Feature_{i}" for i in range(len(importances))]
        )
    else:
        print("このモデルには feature_importances_ がありません")
        return

    # DataFrameを作成
    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)

    # 統計情報
    print(f"\n=== 特徴量重要度統計 ===")
    print(f"総特徴量数: {len(importance_df)}")
    print(f"平均重要度: {importance_df['importance'].mean():.6f}")
    print(f"中央値重要度: {importance_df['importance'].median():.6f}")
    print(f"重要度ゼロの特徴量数: {(importance_df['importance'] == 0).sum()}")

    # 上位20特徴量
    print(f"\n=== 上位20特徴量（最も重要） ===")
    top_20 = importance_df.head(20)
    for idx, row in top_20.iterrows():
        print(f"{row['feature']:40s} | {row['importance']:.6f}")

    # 下位20特徴量
    print(f"\n=== 下位20特徴量（削除候補） ===")
    bottom_20 = importance_df.tail(20)
    for idx, row in bottom_20.iterrows():
        print(f"{row['feature']:40s} | {row['importance']:.6f}")

    # 新規追加した特徴量の分析
    print(f"\n=== 新規追加特徴量の重要度 ===")

    new_features_keywords = [
        "POC_Distance",
        "VAH_Distance",
        "VAL_Distance",
        "In_Value_Area",
        "HVN_Distance",
        "VP_Skewness",
        "Returns_Skewness",
        "Returns_Kurtosis",
        "Volume_Skewness",
        "HL_Ratio_Mean",
        "Return_Asymmetry",
        "OI_Price_Regime",
        "FR_Acceleration",
        "Smart_Money_Flow",
        "Market_Stress_V2",
        "OI_Volume_Interaction",
    ]

    new_features = importance_df[
        importance_df["feature"].str.contains(
            "|".join(new_features_keywords), case=False, na=False
        )
    ]

    if len(new_features) > 0:
        print(f"検出された新規特徴量: {len(new_features)}")
        for idx, row in new_features.iterrows():
            rank = importance_df.index.get_loc(idx) + 1
            print(f"[Rank {rank:3d}] {row['feature']:40s} | {row['importance']:.6f}")
    else:
        print("新規特徴量が検出されませんでした")

    # 削除推奨特徴量（重要度が非常に低い）
    print(f"\n=== 削除推奨（重要度 < 平均の10%） ===")
    threshold = importance_df["importance"].mean() * 0.1
    low_importance = importance_df[importance_df["importance"] < threshold]

    print(f"該当特徴量数: {len(low_importance)}")
    print("削除候補:")
    for idx, row in low_importance.iterrows():
        print(f"  - {row['feature']}")

    # CSVに保存
    output_file = results_path / "feature_importance.csv"
    importance_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"\n特徴量重要度を保存しました: {output_file}")

    return importance_df


if __name__ == "__main__":
    # 最新の結果ディレクトリを自動検出
    results_base = Path(__file__).parent.parent.parent / "results" / "ml_pipeline"
    latest_dir = max(results_base.glob("run_*"), key=lambda x: x.stat().st_mtime)

    print(f"最新の実行結果: {latest_dir}")
    analyze_feature_importance(str(latest_dir))
