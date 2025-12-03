"""
特徴量重要度の詳細分析とマッピング

モデルから特徴量重要度を取得し、実際の特徴量名とマッピングして
低重要度の特徴量を特定します。
"""

import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def analyze_with_feature_names(results_dir: str):
    """特徴量名付きで重要度を分析"""
    results_path = Path(results_dir)

    # モデルファイルを読み込み
    model_files = list(results_path.glob("model_*.joblib"))
    if not model_files:
        print("モデルファイルが見つかりません")
        return

    model_file = model_files[0]
    print(f"\n分析対象: {model_file.name}")

    model = joblib.load(model_file)

    # パラメータファイルから使用した特徴量リストを取得
    # （run_ml_pipeline.pyで保存された情報から推測）

    # 特徴量重要度を取得
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_

        # CatBoostの場合
        if hasattr(model, "feature_names_"):
            feature_names = model.feature_names_
        elif hasattr(model, "feature_name_"):
            feature_names = model.feature_name_
        else:
            # 特徴量名がない場合、結果ディレクトリから推測
            # とりあえずFeature_Nとして扱う
            feature_names = [f"Feature_{i}" for i in range(len(importances))]
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
    print(f"標準偏差: {importance_df['importance'].std():.6f}")

    # パーセンタイル
    percentiles = [10, 25, 50, 75, 90]
    print(f"\nパーセンタイル:")
    for p in percentiles:
        val = np.percentile(importance_df["importance"], p)
        print(f"  {p}%ile: {val:.6f}")

    # 上位30特徴量
    print(f"\n=== 上位30特徴量 ===")
    top_30 = importance_df.head(30)
    for idx, (_, row) in enumerate(top_30.iterrows(), 1):
        print(f"{idx:2d}. {row['feature']:45s} | {row['importance']:.6f}")

    # 下位30特徴量
    print(f"\n=== 下位30特徴量（削除候補） ===")
    bottom_30 = importance_df.tail(30)
    for idx, (_, row) in enumerate(bottom_30.iterrows(), 1):
        rank = len(importance_df) - len(bottom_30) + idx
        print(f"[{rank:3d}] {row['feature']:45s} | {row['importance']:.6f}")

    # 削除推奨（重要度が平均の20%未満）
    threshold_20 = importance_df["importance"].mean() * 0.2
    low_importance_20 = importance_df[importance_df["importance"] < threshold_20]

    print(f"\n=== 削除推奨（重要度 < 平均の20%: {threshold_20:.6f}） ===")
    print(f"該当特徴量数: {len(low_importance_20)}")
    if len(low_importance_20) > 0:
        print("\n削除候補リスト:")
        for _, row in low_importance_20.iterrows():
            print(f"  - {row['feature']:45s} | {row['importance']:.6f}")

    # 削除推奨（重要度が中央値の50%未満）
    threshold_median_50 = importance_df["importance"].median() * 0.5
    low_importance_median = importance_df[
        importance_df["importance"] < threshold_median_50
    ]

    print(f"\n=== 削除推奨（重要度 < 中央値の50%: {threshold_median_50:.6f}） ===")
    print(f"該当特徴量数: {len(low_importance_median)}")

    # JSONファイルとして保存
    output_json = results_path / "feature_importance_detailed.json"
    importance_dict = {
        "statistics": {
            "total_features": len(importance_df),
            "mean": float(importance_df["importance"].mean()),
            "median": float(importance_df["importance"].median()),
            "std": float(importance_df["importance"].std()),
            "threshold_20pct_mean": float(threshold_20),
            "threshold_50pct_median": float(threshold_median_50),
        },
        "features": importance_df.to_dict(orient="records"),
        "low_importance_candidates": low_importance_20["feature"].tolist(),
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(importance_dict, f, indent=2, ensure_ascii=False)

    print(f"\n詳細結果を保存しました: {output_json}")

    return importance_df, low_importance_20


if __name__ == "__main__":
    # 最新の結果ディレクトリを自動検出
    results_base = Path(__file__).parent.parent.parent / "results" / "ml_pipeline"
    latest_dir = max(results_base.glob("run_*"), key=lambda x: x.stat().st_mtime)

    print(f"最新の実行結果: {latest_dir}")
    analyze_with_feature_names(str(latest_dir))
