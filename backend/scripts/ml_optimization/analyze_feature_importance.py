"""
特徴量重要度分析スクリプト

最新のMLパイプライン実行結果から特徴量重要度を抽出し、
統計情報、上位/下位特徴量、および削除推奨候補をレポートします。
"""

import sys
import json
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

    # feature_names.json から特徴量名を読み込むことを優先
    feature_names_file = results_path / "feature_names.json"
    loaded_feature_names = None
    if feature_names_file.exists():
        with open(feature_names_file, "r", encoding="utf-8") as f:
            loaded_feature_names = json.load(f)
        print(f"特徴量名を {feature_names_file.name} から読み込みました。")

    # 最初のモデルを読み込み（通常は最良モデル）
    model_file = model_files[0]
    print(f"\n分析対象: {model_file.name}")

    model = joblib.load(model_file)

    # 特徴量重要度を取得
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        if loaded_feature_names: # 読み込んだ特徴量名を優先
            feature_names = loaded_feature_names
        elif hasattr(model, "feature_names_"): # CatBoostなど
            feature_names = model.feature_names_
        elif hasattr(model, "feature_name_") and model.feature_name_: # LightGBMなど
            feature_names = model.feature_name_
        else: # どちらもなければGeneric名
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
    print(f"重要度ゼロの特徴量数: {(importance_df['importance'] == 0).sum()}")

    # パーセンタイル
    percentiles = [10, 25, 50, 75, 90]
    print(f"\nパーセンタイル:")
    for p in percentiles:
        val = np.percentile(importance_df["importance"], p)
        print(f"  {p}%ile: {val:.6f}")

    # 上位30特徴量
    print(f"\n=== 上位30特徴量（最も重要） ===")
    top_30 = importance_df.head(30)
    for idx, row in top_30.iterrows():
        print(f"{row['feature']:45s} | {row['importance']:.6f}")

    # 下位30特徴量
    print(f"\n=== 下位30特徴量（削除候補） ===")
    bottom_30 = importance_df.tail(30)
    for idx, row in bottom_30.iterrows():
        print(f"{row['feature']:45s} | {row['importance']:.6f}")

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
        # Newly added features keywords
        "FracDiff",
        "Parkinson_Vol",
        "Garman_Klass_Vol",
        "VWAP_Z_Score",
        "RVOL",
        "Absorption_Score",
        "Liquidation_Cascade_Score",
        "Squeeze_Probability",
        "Trend_Quality",
        "OI_Weighted_FR",
        "Liquidity_Efficiency",
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
            print(f"[Rank {rank:3d}] {row['feature']:45s} | {row['importance']:.6f}")
    else:
        print("新規特徴量が検出されませんでした")

    # 削除推奨特徴量（複数の基準で判定）
    print(f"\n=== 削除推奨分析 ===")
    
    # 基準1: 平均の10%未満
    threshold_10 = importance_df["importance"].mean() * 0.1
    low_importance_10 = importance_df[importance_df["importance"] < threshold_10]
    print(f"基準1 (平均の10%未満 < {threshold_10:.6f}): {len(low_importance_10)}件")

    # 基準2: 平均の20%未満
    threshold_20 = importance_df["importance"].mean() * 0.2
    low_importance_20 = importance_df[importance_df["importance"] < threshold_20]
    print(f"基準2 (平均の20%未満 < {threshold_20:.6f}): {len(low_importance_20)}件")

    # 基準3: 中央値の50%未満
    threshold_median_50 = importance_df["importance"].median() * 0.5
    low_importance_median = importance_df[importance_df["importance"] < threshold_median_50]
    print(f"基準3 (中央値の50%未満 < {threshold_median_50:.6f}): {len(low_importance_median)}件")

    print("\n削除候補詳細 (基準2: 平均の20%未満):")
    for idx, row in low_importance_20.iterrows():
        print(f"  - {row['feature']:45s} | {row['importance']:.6f}")

    # CSVに保存
    output_file = results_path / "feature_importance.csv"
    importance_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"\n特徴量重要度(CSV)を保存しました: {output_file}")

    # JSONに保存（詳細情報付き）
    output_json = results_path / "feature_importance_detailed.json"
    importance_dict = {
        "statistics": {
            "total_features": len(importance_df),
            "mean": float(importance_df["importance"].mean()),
            "median": float(importance_df["importance"].median()),
            "std": float(importance_df["importance"].std()),
            "threshold_10pct_mean": float(threshold_10),
            "threshold_20pct_mean": float(threshold_20),
            "threshold_50pct_median": float(threshold_median_50),
        },
        "features": importance_df.to_dict(orient="records"),
        "low_importance_candidates_mean_20pct": low_importance_20["feature"].tolist(),
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(importance_dict, f, indent=2, ensure_ascii=False)
    print(f"特徴量重要度詳細(JSON)を保存しました: {output_json}")

    return importance_df


if __name__ == "__main__":
    # 最新の結果ディレクトリを自動検出
    results_base = Path(__file__).parent.parent.parent / "results" / "ml_pipeline"
    
    # ディレクトリが存在するか確認
    if not results_base.exists():
        print(f"ディレクトリが見つかりません: {results_base}")
        # テスト用のダミー実行を避けるため、ここで終了
        sys.exit(0)
        
    runs = list(results_base.glob("run_*"))
    if not runs:
        print(f"実行結果が見つかりません: {results_base}")
        sys.exit(0)

    latest_dir = max(runs, key=lambda x: x.stat().st_mtime)

    print(f"最新の実行結果: {latest_dir}")
    analyze_feature_importance(str(latest_dir))



