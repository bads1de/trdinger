"""
3モデル（XGBoost/LightGBM/TabNet）でのFR特徴量効果比較

このスクリプトは、ファンディングレート（FR）特徴量の効果を3つの異なる
機械学習モデル（XGBoost、LightGBM、TabNet）で評価し、比較します。

使用方法:
    cd backend
    python scripts/three_models_fr_comparison.py --symbol BTC/USDT:USDT

出力:
    - backend/scripts/results/three_models_fr_comparison_report.md
    - backend/scripts/results/three_models_fr_comparison_results.json
"""

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.ml.feature_engineering.funding_rate_features import (
    FundingRateFeatureCalculator,
)
from database.connection import get_session
from database.repositories.funding_rate_repository import FundingRateRepository
from database.repositories.ohlcv_repository import OHLCVRepository

warnings.filterwarnings("ignore")


def load_all_data(symbol: str = "BTC/USDT:USDT") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    データベースから全データを読み込み

    Args:
        symbol: 取引ペア

    Returns:
        (ohlcv_df, fr_df): OHLCVとファンディングレートのDataFrame
    """
    session = get_session()

    ohlcv_repo = OHLCVRepository(session)
    fr_repo = FundingRateRepository(session)

    print("=" * 70)
    print("データベースから全データを読み込み中...")
    print("=" * 70)

    # 全OHLCVデータ
    ohlcv_data = ohlcv_repo.get_all_by_symbol(symbol=symbol, timeframe="1h")

    # 全Funding Rateデータ
    fr_data = fr_repo.get_all_by_symbol(symbol=symbol)

    if not ohlcv_data:
        raise ValueError(f"データがありません: {symbol}")

    print(f"OHLCV: {len(ohlcv_data)}行を取得")
    print(f"FR: {len(fr_data)}行を取得")

    # DataFrameに変換
    ohlcv_df = pd.DataFrame(
        [
            {
                "timestamp": d.timestamp,
                "open": d.open,
                "high": d.high,
                "low": d.low,
                "close": d.close,
                "volume": d.volume,
            }
            for d in ohlcv_data
        ]
    )

    fr_df = (
        pd.DataFrame(
            [
                {
                    "timestamp": d.funding_timestamp,
                    "funding_rate": d.funding_rate,
                }
                for d in fr_data
            ]
        )
        if fr_data
        else pd.DataFrame()
    )

    # データ期間を表示
    if not ohlcv_df.empty:
        start_date = ohlcv_df["timestamp"].min()
        end_date = ohlcv_df["timestamp"].max()
        days = (end_date - start_date).days
        print(f"データ期間: {start_date} - {end_date} ({days}日間)")

    session.close()
    return ohlcv_df, fr_df


def create_baseline_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    ベースライン特徴量（FR特徴量なし）

    Args:
        df: 入力DataFrame

    Returns:
        特徴量を追加したDataFrame
    """
    df = df.copy()

    # 価格変化率
    for period in [1, 3, 6, 12, 24]:
        df[f"returns_{period}h"] = df["close"].pct_change(period)

    # 移動平均
    for period in [7, 14, 30, 50]:
        df[f"ma_{period}"] = df["close"].rolling(period).mean()
        df[f"ma_ratio_{period}"] = df["close"] / df[f"ma_{period}"]

    # ボラティリティ
    df["volatility_24h"] = df["close"].rolling(24).std()
    df["volatility_168h"] = df["close"].rolling(168).std()

    # 出来高
    df["volume_ma_24h"] = df["volume"].rolling(24).mean()
    df["volume_ratio"] = df["volume"] / (df["volume_ma_24h"] + 1e-10)

    # RSI
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["rsi_14"] = 100 - (100 / (1 + gain / (loss + 1e-10)))

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    データをクリーニング（inf、nan、極端な値を除去）

    Args:
        df: 入力DataFrame

    Returns:
        クリーニング後のDataFrame
    """
    df = df.copy()
    
    # inf値を置換
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # 各列の極端な外れ値をクリップ（99.9パーセンタイル）
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in ["timestamp", "label"]:
            q_low = df[col].quantile(0.001)
            q_high = df[col].quantile(0.999)
            df[col] = df[col].clip(q_low, q_high)
    
    return df


def create_labels(df: pd.DataFrame, threshold: float = 0.005) -> pd.DataFrame:
    """
    3クラス分類ラベルを生成（上昇/下降/レンジ）

    Args:
        df: 入力DataFrame
        threshold: 上昇/下降判定のしきい値（デフォルト: 0.5%）

    Returns:
        ラベルを追加したDataFrame
    """
    df = df.copy()

    # 1時間後の価格変化率
    df["future_returns"] = df["close"].pct_change(1).shift(-1)

    # 3クラスラベル
    # 0: 下降（< -threshold）
    # 1: レンジ（-threshold <= x <= threshold）
    # 2: 上昇（> threshold）
    conditions = [
        df["future_returns"] < -threshold,
        (df["future_returns"] >= -threshold) & (df["future_returns"] <= threshold),
        df["future_returns"] > threshold,
    ]
    df["label"] = np.select(conditions, [0, 1, 2], default=1)

    return df


def evaluate_lightgbm(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Dict[str, float]:
    """
    LightGBMモデルの評価

    Args:
        X_train: 訓練データ特徴量
        X_test: テストデータ特徴量
        y_train: 訓練データラベル
        y_test: テストデータラベル

    Returns:
        評価指標の辞書
    """
    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=6,
        random_state=42,
        verbose=-1,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # 回帰指標用の予測値（期待リターン）
    expected_returns = (
        y_pred_proba[:, 0] * -0.01 + y_pred_proba[:, 1] * 0.0 + y_pred_proba[:, 2] * 0.01
    )

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "mcc": matthews_corrcoef(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test.map({0: -0.01, 1: 0.0, 2: 0.01}), expected_returns)),
        "mae": mean_absolute_error(y_test.map({0: -0.01, 1: 0.0, 2: 0.01}), expected_returns),
        "r2": r2_score(y_test.map({0: -0.01, 1: 0.0, 2: 0.01}), expected_returns),
    }


def evaluate_xgboost(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Dict[str, float]:
    """
    XGBoostモデルの評価

    Args:
        X_train: 訓練データ特徴量
        X_test: テストデータ特徴量
        y_train: 訓練データラベル
        y_test: テストデータラベル

    Returns:
        評価指標の辞書
    """
    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        verbosity=0,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # 回帰指標用の予測値
    expected_returns = (
        y_pred_proba[:, 0] * -0.01 + y_pred_proba[:, 1] * 0.0 + y_pred_proba[:, 2] * 0.01
    )

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "mcc": matthews_corrcoef(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test.map({0: -0.01, 1: 0.0, 2: 0.01}), expected_returns)),
        "mae": mean_absolute_error(y_test.map({0: -0.01, 1: 0.0, 2: 0.01}), expected_returns),
        "r2": r2_score(y_test.map({0: -0.01, 1: 0.0, 2: 0.01}), expected_returns),
    }


def evaluate_tabnet(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Dict[str, float]:
    """
    TabNetモデルの評価

    Args:
        X_train: 訓練データ特徴量
        X_test: テストデータ特徴量
        y_train: 訓練データラベル
        y_test: テストデータラベル

    Returns:
        評価指標の辞書
    """
    model = TabNetClassifier(
        n_d=8,
        n_a=8,
        n_steps=3,
        gamma=1.3,
        n_independent=2,
        n_shared=2,
        seed=42,
        verbose=0,
    )

    # TabNetは入力をnumpy配列として受け取る
    model.fit(
        X_train.values,
        y_train.values,
        max_epochs=50,
        patience=10,
        batch_size=1024,
        virtual_batch_size=128,
    )

    y_pred = model.predict(X_test.values)
    y_pred_proba = model.predict_proba(X_test.values)

    # 回帰指標用の予測値
    expected_returns = (
        y_pred_proba[:, 0] * -0.01 + y_pred_proba[:, 1] * 0.0 + y_pred_proba[:, 2] * 0.01
    )

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "mcc": matthews_corrcoef(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test.map({0: -0.01, 1: 0.0, 2: 0.01}), expected_returns)),
        "mae": mean_absolute_error(y_test.map({0: -0.01, 1: 0.0, 2: 0.01}), expected_returns),
        "r2": r2_score(y_test.map({0: -0.01, 1: 0.0, 2: 0.01}), expected_returns),
    }


def compare_all_models(
    df_baseline: pd.DataFrame, df_fr: pd.DataFrame, n_splits: int = 5
) -> Dict:
    """
    3モデルすべてでFR特徴量の効果を比較

    Args:
        df_baseline: ベースライン特徴量のDataFrame
        df_fr: FR特徴量ありのDataFrame
        n_splits: 交差検証の分割数

    Returns:
        評価結果の辞書
    """
    # ラベル生成
    df_baseline = create_labels(df_baseline)
    df_fr = create_labels(df_fr)
    
    # データクリーニング
    df_baseline = clean_data(df_baseline)
    df_fr = clean_data(df_fr)

    # 欠損値除去
    df_baseline = df_baseline.dropna().reset_index(drop=True)
    df_fr = df_fr.dropna().reset_index(drop=True)

    # 両方のDataFrameのサイズを揃える
    min_len = min(len(df_baseline), len(df_fr))
    df_baseline = df_baseline.iloc[:min_len]
    df_fr = df_fr.iloc[:min_len]

    print(f"\n有効データ: {len(df_baseline)}行")

    # 特徴量列を特定
    baseline_features = [
        col
        for col in df_baseline.columns
        if col
        not in [
            "timestamp",
            "label",
            "future_returns",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    ]

    # FR特徴量のみを抽出
    fr_only_features = [
        col
        for col in df_fr.columns
        if col.startswith("fr_")
        or col.startswith("funding_")
        or col.startswith("regime_")
    ]

    # FR DataFrameで利用可能なベースライン特徴量
    available_baseline_in_fr = [
        col for col in baseline_features if col in df_fr.columns
    ]

    all_features = available_baseline_in_fr + fr_only_features

    print(f"ベースライン特徴量: {len(baseline_features)}個")
    print(f"FR特徴量のみ: {len(fr_only_features)}個")
    print(f"合計（FR側）: {len(all_features)}個")

    # ラベル分布を確認
    print(f"\nラベル分布:")
    print(df_baseline["label"].value_counts().sort_index())

    # 時系列交差検証
    print(f"\n{n_splits}-fold TimeSeriesSplit評価中...")
    tscv = TimeSeriesSplit(n_splits=n_splits)

    results = {
        "lightgbm": {"baseline": [], "fr_enhanced": []},
        "xgboost": {"baseline": [], "fr_enhanced": []},
        "tabnet": {"baseline": [], "fr_enhanced": []},
    }

    for fold, (train_idx, test_idx) in enumerate(tscv.split(df_baseline), 1):
        print(f"\n  Fold {fold}/{n_splits}")

        # ベースライン評価
        X_train_base = df_baseline.iloc[train_idx][baseline_features]
        X_test_base = df_baseline.iloc[test_idx][baseline_features]
        y_train = df_baseline.iloc[train_idx]["label"]
        y_test = df_baseline.iloc[test_idx]["label"]

        # FR特徴量あり評価
        X_train_fr = df_fr.iloc[train_idx][all_features]
        X_test_fr = df_fr.iloc[test_idx][all_features]
        y_train_fr = df_fr.iloc[train_idx]["label"]
        y_test_fr = df_fr.iloc[test_idx]["label"]

        # LightGBM
        print("    LightGBM...", end=" ")
        results["lightgbm"]["baseline"].append(
            evaluate_lightgbm(X_train_base, X_test_base, y_train, y_test)
        )
        results["lightgbm"]["fr_enhanced"].append(
            evaluate_lightgbm(X_train_fr, X_test_fr, y_train_fr, y_test_fr)
        )
        print("OK")

        # XGBoost
        print("    XGBoost...", end=" ")
        results["xgboost"]["baseline"].append(
            evaluate_xgboost(X_train_base, X_test_base, y_train, y_test)
        )
        results["xgboost"]["fr_enhanced"].append(
            evaluate_xgboost(X_train_fr, X_test_fr, y_train_fr, y_test_fr)
        )
        print("OK")

        # TabNet
        print("    TabNet...", end=" ")
        results["tabnet"]["baseline"].append(
            evaluate_tabnet(X_train_base, X_test_base, y_train, y_test)
        )
        results["tabnet"]["fr_enhanced"].append(
            evaluate_tabnet(X_train_fr, X_test_fr, y_train_fr, y_test_fr)
        )
        print("OK")

    return results


def calculate_summary_stats(results: Dict) -> Dict:
    """
    各モデルの結果を集計

    Args:
        results: 評価結果の辞書

    Returns:
        集計結果の辞書
    """
    summary = {}

    for model_name, model_results in results.items():
        summary[model_name] = {}

        for condition in ["baseline", "fr_enhanced"]:
            metrics = {}
            for metric_name in results[model_name][condition][0].keys():
                values = [
                    fold[metric_name] for fold in results[model_name][condition]
                ]
                metrics[metric_name] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                }
            summary[model_name][condition] = metrics

        # 改善率を計算
        summary[model_name]["improvements"] = {}
        for metric_name in summary[model_name]["baseline"].keys():
            baseline_val = summary[model_name]["baseline"][metric_name]["mean"]
            fr_val = summary[model_name]["fr_enhanced"][metric_name]["mean"]

            if metric_name in ["rmse", "mae"]:
                # 低い方が良い指標
                improvement = ((baseline_val - fr_val) / baseline_val) * 100
            else:
                # 高い方が良い指標
                if baseline_val != 0:
                    improvement = ((fr_val - baseline_val) / abs(baseline_val)) * 100
                else:
                    improvement = 0.0

            summary[model_name]["improvements"][metric_name] = improvement

    return summary


def generate_report(summary: Dict, output_path: Path) -> None:
    """
    Markdownレポートを生成

    Args:
        summary: 集計結果
        output_path: 出力ファイルパス
    """
    report_lines = [
        "# 3モデル（XGBoost/LightGBM/TabNet）FR特徴量効果比較レポート",
        "",
        f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 概要",
        "",
        "このレポートは、ファンディングレート（FR）特徴量の効果を",
        "3つの機械学習モデル（LightGBM、XGBoost、TabNet）で評価した結果をまとめたものです。",
        "",
        "### 評価設定",
        "",
        "- **タスク**: 3クラス分類（上昇/レンジ/下降）",
        "- **交差検証**: 5-fold TimeSeriesSplit",
        "- **特徴量**:",
        "  - ベースライン: 価格特徴量、テクニカル指標、ボリューム特徴量",
        "  - FR特徴量あり: ベースライン + 15個のTier 1 FR特徴量",
        "",
        "## 評価指標",
        "",
        "### 分類指標",
        "",
        "- **Accuracy**: 正解率",
        "- **F1 Score (Macro)**: マクロ平均F1スコア",
        "- **MCC**: Matthews相関係数",
        "- **Balanced Accuracy**: バランス正解率",
        "",
        "### 回帰指標（参考）",
        "",
        "- **RMSE**: 二乗平均平方根誤差",
        "- **MAE**: 平均絶対誤差",
        "- **R²**: 決定係数",
        "",
    ]

    # 各モデルの結果
    for model_name in ["lightgbm", "xgboost", "tabnet"]:
        model_display = {
            "lightgbm": "LightGBM",
            "xgboost": "XGBoost",
            "tabnet": "TabNet",
        }[model_name]

        report_lines.extend(
            [
                f"## {model_display}",
                "",
                "### ベースライン（FR特徴量なし）",
                "",
                "| 指標 | 平均 | 標準偏差 | 最小値 | 最大値 |",
                "|------|------|----------|--------|--------|",
            ]
        )

        baseline = summary[model_name]["baseline"]
        for metric_name, values in baseline.items():
            report_lines.append(
                f"| {metric_name} | {values['mean']:.6f} | {values['std']:.6f} | "
                f"{values['min']:.6f} | {values['max']:.6f} |"
            )

        report_lines.extend(
            [
                "",
                "### FR特徴量あり",
                "",
                "| 指標 | 平均 | 標準偏差 | 最小値 | 最大値 |",
                "|------|------|----------|--------|--------|",
            ]
        )

        fr_enhanced = summary[model_name]["fr_enhanced"]
        for metric_name, values in fr_enhanced.items():
            report_lines.append(
                f"| {metric_name} | {values['mean']:.6f} | {values['std']:.6f} | "
                f"{values['min']:.6f} | {values['max']:.6f} |"
            )

        report_lines.extend(
            [
                "",
                "### 改善率",
                "",
                "| 指標 | 改善率 (%) |",
                "|------|-----------|",
            ]
        )

        improvements = summary[model_name]["improvements"]
        for metric_name, improvement in improvements.items():
            sign = "+" if improvement > 0 else ""
            report_lines.append(f"| {metric_name} | {sign}{improvement:.2f}% |")

        report_lines.append("")

    # モデル間比較
    report_lines.extend(
        [
            "## モデル間比較",
            "",
            "### FR特徴量による改善率（主要指標）",
            "",
            "| モデル | Accuracy | F1 (Macro) | MCC | Balanced Acc | RMSE |",
            "|--------|----------|------------|-----|--------------|------|",
        ]
    )

    for model_name in ["lightgbm", "xgboost", "tabnet"]:
        model_display = {
            "lightgbm": "LightGBM",
            "xgboost": "XGBoost",
            "tabnet": "TabNet",
        }[model_name]

        imp = summary[model_name]["improvements"]
        report_lines.append(
            f"| {model_display} | {imp['accuracy']:+.2f}% | {imp['f1_macro']:+.2f}% | "
            f"{imp['mcc']:+.2f}% | {imp['balanced_accuracy']:+.2f}% | {imp['rmse']:+.2f}% |"
        )

    # 結論
    report_lines.extend(
        [
            "",
            "## 結論",
            "",
        ]
    )

    # 最も改善率が高いモデルを特定
    best_model = max(
        summary.keys(),
        key=lambda m: summary[m]["improvements"]["accuracy"],
    )
    best_improvement = summary[best_model]["improvements"]["accuracy"]

    model_display_names = {
        "lightgbm": "LightGBM",
        "xgboost": "XGBoost",
        "tabnet": "TabNet",
    }

    report_lines.extend(
        [
            f"**最も改善効果が高かったモデル**: {model_display_names[best_model]} "
            f"(Accuracy改善率: {best_improvement:+.2f}%)",
            "",
            "### モデル別の特徴",
            "",
        ]
    )

    for model_name in ["lightgbm", "xgboost", "tabnet"]:
        model_display = model_display_names[model_name]
        imp = summary[model_name]["improvements"]

        report_lines.extend(
            [
                f"**{model_display}**:",
                f"- Accuracy改善: {imp['accuracy']:+.2f}%",
                f"- F1 Score改善: {imp['f1_macro']:+.2f}%",
                f"- MCC改善: {imp['mcc']:+.2f}%",
                "",
            ]
        )

    # 全体的な評価
    avg_improvement = np.mean(
        [summary[m]["improvements"]["accuracy"] for m in summary.keys()]
    )

    report_lines.extend(
        [
            "### 総合評価",
            "",
            f"3モデルの平均Accuracy改善率: {avg_improvement:+.2f}%",
            "",
        ]
    )

    if avg_improvement >= 5:
        report_lines.append(
            "[SUCCESS] FR特徴量は3モデルすべてで**顕著な改善効果**が確認されました（5%以上）"
        )
    elif avg_improvement >= 2:
        report_lines.append(
            "[SUCCESS] FR特徴量は3モデルすべてで**明確な改善効果**が確認されました（2%以上）"
        )
    elif avg_improvement >= 0:
        report_lines.append(
            "[WARNING] FR特徴量により改善が見られますが、効果は限定的です"
        )
    else:
        report_lines.append("❌ FR特徴量による明確な改善は確認できませんでした")

    # ファイルに書き込み
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\n[SUCCESS] レポートを生成しました: {output_path}")


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description="3モデル（XGBoost/LightGBM/TabNet）でのFR特徴量効果比較"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTC/USDT:USDT",
        help="取引ペア（デフォルト: BTC/USDT:USDT）",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="交差検証の分割数（デフォルト: 5）",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("3モデルFR特徴量効果比較")
    print("=" * 70)

    # データ読み込み
    try:
        ohlcv_df, fr_df = load_all_data(args.symbol)
    except Exception as e:
        print(f"\n❌ エラー: {e}")
        print("\nデータ収集が必要な場合:")
        print("  python -m app.services.data_collection.bybit.ohlcv_service")
        print("  python -m app.services.data_collection.bybit.funding_rate_service")
        return 1

    # ベースライン特徴量作成
    print("\n" + "=" * 70)
    print("ベースライン特徴量作成中...")
    print("=" * 70)
    baseline_df = create_baseline_features(ohlcv_df)

    # FR特徴量作成
    print("\n" + "=" * 70)
    print("FR特徴量計算中...")
    print("=" * 70)
    calculator = FundingRateFeatureCalculator()
    fr_enhanced_df = calculator.calculate_features(ohlcv_df, fr_df)

    # 3モデル評価
    print("\n" + "=" * 70)
    print("3モデル評価実行中...")
    print("=" * 70)
    results = compare_all_models(baseline_df, fr_enhanced_df, n_splits=args.n_splits)

    # 結果集計
    print("\n" + "=" * 70)
    print("結果集計中...")
    print("=" * 70)
    summary = calculate_summary_stats(results)

    # 結果表示
    print("\n" + "=" * 70)
    print("評価結果")
    print("=" * 70)

    for model_name in ["lightgbm", "xgboost", "tabnet"]:
        model_display = {
            "lightgbm": "LightGBM",
            "xgboost": "XGBoost",
            "tabnet": "TabNet",
        }[model_name]

        print(f"\n## {model_display}")
        print("-" * 70)

        baseline = summary[model_name]["baseline"]
        fr_enhanced = summary[model_name]["fr_enhanced"]
        improvements = summary[model_name]["improvements"]

        print("\n[ベースライン]")
        print(f"  Accuracy:         {baseline['accuracy']['mean']:.6f}")
        print(f"  F1 (Macro):       {baseline['f1_macro']['mean']:.6f}")
        print(f"  MCC:              {baseline['mcc']['mean']:.6f}")
        print(f"  Balanced Acc:     {baseline['balanced_accuracy']['mean']:.6f}")

        print("\n[FR特徴量あり]")
        print(f"  Accuracy:         {fr_enhanced['accuracy']['mean']:.6f}")
        print(f"  F1 (Macro):       {fr_enhanced['f1_macro']['mean']:.6f}")
        print(f"  MCC:              {fr_enhanced['mcc']['mean']:.6f}")
        print(f"  Balanced Acc:     {fr_enhanced['balanced_accuracy']['mean']:.6f}")

        print("\n[改善率]")
        print(f"  Accuracy:         {improvements['accuracy']:+.2f}%")
        print(f"  F1 (Macro):       {improvements['f1_macro']:+.2f}%")
        print(f"  MCC:              {improvements['mcc']:+.2f}%")
        print(f"  Balanced Acc:     {improvements['balanced_accuracy']:+.2f}%")

    # JSON出力
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    json_path = results_dir / "three_models_fr_comparison_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": summary,
                "raw_results": {
                    model: {
                        condition: [
                            {k: float(v) for k, v in fold.items()}
                            for fold in folds
                        ]
                        for condition, folds in conditions.items()
                    }
                    for model, conditions in results.items()
                },
                "metadata": {
                    "symbol": args.symbol,
                    "n_splits": args.n_splits,
                    "generated_at": datetime.now().isoformat(),
                },
            },
            f,
            indent=2,
        )

    print(f"\n[SUCCESS] JSON結果を保存しました: {json_path}")

    # Markdownレポート生成
    report_path = results_dir / "three_models_fr_comparison_report.md"
    generate_report(summary, report_path)

    print("\n" + "=" * 70)
    print("完了")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())