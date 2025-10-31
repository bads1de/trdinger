"""
ラベル生成ユーティリティ関数群

Pipeline作成、AutoML用ターゲット計算、GridSearch最適化などのユーティリティ関数を提供します。
"""

import logging
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from .transformer import PriceChangeTransformer
from .main import LabelGenerator
from .enums import ThresholdMethod

logger = logging.getLogger(__name__)


def create_label_pipeline(
    n_bins: int = 3, strategy: str = "quantile", encode: str = "ordinal"
) -> Pipeline:
    """
    ラベル生成用のPipelineを作成（関数ベース）

    Args:
        n_bins: ビン数（デフォルト3: 下落、レンジ、上昇）
        strategy: 分割戦略（'uniform', 'quantile', 'kmeans'）
        encode: エンコード方法（'ordinal', 'onehot'）

    Returns:
        scikit-learnのPipeline
    """
    return Pipeline(
        [
            ("price_change", PriceChangeTransformer()),
            (
                "discretizer",
                KBinsDiscretizer(
                    n_bins=n_bins,
                    encode=encode,
                    strategy=strategy,  # subsample=None
                ),
            ),
        ]
    )


def calculate_target_for_automl(
    ohlcv_data: pd.DataFrame, config: Any = None
) -> Optional[pd.Series]:
    """
    AutoML特徴量生成用のターゲット変数を計算

    BaseMLTrainerから移管されたロジック。
    ラベル生成の責務を一元管理するため、このモジュールに配置。

    Args:
        ohlcv_data: OHLCVデータ
        config: 設定オブジェクト（オプション）

    Returns:
        ターゲット変数のSeries（計算できない場合はNone）
    """
    try:
        if ohlcv_data.empty or "Close" not in ohlcv_data.columns:
            logger.warning("ターゲット変数計算用のデータが不足しています")
            return None

        # 価格変化率を計算（次の期間の価格変化）
        close_prices = ohlcv_data["Close"].copy()

        # 将来の価格変化率を計算（24時間後の変化率）
        prediction_horizon = 24
        if config and hasattr(config, "training"):
            prediction_horizon = getattr(config.training, "PREDICTION_HORIZON", 24)

        future_returns = close_prices.pct_change(periods=prediction_horizon).shift(
            -prediction_horizon
        )

        # 動的閾値を使用してクラス分類
        label_generator = LabelGenerator()

        # 設定から動的閾値パラメータを取得
        label_method = "dynamic_volatility"
        if config and hasattr(config, "training"):
            label_method = getattr(
                config.training, "LABEL_METHOD", "dynamic_volatility"
            )

        if label_method == "dynamic_volatility":
            # 動的ボラティリティベースのラベル生成
            volatility_window = 24
            threshold_multiplier = 0.5
            min_threshold = 0.005
            max_threshold = 0.05

            if config and hasattr(config, "training"):
                volatility_window = getattr(config.training, "VOLATILITY_WINDOW", 24)
                threshold_multiplier = getattr(
                    config.training, "THRESHOLD_MULTIPLIER", 0.5
                )
                min_threshold = getattr(config.training, "MIN_THRESHOLD", 0.005)
                max_threshold = getattr(config.training, "MAX_THRESHOLD", 0.05)

            labels, threshold_info = label_generator.generate_labels(
                ohlcv_data["Close"],  # type: ignore[arg-type]
                method=ThresholdMethod.DYNAMIC_VOLATILITY,
                volatility_window=volatility_window,
                threshold_multiplier=threshold_multiplier,
                min_threshold=min_threshold,
                max_threshold=max_threshold,
            )
            target = labels
        else:
            # 従来の固定閾値（後方互換性）
            threshold_up = 0.02
            threshold_down = -0.02

            if config and hasattr(config, "training"):
                threshold_up = getattr(config.training, "THRESHOLD_UP", 0.02)
                threshold_down = getattr(config.training, "THRESHOLD_DOWN", -0.02)

            # 3クラス分類：0=下落、1=横ばい、2=上昇
            target = pd.Series(1, index=future_returns.index)  # デフォルトは横ばい
            target[future_returns > threshold_up] = 2  # 上昇
            target[future_returns < threshold_down] = 0  # 下落

        # NaNを除去
        target = target.dropna()

        logger.info(f"AutoML用ターゲット変数を計算: {len(target)}サンプル")
        logger.info(
            f"クラス分布 - 下落: {(target == 0).sum()}, 横ばい: {(target == 1).sum()}, 上昇: {(target == 2).sum()}"
        )

        return target

    except Exception as e:
        logger.warning(f"AutoML用ターゲット変数計算エラー: {e}")
        return None


def optimize_label_generation_with_gridsearch(
    price_data: pd.Series,
    param_grid: Optional[Dict[str, Any]] = None,
    cv: int = 3,
    scoring: str = "balanced_accuracy",
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    GridSearchCVを使用したラベル生成の最適化

    ハイパーパラメータ最適化をPipelineに組み込んだ高度なアプローチ。

    Args:
        price_data: 価格データ（Close価格）
        param_grid: パラメータグリッド
        cv: クロスバリデーションの分割数
        scoring: スコアリング方法

    Returns:
        最適化されたPipeline, 結果情報の辞書
    """
    if param_grid is None:
        param_grid = {
            "discretizer__strategy": ["uniform", "quantile", "kmeans"],
            "discretizer__n_bins": [3, 4, 5],
        }

    logger.info(f"GridSearch ラベル生成最適化開始: {len(price_data)}行")

    # ベースPipelineを作成
    base_pipeline = create_label_pipeline()

    # 価格変化率を計算（ターゲット用）
    price_change = price_data.pct_change()

    # 簡単なターゲット（価格上昇/下降の2値分類）を作成
    simple_target = (price_change > 0).astype(int)

    # 学習データとターゲットの整合（同一長・同一インデックス）
    aligned = pd.concat({"X": price_data, "y": simple_target}, axis=1).dropna()
    X_aligned = aligned["X"]
    y_aligned = aligned["y"].astype(int)

    # GridSearchCVを実行
    grid_search = GridSearchCV(
        base_pipeline, param_grid=param_grid, cv=cv, scoring=scoring, n_jobs=-1
    )

    grid_search.fit(X_aligned, y_aligned)

    # 最適なパラメータでラベル生成
    best_pipeline = grid_search.best_estimator_
    labels_array = best_pipeline.transform(price_data)

    # 価格変化率のインデックスを取得
    price_change_index = price_data.pct_change().dropna().index

    # Seriesとして変換
    labels = pd.Series(labels_array.flatten(), index=price_change_index, dtype=int)

    # 結果情報
    info = {
        "method": "gridsearch_optimized",
        "best_params": grid_search.best_params_,
        "best_score": grid_search.best_score_,
        "cv_results": grid_search.cv_results_,
        "labels": labels,
    }

    logger.info(
        f"GridSearch ラベル生成最適化完了: "
        f"最適パラメータ={grid_search.best_params_}, "
        f"スコア={grid_search.best_score_:.3f}"
    )

    return best_pipeline, info
