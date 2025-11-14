"""
ラベル生成ユーティリティ関数群

Pipeline作成、GridSearch最適化などのユーティリティ関数を提供します。
"""

import logging
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer

from .enums import ThresholdMethod  # noqa: F401
from .main import LabelGenerator  # noqa: F401
from .transformer import PriceChangeTransformer

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
