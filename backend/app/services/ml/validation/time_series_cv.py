"""
時系列クロスバリデーション

分析報告書で提案された時系列データに適したクロスバリデーション手法を実装。
データリークを防ぎ、実際の取引環境に近い評価を提供します。
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)


class CVStrategy(Enum):
    """クロスバリデーション戦略"""

    TIME_SERIES_SPLIT = "time_series_split"
    WALK_FORWARD = "walk_forward"
    PURGED_CV = "purged_cv"
    EXPANDING_WINDOW = "expanding_window"


@dataclass
class CVConfig:
    """クロスバリデーション設定"""

    strategy: CVStrategy = CVStrategy.TIME_SERIES_SPLIT
    n_splits: int = 5
    max_train_size: Optional[int] = None
    test_size: Optional[int] = None
    gap: int = 0  # Purged CVでのギャップ
    min_train_size: int = 100  # 最小学習サンプル数


class TimeSeriesCrossValidator:
    """
    時系列データ専用クロスバリデーター

    データリークを防ぎ、時系列データの特性を考慮した
    堅牢なクロスバリデーションを提供します。
    """

    def __init__(self, config: "Optional[CVConfig]" = None):
        """
        初期化

        Args:
            config: クロスバリデーション設定
        """
        self.config = config or CVConfig()
        self.cv_results = []

    def _create_splitter(self):
        """分割戦略に応じてスプリッターを作成"""
        if self.config.strategy == CVStrategy.TIME_SERIES_SPLIT:
            return TimeSeriesSplit(
                n_splits=self.config.n_splits,
                max_train_size=self.config.max_train_size,
                test_size=self.config.test_size,
            )
        elif self.config.strategy == CVStrategy.WALK_FORWARD:
            return self._walk_forward_splitter()
        elif self.config.strategy == CVStrategy.PURGED_CV:
            return self._purged_cv_splitter()
        elif self.config.strategy == CVStrategy.EXPANDING_WINDOW:
            return self._expanding_window_splitter()
        else:
            raise ValueError(f"未対応の戦略: {self.config.strategy}")

    def _walk_forward_splitter(self):
        """ウォークフォワード分析用スプリッター"""

        class WalkForwardSplitter:
            def __init__(self, n_splits, test_size):
                self.n_splits = n_splits
                self.test_size = test_size

            def split(self, X):
                n_samples = len(X)
                test_size = self.test_size or n_samples // (self.n_splits + 1)

                for i in range(self.n_splits):
                    test_start = n_samples - (self.n_splits - i) * test_size
                    test_end = test_start + test_size
                    train_end = test_start

                    if train_end < 100:  # 最小学習サイズ
                        continue

                    train_idx = np.arange(0, train_end)
                    test_idx = np.arange(test_start, min(test_end, n_samples))

                    yield train_idx, test_idx

        return WalkForwardSplitter(self.config.n_splits, self.config.test_size)

    def _purged_cv_splitter(self):
        """パージドクロスバリデーション用スプリッター"""

        class PurgedCVSplitter:
            def __init__(self, n_splits, gap):
                self.n_splits = n_splits
                self.gap = gap

            def split(self, X):
                n_samples = len(X)
                test_size = n_samples // self.n_splits

                for i in range(self.n_splits):
                    test_start = i * test_size
                    test_end = min((i + 1) * test_size, n_samples)

                    # ギャップを考慮した学習データ
                    train_idx = np.concatenate(
                        [
                            np.arange(0, max(0, test_start - self.gap)),
                            np.arange(min(test_end + self.gap, n_samples), n_samples),
                        ]
                    )
                    test_idx = np.arange(test_start, test_end)

                    if len(train_idx) < 100:  # 最小学習サイズ
                        continue

                    yield train_idx, test_idx

        return PurgedCVSplitter(self.config.n_splits, self.config.gap)

    def _expanding_window_splitter(self):
        """拡張ウィンドウ用スプリッター"""

        class ExpandingWindowSplitter:
            def __init__(self, n_splits, test_size):
                self.n_splits = n_splits
                self.test_size = test_size

            def split(self, X):
                n_samples = len(X)
                test_size = self.test_size or n_samples // (self.n_splits + 1)

                for i in range(1, self.n_splits + 1):
                    test_start = n_samples - (self.n_splits - i + 1) * test_size
                    test_end = test_start + test_size
                    train_end = test_start

                    train_idx = np.arange(0, train_end)
                    test_idx = np.arange(test_start, min(test_end, n_samples))

                    if len(train_idx) < 100:  # 最小学習サイズ
                        continue

                    yield train_idx, test_idx

        return ExpandingWindowSplitter(self.config.n_splits, self.config.test_size)

    def _validate_data(self, X: pd.DataFrame, y: pd.Series):
        """データの検証"""
        if len(X) != len(y):
            raise ValueError("特徴量とターゲットの長さが一致しません")

        if len(X) < self.config.n_splits * self.config.min_train_size:
            raise ValueError(
                f"データが不足しています: {len(X)} < {self.config.n_splits * self.config.min_train_size}"
            )

        if not isinstance(X.index, pd.DatetimeIndex):
            logger.warning("インデックスがDatetimeIndexではありません")

    def _calculate_scores(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray],
        scoring: List[str],
    ) -> Dict[str, float]:
        """評価指標を計算"""
        scores = {}

        try:
            if "accuracy" in scoring:
                scores["accuracy"] = accuracy_score(y_true, y_pred)

            if any(metric in scoring for metric in ["precision", "recall", "f1"]):
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_true, y_pred, average="weighted", zero_division=0  # type: ignore
                )
                if "precision" in scoring:
                    scores["precision"] = precision
                if "recall" in scoring:
                    scores["recall"] = recall
                if "f1" in scoring:
                    scores["f1"] = f1

            if "roc_auc" in scoring and y_proba is not None:
                try:
                    if y_proba.ndim == 2 and y_proba.shape[1] > 2:
                        # 多クラス分類の場合
                        scores["roc_auc"] = roc_auc_score(
                            y_true, y_proba, multi_class="ovr", average="weighted"
                        )
                    else:
                        # 二値分類の場合
                        scores["roc_auc"] = roc_auc_score(
                            y_true, y_proba[:, 1] if y_proba.ndim == 2 else y_proba
                        )
                except Exception as e:
                    logger.warning(f"ROC-AUC計算エラー: {e}")
                    scores["roc_auc"] = 0.0

        except Exception as e:
            logger.error(f"評価指標計算エラー: {e}")

        return scores

    def _aggregate_results(
        self, scores: Dict[str, List[float]], fold_results: List[Dict]
    ) -> Dict[str, Any]:
        """結果を集計"""
        aggregated = {
            "fold_results": fold_results,
            "n_splits": len(fold_results),
            "strategy": self.config.strategy.value,
        }

        for metric, values in scores.items():
            if values:
                aggregated[f"{metric}_scores"] = values
                aggregated[f"{metric}_mean"] = np.mean(values)
                aggregated[f"{metric}_std"] = np.std(values)
                aggregated[f"{metric}_min"] = np.min(values)
                aggregated[f"{metric}_max"] = np.max(values)

        return aggregated

    def validate_model(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        scoring: Optional[List[str]] = None,
        return_predictions: bool = False,
    ) -> Dict[str, Any]:
        """
        モデルをクロスバリデーションで評価

        Args:
            model: 評価するモデル
            X: 特徴量データ
            y: ターゲットデータ
            scoring: 評価指標のリスト
            return_predictions: 予測結果を返すかどうか

        Returns:
            評価結果の辞書
        """
        if scoring is None:
            scoring = ["accuracy", "precision", "recall", "f1"]

        self._validate_data(X, y)

        splitter = self._create_splitter()
        scores = {metric: [] for metric in scoring}
        fold_results = []
        predictions: List[Dict[str, Any]] = []

        for fold, (train_idx, test_idx) in enumerate(splitter.split(X)):
            try:
                # データ分割
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                # モデル学習
                model.fit(X_train, y_train)

                # 予測
                y_pred = model.predict(X_test)
                y_proba = getattr(model, 'predict_proba', lambda x: None)(X_test)

                # 評価指標計算
                fold_scores = self._calculate_scores(y_test, y_pred, y_proba, scoring)

                # 結果保存
                fold_result = {
                    "fold": fold,
                    "train_size": len(train_idx),
                    "test_size": len(test_idx),
                    **fold_scores,
                }
                fold_results.append(fold_result)

                for metric in scoring:
                    if metric in fold_scores:
                        scores[metric].append(fold_scores[metric])

                if return_predictions:
                    predictions.extend(
                        [
                            {
                                "fold": fold,
                                "index": idx,
                                "true": true_val,
                                "pred": pred_val,
                                "proba": proba_val.tolist() if proba_val is not None else None,
                            }
                            for idx, (true_val, pred_val, proba_val) in zip(
                                test_idx,
                                zip(y_test, y_pred, y_proba if y_proba is not None else [None] * len(y_test))
                            )
                        ]
                    )

            except Exception as e:
                logger.error(f"フォールド {fold} の評価でエラーが発生: {e}")
                continue

        result = self._aggregate_results(scores, fold_results)

        if return_predictions:
            result["predictions"] = predictions

        return result
