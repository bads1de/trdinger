"""
ラベル生成ユーティリティ

価格変化率から適切なラベルを生成するためのユーティリティ関数を提供します。
動的閾値計算機能により、データの特性に応じた最適なラベル分布を実現します。
"""

import logging
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class ThresholdMethod(Enum):
    """閾値計算方法"""

    FIXED = "fixed"  # 固定閾値
    QUANTILE = "quantile"  # 分位数ベース
    STD_DEVIATION = "std_deviation"  # 標準偏差ベース
    ADAPTIVE = "adaptive"  # 適応的閾値
    DYNAMIC_VOLATILITY = "dynamic_volatility"  # 動的ボラティリティベース


class LabelGenerator:
    """
    ラベル生成クラス

    価格変化率から3クラス分類（上昇・下落・レンジ）のラベルを生成します。
    データの特性に応じて動的に閾値を調整する機能を提供します。
    """

    def __init__(self):
        """初期化"""
        pass

    def generate_labels(
        self,
        price_data: pd.Series,
        method: ThresholdMethod = ThresholdMethod.STD_DEVIATION,
        target_distribution: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> Tuple[pd.Series, Dict[str, Any]]:
        """
        価格データからラベルを生成

        Args:
            price_data: 価格データ（Close価格）
            method: 閾値計算方法
            target_distribution: 目標分布（例: {"up": 0.33, "down": 0.33, "range": 0.34}）
            **kwargs: 各方法固有のパラメータ

        Returns:
            ラベルSeries, 閾値情報の辞書
        """
        try:
            # 価格変化率を計算
            price_change = price_data.pct_change().shift(-1)

            # NaNを除去
            valid_mask = price_change.notna()
            price_change_clean = price_change[valid_mask]

            if len(price_change_clean) == 0:
                raise ValueError("有効な価格変化率データがありません")

            # 閾値を計算
            threshold_info = self._calculate_thresholds(
                price_change_clean, method, target_distribution, **kwargs
            )

            threshold_up = threshold_info["threshold_up"]
            threshold_down = threshold_info["threshold_down"]

            # ラベルを生成
            labels = pd.Series(
                1, index=price_change.index, dtype=int
            )  # デフォルト：レンジ
            labels[price_change > threshold_up] = 2  # 上昇
            labels[price_change < threshold_down] = 0  # 下落

            # 最後の行は予測できないので除外
            labels = labels.iloc[:-1]

            # 分布情報を追加
            label_counts = labels.value_counts().sort_index()
            total_count = len(labels)

            distribution_info = {
                "down_count": label_counts.get(0, 0),
                "range_count": label_counts.get(1, 0),
                "up_count": label_counts.get(2, 0),
                "down_ratio": label_counts.get(0, 0) / total_count,
                "range_ratio": label_counts.get(1, 0) / total_count,
                "up_ratio": label_counts.get(2, 0) / total_count,
                "total_count": total_count,
            }

            threshold_info.update(distribution_info)

            logger.info(
                f"ラベル生成完了: 上昇={distribution_info['up_count']}({distribution_info['up_ratio']*100:.1f}%), "
                f"下落={distribution_info['down_count']}({distribution_info['down_ratio']*100:.1f}%), "
                f"レンジ={distribution_info['range_count']}({distribution_info['range_ratio']*100:.1f}%)"
            )

            return labels, threshold_info

        except Exception as e:
            logger.error(f"ラベル生成エラー: {e}")
            raise

    def _calculate_thresholds(
        self,
        price_change: pd.Series,
        method: ThresholdMethod,
        target_distribution: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        閾値を計算

        Args:
            price_change: 価格変化率データ
            method: 閾値計算方法
            target_distribution: 目標分布
            **kwargs: 各方法固有のパラメータ

        Returns:
            閾値情報の辞書
        """
        if method == ThresholdMethod.FIXED:
            return self._calculate_fixed_thresholds(price_change, **kwargs)
        elif method == ThresholdMethod.QUANTILE:
            return self._calculate_quantile_thresholds(
                price_change, target_distribution, **kwargs
            )
        elif method == ThresholdMethod.STD_DEVIATION:
            return self._calculate_std_thresholds(price_change, **kwargs)
        elif method == ThresholdMethod.ADAPTIVE:
            return self._calculate_adaptive_thresholds(
                price_change, target_distribution, **kwargs
            )
        elif method == ThresholdMethod.DYNAMIC_VOLATILITY:
            return self._calculate_dynamic_volatility_thresholds(
                price_change, **kwargs
            )
        else:
            raise ValueError(f"未対応の閾値計算方法: {method}")

    def _calculate_fixed_thresholds(
        self, price_change: pd.Series, threshold: float = 0.02, **kwargs
    ) -> Dict[str, Any]:
        """固定閾値を計算"""
        threshold_up = threshold
        threshold_down = -threshold

        return {
            "method": "fixed",
            "threshold_up": threshold_up,
            "threshold_down": threshold_down,
            "threshold_value": threshold,
            "description": f"固定閾値±{threshold*100:.2f}%",
        }

    def _calculate_quantile_thresholds(
        self,
        price_change: pd.Series,
        target_distribution: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """分位数ベースの閾値を計算"""
        if target_distribution is None:
            # デフォルト：各クラス約33%
            down_ratio = 0.33
            up_ratio = 0.33
        else:
            down_ratio = target_distribution.get("down", 0.33)
            up_ratio = target_distribution.get("up", 0.33)

        # 分位数を計算
        threshold_down = price_change.quantile(down_ratio)
        threshold_up = price_change.quantile(1 - up_ratio)

        return {
            "method": "quantile",
            "threshold_up": threshold_up,
            "threshold_down": threshold_down,
            "target_down_ratio": down_ratio,
            "target_up_ratio": up_ratio,
            "description": f"分位数ベース（下落{down_ratio*100:.0f}%、上昇{up_ratio*100:.0f}%）",
        }

    def _calculate_std_thresholds(
        self, price_change: pd.Series, std_multiplier: float = 0.25, **kwargs
    ) -> Dict[str, Any]:
        """標準偏差ベースの閾値を計算"""
        std_value = price_change.std()
        threshold_up = std_multiplier * std_value
        threshold_down = -std_multiplier * std_value

        return {
            "method": "std_deviation",
            "threshold_up": threshold_up,
            "threshold_down": threshold_down,
            "std_multiplier": std_multiplier,
            "std_value": std_value,
            "description": f"{std_multiplier}標準偏差ベース",
        }

    def _calculate_adaptive_thresholds(
        self,
        price_change: pd.Series,
        target_distribution: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """適応的閾値を計算（複数の方法を試して最適なものを選択）"""
        if target_distribution is None:
            target_distribution = {"up": 0.33, "down": 0.33, "range": 0.34}

        # 複数の方法を試す
        methods_to_try = [
            (ThresholdMethod.STD_DEVIATION, {"std_multiplier": 0.25}),
            (ThresholdMethod.STD_DEVIATION, {"std_multiplier": 0.5}),
            (ThresholdMethod.STD_DEVIATION, {"std_multiplier": 0.75}),
            (ThresholdMethod.QUANTILE, {"target_distribution": target_distribution}),
            (ThresholdMethod.FIXED, {"threshold": 0.005}),
            (ThresholdMethod.FIXED, {"threshold": 0.01}),
            (ThresholdMethod.FIXED, {"threshold": 0.015}),
        ]

        best_method = None
        best_score = float("inf")
        best_info = None

        for method, params in methods_to_try:
            try:
                # 閾値を計算
                threshold_info = self._calculate_thresholds(
                    price_change, method, target_distribution, **params
                )

                # 実際の分布を計算
                threshold_up = threshold_info["threshold_up"]
                threshold_down = threshold_info["threshold_down"]

                up_count = (price_change > threshold_up).sum()
                down_count = (price_change < threshold_down).sum()
                range_count = len(price_change) - up_count - down_count
                total_count = len(price_change)

                actual_distribution = {
                    "up": up_count / total_count,
                    "down": down_count / total_count,
                    "range": range_count / total_count,
                }

                # 目標分布との差を計算（スコアが小さいほど良い）
                score = sum(
                    abs(actual_distribution[key] - target_distribution[key])
                    for key in target_distribution.keys()
                )

                if score < best_score:
                    best_score = score
                    best_method = method
                    best_info = threshold_info.copy()
                    best_info.update(
                        {
                            "actual_distribution": actual_distribution,
                            "balance_score": score,
                        }
                    )

            except Exception as e:
                logger.warning(f"適応的閾値計算で方法 {method} が失敗: {e}")
                continue

        if best_info is None:
            # フォールバック：標準偏差ベース
            logger.warning("適応的閾値計算が失敗、標準偏差ベースにフォールバック")
            return self._calculate_std_thresholds(price_change, std_multiplier=0.25)

        best_info["method"] = "adaptive"
        best_info["best_underlying_method"] = best_method.value
        best_info["description"] = f"適応的閾値（最適方法: {best_info['description']}）"

        return best_info

    @staticmethod
    def validate_label_distribution(
        labels: pd.Series, min_class_ratio: float = 0.05, max_class_ratio: float = 0.95
    ) -> Dict[str, Any]:
        """
        ラベル分布を検証

        Args:
            labels: ラベルSeries
            min_class_ratio: 最小クラス比率
            max_class_ratio: 最大クラス比率

        Returns:
            検証結果の辞書
        """
        label_counts = labels.value_counts().sort_index()
        total_count = len(labels)

        validation_result = {
            "is_valid": True,
            "warnings": [],
            "errors": [],
            "distribution": {},
        }

        for label in [0, 1, 2]:  # 下落、レンジ、上昇
            count = label_counts.get(label, 0)
            ratio = count / total_count if total_count > 0 else 0

            class_name = ["down", "range", "up"][label]
            validation_result["distribution"][class_name] = {
                "count": count,
                "ratio": ratio,
            }

            if ratio < min_class_ratio:
                validation_result["warnings"].append(
                    f"{class_name}クラスの比率が低すぎます: {ratio*100:.2f}% < {min_class_ratio*100:.2f}%"
                )

            if ratio > max_class_ratio:
                validation_result["errors"].append(
                    f"{class_name}クラスの比率が高すぎます: {ratio*100:.2f}% > {max_class_ratio*100:.2f}%"
                )
                validation_result["is_valid"] = False

        # 1クラスしかない場合の特別チェック
        unique_labels = labels.nunique()
        if unique_labels <= 1:
            validation_result["errors"].append(
                f"ラベルが{unique_labels}種類しかありません。機械学習には最低2種類必要です。"
            )
            validation_result["is_valid"] = False

        return validation_result

    def _calculate_dynamic_volatility_thresholds(
        self,
        price_change: pd.Series,
        volatility_window: int = 24,
        threshold_multiplier: float = 0.5,
        min_threshold: float = 0.005,
        max_threshold: float = 0.05,
        **kwargs
    ) -> Dict[str, Any]:
        """
        動的ボラティリティベースの閾値を計算

        市場のボラティリティに応じて動的に閾値を調整し、
        クラス分布の均衡を保つ。

        Args:
            price_change: 価格変化率データ
            volatility_window: ボラティリティ計算ウィンドウ
            threshold_multiplier: 閾値乗数
            min_threshold: 最小閾値
            max_threshold: 最大閾値

        Returns:
            閾値情報の辞書
        """
        # ローリングボラティリティを計算
        volatility = price_change.rolling(window=volatility_window).std()

        # 動的閾値を計算
        dynamic_threshold = volatility * threshold_multiplier

        # 閾値を最小・最大値でクリップ
        dynamic_threshold = dynamic_threshold.clip(min_threshold, max_threshold)

        # 平均閾値を計算（NaNを除外）
        avg_threshold = dynamic_threshold.dropna().mean()

        # 上昇・下落閾値を設定
        threshold_up = avg_threshold
        threshold_down = -avg_threshold

        # 統計情報を計算
        volatility_stats = {
            'mean_volatility': volatility.dropna().mean(),
            'std_volatility': volatility.dropna().std(),
            'min_volatility': volatility.dropna().min(),
            'max_volatility': volatility.dropna().max(),
        }

        threshold_stats = {
            'mean_threshold': avg_threshold,
            'min_threshold_used': dynamic_threshold.dropna().min(),
            'max_threshold_used': dynamic_threshold.dropna().max(),
            'threshold_std': dynamic_threshold.dropna().std(),
        }

        return {
            "method": "dynamic_volatility",
            "threshold_up": threshold_up,
            "threshold_down": threshold_down,
            "volatility_window": volatility_window,
            "threshold_multiplier": threshold_multiplier,
            "min_threshold": min_threshold,
            "max_threshold": max_threshold,
            "volatility_stats": volatility_stats,
            "threshold_stats": threshold_stats,
            "description": f"動的ボラティリティベース（平均閾値±{avg_threshold*100:.2f}%）",
        }

    def generate_labels_with_dynamic_threshold(
        self,
        price_data: pd.Series,
        volatility_window: int = 24,
        threshold_multiplier: float = 0.5,
        min_threshold: float = 0.005,
        max_threshold: float = 0.05,
    ) -> Tuple[pd.Series, Dict[str, Any]]:
        """
        ボラティリティに基づく動的閾値でラベルを生成

        改善計画で提案された動的閾値機能の実装。
        市場のボラティリティに応じて適応的に閾値を調整する。

        Args:
            price_data: 価格データ（Close価格）
            volatility_window: ボラティリティ計算ウィンドウ
            threshold_multiplier: 閾値乗数
            min_threshold: 最小閾値
            max_threshold: 最大閾値

        Returns:
            ラベルSeries, 閾値情報の辞書
        """
        return self.generate_labels(
            price_data,
            method=ThresholdMethod.DYNAMIC_VOLATILITY,
            volatility_window=volatility_window,
            threshold_multiplier=threshold_multiplier,
            min_threshold=min_threshold,
            max_threshold=max_threshold,
        )
