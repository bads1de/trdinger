"""
ラベル生成ユーティリティ

価格変化率から適切なラベルを生成するためのユーティリティ関数を提供します。
動的閾値計算機能により、データの特性に応じた最適なラベル分布を実現します。
"""

import logging
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

logger = logging.getLogger(__name__)


class ThresholdMethod(Enum):
    """閾値計算方法"""

    FIXED = "fixed"  # 固定閾値
    QUANTILE = "quantile"  # 分位数ベース
    # 後方互換エイリアス（テストでのPERCENTILE呼称に対応）
    PERCENTILE = "quantile"
    STD_DEVIATION = "std_deviation"  # 標準偏差ベース
    ADAPTIVE = "adaptive"  # 適応的閾値
    DYNAMIC_VOLATILITY = "dynamic_volatility"  # 動的ボラティリティベース
    KBINS_DISCRETIZER = "kbins_discretizer"  # KBinsDiscretizerベース


class LabelGenerator:
    """
    ラベル生成クラス

    価格変化率から3クラス分類（上昇・下落・レンジ）のラベルを生成します。
    データの特性に応じて動的に閾値を調整する機能を提供します。
    """

    def __init__(self):
        """初期化"""

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
            # DataFrame入力 + target_column指定のサポート
            # tests/label_generation では DataFrame と target_column を渡す呼び方がある
            target_column = kwargs.pop("target_column", None)
            series_input: pd.Series
            if isinstance(price_data, pd.DataFrame):
                if not target_column or target_column not in price_data.columns:
                    raise ValueError("target_column が必要です")
                series_input = price_data[target_column]
            else:
                series_input = price_data

            # 価格変化率を計算
            # - Series入力: 次期変化率（forward）で学習用に整合
            # - DataFrame+target_column入力: 過去→現在（backward）でテスト互換
            if target_column is not None:
                price_change = series_input.pct_change()
            else:
                price_change = series_input.pct_change().shift(-1)

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

            # 返却仕様の互換性:
            # - Series入力時: (labels, threshold_info) のタプル
            # - DataFrame+target_column入力時: labels のみ
            if target_column is not None:
                return labels  # type: ignore[return-value]
            else:
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
            return self._calculate_dynamic_volatility_thresholds(price_change, **kwargs)
        elif method == ThresholdMethod.KBINS_DISCRETIZER:
            return self._calculate_kbins_discretizer_thresholds(
                price_change, target_distribution, **kwargs
            )
        else:
            raise ValueError(f"未対応の閾値計算方法: {method}")

    def _calculate_fixed_thresholds(
        self, price_change: pd.Series, threshold: float = 0.02, **kwargs
    ) -> Dict[str, Any]:
        """固定閾値を計算

        優先順位:
        1) threshold_up/threshold_down が与えられればそれを使用
        2) threshold（正値）から対称閾値 ±threshold を構成
        3) デフォルト 0.02（±2%）
        """
        thr_up = kwargs.get("threshold_up", None)
        thr_dn = kwargs.get("threshold_down", None)

        if isinstance(thr_up, (int, float)) and isinstance(thr_dn, (int, float)):
            threshold_up = float(thr_up)
            threshold_down = float(thr_dn)
            base_thr = max(abs(threshold_up), abs(threshold_down))
        else:
            threshold_up = float(threshold)
            threshold_down = -float(threshold)
            base_thr = float(threshold)

        return {
            "method": "fixed",
            "threshold_up": threshold_up,
            "threshold_down": threshold_down,
            "threshold_value": base_thr,
            "description": f"固定閾値（上{threshold_up*100:.2f}%, 下{threshold_down*100:.2f}%）",
        }

    def _calculate_quantile_thresholds(
        self,
        price_change: pd.Series,
        target_distribution: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """分位数ベースの閾値を計算

        サポート:
        - threshold_up/threshold_down を [0,1] の分位として直接指定（推奨）
        - target_distribution={'up': r_up, 'down': r_down} による比率指定（従来互換）
        """
        # 優先: 明示的な分位パラメータ
        perc_up = kwargs.get("threshold_up", None)
        perc_down = kwargs.get("threshold_down", None)
        if (
            isinstance(perc_up, (int, float))
            and isinstance(perc_down, (int, float))
            and 0.0 <= float(perc_up) <= 1.0
            and 0.0 <= float(perc_down) <= 1.0
        ):
            threshold_down = price_change.quantile(float(perc_down))
            threshold_up = price_change.quantile(float(perc_up))
            return {
                "method": "quantile",
                "threshold_up": threshold_up,
                "threshold_down": threshold_down,
                "target_down_ratio": perc_down,
                "target_up_ratio": 1 - perc_up,  # 上側割合（参考情報）
                "description": f"分位数ベース（ダウン{perc_down*100:.0f}パーセンタイル、アップ{perc_up*100:.0f}パーセンタイル）",
            }

        # フォールバック: 目標分布に基づく比率指定
        if target_distribution is None:
            # デフォルト：各クラス約33%
            down_ratio = 0.33
            up_ratio = 0.33
        else:
            down_ratio = target_distribution.get("down", 0.33)
            up_ratio = target_distribution.get("up", 0.33)

        # 分位数を計算（上側は 1 - up_ratio の分位値）
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

        # ユーザー指定の閾値があれば最優先でその解釈に従い、最適化ロジックをバイパス
        user_thr_up = kwargs.get("threshold_up", None)
        user_thr_dn = kwargs.get("threshold_down", None)
        if isinstance(user_thr_up, (int, float)) and isinstance(
            user_thr_dn, (int, float)
        ):
            # 分位数指定（0-1）
            if 0.0 <= float(user_thr_up) <= 1.0 and 0.0 <= float(user_thr_dn) <= 1.0:
                threshold_down = price_change.quantile(float(user_thr_dn))
                threshold_up = price_change.quantile(float(user_thr_up))
                return {
                    "method": "adaptive_quantile",
                    "threshold_up": threshold_up,
                    "threshold_down": threshold_down,
                    "target_down_ratio": float(user_thr_dn),
                    "target_up_ratio": 1 - float(user_thr_up),
                    "description": f"適応（ユーザー分位: ダウン{float(user_thr_dn)*100:.0f}%, アップ{float(user_thr_up)*100:.0f}%）",
                }
            # 標準偏差倍率指定
            else:
                std_value = price_change.std()
                std_mult = max(abs(float(user_thr_up)), abs(float(user_thr_dn)))
                threshold_up = std_mult * std_value
                threshold_down = -std_mult * std_value
                return {
                    "method": "adaptive_std_deviation",
                    "threshold_up": threshold_up,
                    "threshold_down": threshold_down,
                    "std_multiplier": std_mult,
                    "std_value": std_value,
                    "description": f"適応（ユーザー指定 {std_mult}σ）",
                }

        # ここからは最適化ロジック（ユーザー指定がない場合）
        methods_to_try = []  # type: list[tuple[ThresholdMethod, dict]]

        # 候補の追加
        methods_to_try.extend(
            [
                (ThresholdMethod.STD_DEVIATION, {"std_multiplier": 0.25}),
                (ThresholdMethod.STD_DEVIATION, {"std_multiplier": 0.5}),
                (ThresholdMethod.STD_DEVIATION, {"std_multiplier": 0.75}),
                (
                    ThresholdMethod.QUANTILE,
                    {"target_distribution": target_distribution},
                ),
                (ThresholdMethod.FIXED, {"threshold": 0.005}),
                (ThresholdMethod.FIXED, {"threshold": 0.01}),
                (ThresholdMethod.FIXED, {"threshold": 0.015}),
            ]
        )

        best_method = None
        best_score = float("inf")
        best_info: Optional[Dict[str, Any]] = None

        for method, params in methods_to_try:
            try:
                # 閾値を計算（target_distribution は kwargs に含めるよう統一し、重複引数を避ける）
                if (
                    method == ThresholdMethod.QUANTILE
                    and "target_distribution" not in params
                ):
                    params = {**params, "target_distribution": target_distribution}

                threshold_info = self._calculate_thresholds(
                    price_change, method, None, **params  # type: ignore[arg-type]
                )

                # 実際の分布を計算
                threshold_up_val = threshold_info["threshold_up"]
                threshold_down_val = threshold_info["threshold_down"]

                up_count = (price_change > threshold_up_val).sum()
                down_count = (price_change < threshold_down_val).sum()
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
        **kwargs,
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
            "mean_volatility": volatility.dropna().mean(),
            "std_volatility": volatility.dropna().std(),
            "min_volatility": volatility.dropna().min(),
            "max_volatility": volatility.dropna().max(),
        }

        threshold_stats = {
            "mean_threshold": avg_threshold,
            "min_threshold_used": dynamic_threshold.dropna().min(),
            "max_threshold_used": dynamic_threshold.dropna().max(),
            "threshold_std": dynamic_threshold.dropna().std(),
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

    def _calculate_kbins_discretizer_thresholds(
        self,
        price_change: pd.Series,
        target_distribution: Optional[Dict[str, float]] = None,
        strategy: str = "quantile",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        KBinsDiscretizerを使った閾値計算

        scikit-learnのKBinsDiscretizerを使用して、複雑な条件分岐を
        簡素化し、より効率的で保守性の高い実装を提供します。

        Args:
            price_change: 価格変化率データ
            target_distribution: 目標分布（使用されませんが互換性のため保持）
            strategy: 分割戦略 ('uniform', 'quantile', 'kmeans')
            **kwargs: その他のパラメータ

        Returns:
            閾値情報の辞書
        """
        try:
            # 価格変化率を2次元配列に変換（KBinsDiscretizerの要求）
            price_change_clean = price_change.dropna()
            if len(price_change_clean) == 0:
                raise ValueError("有効な価格変化率データがありません")

            X = price_change_clean.values.reshape(-1, 1)

            # KBinsDiscretizerで3つのビンに分割
            discretizer = KBinsDiscretizer(
                n_bins=3,
                encode="ordinal",
                strategy=strategy,
                subsample=None,  # 全データを使用
            )

            # フィットして境界値を取得
            discretizer.fit(X)
            bin_edges = discretizer.bin_edges_[0]  # 最初の特徴量の境界値

            # 閾値を設定（下落/レンジ/上昇の境界）
            threshold_down = bin_edges[1]  # 下落とレンジの境界
            threshold_up = bin_edges[2]  # レンジと上昇の境界

            # 実際の分布を計算
            up_count = (price_change_clean > threshold_up).sum()
            down_count = (price_change_clean < threshold_down).sum()
            range_count = len(price_change_clean) - up_count - down_count
            total_count = len(price_change_clean)

            actual_distribution = {
                "up": up_count / total_count,
                "down": down_count / total_count,
                "range": range_count / total_count,
            }

            return {
                "method": "kbins_discretizer",
                "threshold_up": threshold_up,
                "threshold_down": threshold_down,
                "strategy": strategy,
                "bin_edges": bin_edges.tolist(),
                "actual_distribution": actual_distribution,
                "n_samples": total_count,
                "description": f"KBinsDiscretizer（{strategy}戦略、3ビン分割）",
            }

        except Exception as e:
            logger.error(f"KBinsDiscretizer閾値計算エラー: {e}")
            # フォールバック：分位数ベース
            logger.warning("KBinsDiscretizer計算が失敗、分位数ベースにフォールバック")
            return self._calculate_quantile_thresholds(
                price_change, target_distribution
            )

    def generate_labels_with_kbins_discretizer(
        self,
        price_data: pd.Series,
        strategy: str = "quantile",
    ) -> Tuple[pd.Series, Dict[str, Any]]:
        """
        KBinsDiscretizerを使用したラベル生成

        レポートで提案されたKBinsDiscretizerによる簡素化実装。
        複雑な条件分岐を排除し、scikit-learnの標準機能を活用します。

        Args:
            price_data: 価格データ（Close価格）
            strategy: 分割戦略 ('uniform', 'quantile', 'kmeans')

        Returns:
            ラベルSeries, 閾値情報の辞書
        """
        return self.generate_labels(
            price_data,
            method=ThresholdMethod.KBINS_DISCRETIZER,
            strategy=strategy,
        )
