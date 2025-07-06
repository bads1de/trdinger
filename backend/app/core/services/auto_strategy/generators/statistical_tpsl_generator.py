"""
統計的優位性ベースのTP/SL生成機能

このモジュールは、過去のバックテスト結果や市場データから学習した
統計的優位性に基づいて、最適なテイクプロフィット（TP）と
ストップロス（SL）を生成する機能を提供します。
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


logger = logging.getLogger(__name__)


class OptimizationObjective(Enum):
    """最適化目標の種類"""

    SHARPE_RATIO = "sharpe_ratio"
    TOTAL_RETURN = "total_return"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    MAX_DRAWDOWN = "max_drawdown"


@dataclass
class StatisticalConfig:
    """統計的TP/SL生成の設定"""

    optimization_objective: OptimizationObjective = OptimizationObjective.SHARPE_RATIO
    lookback_period_days: int = 365  # 学習期間（日）
    min_sample_size: int = 50  # 最小サンプル数
    confidence_threshold: float = 0.7  # 信頼度閾値
    symbol_specific: bool = True  # シンボル固有の最適化
    timeframe_specific: bool = True  # 時間軸固有の最適化
    market_regime_aware: bool = True  # 市場レジーム考慮
    performance_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "sharpe_ratio": 0.3,
            "total_return": 0.25,
            "win_rate": 0.2,
            "max_drawdown": 0.15,
            "profit_factor": 0.1,
        }
    )


@dataclass
class StatisticalResult:
    """統計的TP/SL生成結果"""

    stop_loss_pct: float
    take_profit_pct: float
    expected_performance: Dict[str, float]
    confidence_score: float
    sample_size: int
    optimization_objective: str
    historical_performance: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


class StatisticalTPSLGenerator:
    """
    統計的優位性ベースのTP/SL生成機能

    過去のバックテスト結果や市場データから学習した統計的優位性に基づいて、
    最適なTP/SL設定を提供します。
    """

    def __init__(self, db_path: Optional[str] = None):
        """ジェネレーターを初期化"""
        self.logger = logging.getLogger(__name__)

        # データベース接続（将来的な実装用）
        self.db_path = db_path or "data/statistical_tpsl.db"

        # サンプル統計データ（実際の実装では外部データソースから取得）
        self.sample_data = self._initialize_sample_data()

        # 市場レジーム分類
        self.market_regimes = {
            "bull_market": {"trend": "up", "volatility": "low"},
            "bear_market": {"trend": "down", "volatility": "high"},
            "sideways": {"trend": "neutral", "volatility": "medium"},
            "high_volatility": {"trend": "mixed", "volatility": "very_high"},
        }

    def generate_statistical_tpsl(
        self,
        config: StatisticalConfig,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        market_conditions: Optional[Dict[str, Any]] = None,
    ) -> StatisticalResult:
        """
        統計的優位性に基づいてTP/SLを生成

        Args:
            config: 統計的生成設定
            symbol: 取引シンボル
            timeframe: 時間軸
            market_conditions: 現在の市場条件

        Returns:
            統計的TP/SL生成結果
        """
        try:
            self.logger.info(
                f"統計的TP/SL生成開始: 目標={config.optimization_objective.value}, "
                f"シンボル={symbol}, 時間軸={timeframe}"
            )

            # 関連する過去データを取得
            historical_data = self._get_historical_performance_data(
                symbol, timeframe, config
            )

            # 市場レジームを判定
            current_regime = self._determine_market_regime(market_conditions)

            # レジーム固有のデータをフィルタリング
            regime_data = self._filter_by_market_regime(historical_data, current_regime)

            # 最適なTP/SL組み合わせを見つける
            optimal_tpsl = self._find_optimal_tpsl_combination(regime_data, config)

            # パフォーマンス予測を計算
            expected_performance = self._calculate_expected_performance(
                optimal_tpsl, regime_data, config
            )

            # 信頼度スコアを計算
            confidence_score = self._calculate_statistical_confidence(
                regime_data, optimal_tpsl, config
            )

            result = StatisticalResult(
                stop_loss_pct=optimal_tpsl["sl"],
                take_profit_pct=optimal_tpsl["tp"],
                expected_performance=expected_performance,
                confidence_score=confidence_score,
                sample_size=len(regime_data),
                optimization_objective=config.optimization_objective.value,
                historical_performance=self._summarize_historical_performance(
                    regime_data
                ),
                metadata={
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "market_regime": current_regime,
                    "lookback_days": config.lookback_period_days,
                    "regime_specific_samples": len(regime_data),
                },
            )

            self.logger.info(
                f"統計的生成完了: SL={optimal_tpsl['sl']:.3f}, "
                f"TP={optimal_tpsl['tp']:.3f}, 信頼度={confidence_score:.2f}"
            )

            return result

        except Exception as e:
            self.logger.error(f"統計的TP/SL生成エラー: {e}", exc_info=True)
            return self._generate_fallback_statistical_result(config)

    def _initialize_sample_data(self) -> List[Dict[str, Any]]:
        """サンプル統計データを初期化"""
        # 実際の実装では、データベースや外部APIから取得
        sample_data = []

        # 様々なTP/SL組み合わせのサンプルパフォーマンス
        sl_values = [0.01, 0.02, 0.03, 0.04, 0.05]
        tp_values = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15]

        for sl in sl_values:
            for tp in tp_values:
                if tp > sl:  # TPはSLより大きい必要がある
                    # ランダムなパフォーマンスデータを生成（実際は過去データ）
                    performance = {
                        "sl": sl,
                        "tp": tp,
                        "sharpe_ratio": np.random.normal(0.5, 0.3),
                        "total_return": np.random.normal(0.1, 0.15),
                        "win_rate": np.random.uniform(0.3, 0.7),
                        "max_drawdown": np.random.uniform(0.05, 0.25),
                        "profit_factor": np.random.uniform(0.8, 2.5),
                        "total_trades": np.random.randint(50, 500),
                        "market_regime": np.random.choice(
                            list(self.market_regimes.keys())
                        ),
                        "symbol": "BTC/USDT",  # サンプル用
                        "timeframe": "1h",
                    }
                    sample_data.append(performance)

        return sample_data

    def _get_historical_performance_data(
        self, symbol: Optional[str], timeframe: Optional[str], config: StatisticalConfig
    ) -> List[Dict[str, Any]]:
        """過去のパフォーマンスデータを取得"""
        try:
            # フィルタリング条件
            filtered_data = self.sample_data.copy()

            # シンボル固有フィルタリング
            if config.symbol_specific and symbol:
                filtered_data = [d for d in filtered_data if d.get("symbol") == symbol]

            # 時間軸固有フィルタリング
            if config.timeframe_specific and timeframe:
                filtered_data = [
                    d for d in filtered_data if d.get("timeframe") == timeframe
                ]

            # 最小サンプル数チェック
            if len(filtered_data) < config.min_sample_size:
                self.logger.warning(
                    f"サンプル数不足 ({len(filtered_data)} < {config.min_sample_size}), "
                    "全データを使用"
                )
                filtered_data = self.sample_data.copy()

            return filtered_data

        except Exception as e:
            self.logger.error(f"過去データ取得エラー: {e}")
            return self.sample_data.copy()

    def _determine_market_regime(
        self, market_conditions: Optional[Dict[str, Any]]
    ) -> str:
        """現在の市場レジームを判定"""
        if not market_conditions:
            return "sideways"  # デフォルト

        try:
            trend = market_conditions.get("trend", "neutral")
            volatility = market_conditions.get("volatility", "medium")

            # 簡単なルールベース判定
            if trend == "up" and volatility in ["low", "medium"]:
                return "bull_market"
            elif trend == "down" and volatility in ["high", "very_high"]:
                return "bear_market"
            elif volatility == "very_high":
                return "high_volatility"
            else:
                return "sideways"

        except Exception as e:
            self.logger.error(f"市場レジーム判定エラー: {e}")
            return "sideways"

    def _filter_by_market_regime(
        self, data: List[Dict[str, Any]], regime: str
    ) -> List[Dict[str, Any]]:
        """市場レジームでデータをフィルタリング"""
        try:
            regime_data = [d for d in data if d.get("market_regime") == regime]

            # レジーム固有データが少ない場合は全データを使用
            if len(regime_data) < 20:
                self.logger.warning(
                    f"レジーム固有データ不足 ({len(regime_data)}), 全データを使用"
                )
                return data

            return regime_data

        except Exception as e:
            self.logger.error(f"レジームフィルタリングエラー: {e}")
            return data

    def _find_optimal_tpsl_combination(
        self, data: List[Dict[str, Any]], config: StatisticalConfig
    ) -> Dict[str, float]:
        """最適なTP/SL組み合わせを見つける"""
        try:
            if not data:
                return {"sl": 0.03, "tp": 0.06}  # デフォルト

            # 目標指標に基づいてソート
            objective = config.optimization_objective.value

            if objective == "max_drawdown":
                # 最大ドローダウンは小さい方が良い
                best_entry = min(data, key=lambda x: x.get(objective, 1.0))
            else:
                # その他の指標は大きい方が良い
                best_entry = max(data, key=lambda x: x.get(objective, 0.0))

            return {"sl": best_entry["sl"], "tp": best_entry["tp"]}

        except Exception as e:
            self.logger.error(f"最適組み合わせ検索エラー: {e}")
            return {"sl": 0.03, "tp": 0.06}

    def _calculate_expected_performance(
        self,
        tpsl: Dict[str, float],
        data: List[Dict[str, Any]],
        config: StatisticalConfig,
    ) -> Dict[str, float]:
        """期待パフォーマンスを計算"""
        try:
            # 同じTP/SL設定のデータを抽出
            matching_data = [
                d
                for d in data
                if abs(d["sl"] - tpsl["sl"]) < 0.005
                and abs(d["tp"] - tpsl["tp"]) < 0.005
            ]

            if not matching_data:
                # 近似データを使用
                matching_data = data[:5]  # 上位5件

            # 平均パフォーマンスを計算
            performance_metrics = [
                "sharpe_ratio",
                "total_return",
                "win_rate",
                "max_drawdown",
                "profit_factor",
            ]

            expected = {}
            for metric in performance_metrics:
                values = [d.get(metric, 0) for d in matching_data]
                expected[metric] = np.mean(values) if values else 0.0

            return expected

        except Exception as e:
            self.logger.error(f"期待パフォーマンス計算エラー: {e}")
            return {
                "sharpe_ratio": 0.5,
                "total_return": 0.1,
                "win_rate": 0.5,
                "max_drawdown": 0.15,
                "profit_factor": 1.2,
            }

    def _calculate_statistical_confidence(
        self,
        data: List[Dict[str, Any]],
        tpsl: Dict[str, float],
        config: StatisticalConfig,
    ) -> float:
        """統計的信頼度を計算"""
        try:
            base_confidence = min(len(data) / config.min_sample_size, 1.0)

            # データ品質による調整
            quality_factors = []

            # サンプル数による調整
            if len(data) >= config.min_sample_size * 2:
                quality_factors.append(1.1)
            elif len(data) < config.min_sample_size:
                quality_factors.append(0.8)

            # パフォーマンス一貫性による調整
            objective_values = [
                d.get(config.optimization_objective.value, 0) for d in data
            ]
            if objective_values:
                cv = (
                    np.std(objective_values) / np.mean(objective_values)
                    if np.mean(objective_values) != 0
                    else 1
                )
                if cv < 0.3:  # 低い変動係数は高い一貫性を示す
                    quality_factors.append(1.1)
                elif cv > 0.7:
                    quality_factors.append(0.9)

            # 最終信頼度
            final_confidence = base_confidence * np.prod(quality_factors)
            return max(0.1, min(1.0, final_confidence))

        except Exception as e:
            self.logger.error(f"信頼度計算エラー: {e}")
            return 0.5

    def _summarize_historical_performance(
        self, data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """過去パフォーマンスの要約を作成"""
        try:
            if not data:
                return {}

            metrics = ["sharpe_ratio", "total_return", "win_rate", "max_drawdown"]
            summary = {}

            for metric in metrics:
                values = [d.get(metric, 0) for d in data]
                if values:
                    summary[f"{metric}_mean"] = np.mean(values)
                    summary[f"{metric}_std"] = np.std(values)
                    summary[f"{metric}_min"] = np.min(values)
                    summary[f"{metric}_max"] = np.max(values)

            summary["total_samples"] = len(data)
            return summary

        except Exception as e:
            self.logger.error(f"パフォーマンス要約エラー: {e}")
            return {"total_samples": len(data)}

    def _generate_fallback_statistical_result(
        self, config: StatisticalConfig
    ) -> StatisticalResult:
        """フォールバック統計結果を生成"""
        return StatisticalResult(
            stop_loss_pct=0.03,
            take_profit_pct=0.06,
            expected_performance={
                "sharpe_ratio": 0.5,
                "total_return": 0.1,
                "win_rate": 0.5,
                "max_drawdown": 0.15,
                "profit_factor": 1.2,
            },
            confidence_score=0.3,
            sample_size=0,
            optimization_objective=config.optimization_objective.value,
            historical_performance={},
            metadata={"fallback": True},
        )
