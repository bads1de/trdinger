"""
フィットネス計算モジュール

バックテスト結果からパフォーマンスメトリクスを抽出し、
適応度（Fitness）値を計算します。
"""

import hashlib
import json
import logging
import math
from typing import Any, Dict, Mapping, Optional, Tuple, cast

import numpy as np

from app.types import SerializableValue
from app.services.auto_strategy.config.ga import GAConfig
from app.services.auto_strategy.config import objective_registry

from ..evaluation.evaluation_metrics import (
    calculate_trade_frequency_penalty,
    calculate_ulcer_index,
)

logger = logging.getLogger(__name__)


class FitnessCalculator:
    """
    フィットネス計算を担当するクラス

    IndividualEvaluator から委譲を受け、パフォーマンスメトリクスの抽出、
    単一目的・多目的フィットネスの計算、ロング・ショートバランス評価を行います。
    """

    def __init__(self) -> None:
        # メトリクスキャッシュ（同一 backtest_result の再計算を避ける）
        self._metrics_cache: Dict[str, Dict[str, float]] = {}
        self._cache_enabled = True
        self._recent_metrics_result: Optional[Dict[str, SerializableValue]] = None
        self._recent_metrics_signature: object = None
        self._recent_metrics_value: Optional[Dict[str, float]] = None

    def clear_cache(self) -> None:
        """メトリクスキャッシュをクリアする。"""
        self._metrics_cache.clear()
        self._recent_metrics_result = None
        self._recent_metrics_signature = None
        self._recent_metrics_value = None

    @staticmethod
    def _json_default(value: object) -> SerializableValue:
        """JSONキー生成時の非標準値を文字列または素値に落とす。

        NumPyのスカラー型など、JSONに変換できない値を
        Pythonの標準型に変換します。

        Args:
            value: 変換対象の値。

        Returns:
            SerializableValue: JSONに変換可能な値。NumPyスカラーの場合は.item()で変換、
                それ以外はstr()変換。
        """
        if isinstance(value, np.generic):
            return value.item()
        return str(value)

    @staticmethod
    def _normalize_signature_value(value: object) -> object:
        """直近result判定用に値を軽量に正規化する。

        NumPyスカラー、floatの特殊値（NaN、Inf）、
        その他の非標準値を比較可能な形式に変換します。

        Args:
            value: 正規化する値。

        Returns:
            Any: 正規化された値。NaN/Infの場合は特殊タプル、
                それ以外はそのまま、またはrepr()変換。
        """
        if isinstance(value, np.generic):
            value = value.item()

        if isinstance(value, float):
            if math.isnan(value):
                return ("float", "nan")
            if math.isinf(value):
                return ("float", "inf" if value > 0 else "-inf")
        if value is None or isinstance(value, (str, int, float, bool, bytes)):
            return value

        return repr(value)

    def _sample_sequence_signature(self, value: object) -> tuple[object, ...]:
        """大きいシーケンスの軽量シグネチャを作る。

        長大なリストやタプルの全要素をハッシュするのではなく、
        代表点（先頭、中央、末尾）を抽出することで
        高速な同一性判定を実現します。

        Args:
            value: シーケンス（listまたはtuple）。

        Returns:
            tuple[object, ...]: 代表点の値を正規化したタプル。
                空シーケンスの場合は空タプル。
        """
        if not isinstance(value, (list, tuple)) or not value:
            return ()

        candidate_indices = [0, len(value) // 2, len(value) - 1]
        indices: list[int] = []
        for idx in candidate_indices:
            if idx not in indices:
                indices.append(idx)

        return tuple(self._normalize_signature_value(value[idx]) for idx in indices)

    def _build_recent_result_signature(self, backtest_result: Dict[str, SerializableValue]) -> object:
        """同一resultの直近再利用判定に使う軽量シグネチャを作る。

        バックテスト結果の主要メトリクス、資産曲線、取引履歴の
        代表点を抽出して、高速な同一性比較が可能な形式を生成します。

        Args:
            backtest_result: バックテスト結果の辞書。
                performance_metrics、equity_curve、trade_historyなどを含む。

        Returns:
            Any: シグネチャタプル。バックテスト結果の内容を
                軽量に比較可能な形式で表現。
        """
        performance_metrics = backtest_result.get("performance_metrics") or {}
        if isinstance(performance_metrics, dict):
            perf_metrics_dict: Mapping[str, object] = performance_metrics
        else:
            perf_metrics_dict = {}
        equity_curve = backtest_result.get("equity_curve") or []
        trade_history = backtest_result.get("trade_history") or []

        metric_keys = (
            "total_return",
            "sharpe_ratio",
            "max_drawdown",
            "win_rate",
            "profit_factor",
            "sortino_ratio",
            "calmar_ratio",
            "total_trades",
        )

        return (
            id(backtest_result),
            tuple(
                (
                    key,
                    self._normalize_signature_value(perf_metrics_dict.get(key)),
                )
                for key in metric_keys
            ),
            len(equity_curve) if isinstance(equity_curve, (list, tuple)) else None,
            self._sample_sequence_signature(equity_curve),
            len(trade_history) if isinstance(trade_history, (list, tuple)) else None,
            self._sample_sequence_signature(trade_history),
            self._normalize_signature_value(backtest_result.get("start_date")),
            self._normalize_signature_value(backtest_result.get("end_date")),
        )

    def _generate_cache_key(self, backtest_result: Dict[str, SerializableValue]) -> str:
        """バックテスト結果の内容から安定したキャッシュキーを生成する。"""
        payload = {
            "performance_metrics": backtest_result.get("performance_metrics") or {},
            "equity_curve": backtest_result.get("equity_curve") or [],
            "trade_history": backtest_result.get("trade_history") or [],
            "start_date": backtest_result.get("start_date"),
            "end_date": backtest_result.get("end_date"),
        }

        try:
            cache_text = json.dumps(
                payload,
                sort_keys=True,
                ensure_ascii=False,
                default=self._json_default,
            )
        except (TypeError, ValueError):
            cache_text = repr(payload)

        return hashlib.sha256(cache_text.encode("utf-8")).hexdigest()

    def get_penalty_values(self, config: "GAConfig") -> Tuple[float, ...]:
        """一貫したペナルティ値のタプルを返す。

        評価エラー時や制約違反時に使用する。目的関数の方向性に応じて
        適切なペナルティ値を設定する。
        """
        penalty_values = []
        for obj in config.objectives:
            if objective_registry.is_minimize_objective(obj):
                penalty_values.append(float("inf"))
            else:
                penalty_values.append(-float("inf"))
        return tuple(penalty_values)

    def extract_performance_metrics(
        self, backtest_result: Dict[str, SerializableValue]
    ) -> Dict[str, float]:
        """
        バックテスト結果からパフォーマンスメトリクスを抽出

        Args:
            backtest_result: バックテスト結果

        Returns:
            抽出されたパフォーマンスメトリクス
        """
        recent_signature = self._build_recent_result_signature(backtest_result)
        if (
            self._cache_enabled
            and self._recent_metrics_result is backtest_result
            and self._recent_metrics_signature == recent_signature
            and self._recent_metrics_value is not None
        ):
            return self._recent_metrics_value

        cache_key = self._generate_cache_key(backtest_result)
        if self._cache_enabled and cache_key in self._metrics_cache:
            cached_metrics = self._metrics_cache[cache_key]
            self._recent_metrics_result = backtest_result
            self._recent_metrics_signature = recent_signature
            self._recent_metrics_value = cached_metrics
            return cached_metrics

        performance_metrics = backtest_result.get("performance_metrics") or {}
        if not isinstance(performance_metrics, dict):
            performance_metrics = {}
        perf_metrics: Mapping[str, object] = performance_metrics

        total_return = perf_metrics.get("total_return", 0.0)
        sharpe_ratio = perf_metrics.get("sharpe_ratio", 0.0)
        max_drawdown = perf_metrics.get("max_drawdown", 1.0)
        win_rate = perf_metrics.get("win_rate", 0.0)
        profit_factor = perf_metrics.get("profit_factor", 0.0)
        sortino_ratio = perf_metrics.get("sortino_ratio", 0.0)
        calmar_ratio = perf_metrics.get("calmar_ratio", 0.0)
        total_trades = perf_metrics.get("total_trades", 0)

        def _sanitize_float(value: object, default: float) -> float:
            if value is None or not isinstance(value, (int, float)):
                return default
            value_float = float(value)
            if isinstance(value_float, float) and not math.isfinite(value_float):
                return default
            return value_float

        total_return = _sanitize_float(total_return, 0.0)
        sharpe_ratio = _sanitize_float(sharpe_ratio, 0.0)
        max_drawdown = _sanitize_float(max_drawdown, 1.0)
        win_rate = _sanitize_float(win_rate, 0.0)
        profit_factor = _sanitize_float(profit_factor, 0.0)
        sortino_ratio = _sanitize_float(sortino_ratio, 0.0)
        calmar_ratio = _sanitize_float(calmar_ratio, 0.0)
        total_trades = int(_sanitize_float(total_trades, 0.0))

        if max_drawdown < 0:
            max_drawdown = 0.0

        equity_curve = backtest_result.get("equity_curve")
        equity_curve_list = cast(list[dict[str, Any]], equity_curve) if isinstance(equity_curve, list) else []
        ulcer_index = calculate_ulcer_index(equity_curve_list) if equity_curve_list else 0.0

        trade_history = backtest_result.get("trade_history")
        trade_history_list = cast(list[dict[str, Any]], trade_history) if isinstance(trade_history, list) else []
        trade_penalty = (
            calculate_trade_frequency_penalty(
                total_trades=total_trades,
                start_date=backtest_result.get("start_date"),
                end_date=backtest_result.get("end_date"),
                trade_history=trade_history_list,
            )
            if trade_history_list
            else 0.0
        )

        metrics = {
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "sortino_ratio": sortino_ratio,
            "calmar_ratio": calmar_ratio,
            "total_trades": total_trades,
            "ulcer_index": ulcer_index,
            "trade_frequency_penalty": trade_penalty,
        }

        if self._cache_enabled:
            self._metrics_cache[cache_key] = metrics
            self._recent_metrics_result = backtest_result
            self._recent_metrics_signature = recent_signature
            self._recent_metrics_value = metrics

        return metrics

    def meets_constraints(
        self, metrics: Mapping[str, float], config: "GAConfig"
    ) -> bool:
        """単一目的評価で使う制約判定をまとめて返す。"""
        try:
            constraints = getattr(config, "fitness_constraints", {}) or {}

            total_trades = int(metrics.get("total_trades", 0))
            if total_trades <= 0:
                return False

            min_trades_req = int(constraints.get("min_trades", 0) or 0)
            if total_trades < min_trades_req:
                return False

            max_drawdown_limit = constraints.get("max_drawdown_limit", None)
            max_drawdown = metrics.get("max_drawdown", 0.0)
            if isinstance(max_drawdown_limit, (float, int)) and max_drawdown > float(
                max_drawdown_limit
            ):
                return False

            total_return = metrics.get("total_return", 0.0)
            if total_return < 0.0:
                return False

            sharpe_ratio = metrics.get("sharpe_ratio", 0.0)
            min_sharpe_ratio = float(constraints.get("min_sharpe_ratio", 0.0) or 0.0)
            if sharpe_ratio < min_sharpe_ratio:
                return False

            return True
        except (KeyError, TypeError, ValueError):
            return False

    def calculate_fitness(
        self, backtest_result: Dict[str, SerializableValue], config: GAConfig, **kwargs
    ) -> float:
        """
        フィットネス計算（ロング・ショートバランス評価を含む）

        Args:
            backtest_result: バックテスト結果
            config: GA設定

        Returns:
            フィットネス値
        """
        try:
            metrics = self.extract_performance_metrics(backtest_result)

            total_return = metrics["total_return"]
            sharpe_ratio = metrics["sharpe_ratio"]
            max_drawdown = metrics["max_drawdown"]
            win_rate = metrics["win_rate"]
            total_trades = int(metrics["total_trades"])
            ulcer_index = metrics.get("ulcer_index", 0.0)
            trade_penalty = metrics.get("trade_frequency_penalty", 0.0)

            if total_trades == 0:
                # GAの探索では zero-trade はよくある境界条件なので、警告ではなく debug に落とす。
                logger.debug("取引回数が0のため、低いフィットネス値を設定")
                return config.zero_trades_penalty

            if not self.meets_constraints(metrics, config):
                return config.constraint_violation_penalty

            balance_score = self.calculate_long_short_balance(backtest_result)

            fitness_weights = config.fitness_weights.copy()

            fitness = (
                float(fitness_weights.get("total_return", 0.3)) * total_return
                + float(fitness_weights.get("sharpe_ratio", 0.4)) * sharpe_ratio
                + float(fitness_weights.get("max_drawdown", 0.2)) * (1 - max_drawdown)
                + float(fitness_weights.get("win_rate", 0.1)) * win_rate
                + float(fitness_weights.get("balance_score", 0.1)) * balance_score
            )

            ulcer_scale = 1.0
            trade_scale = 1.0
            if getattr(config, "dynamic_objective_reweighting", False):
                dynamic_scalars = getattr(config, "objective_dynamic_scalars", {})
                ulcer_scale = float(dynamic_scalars.get("ulcer_index", 1.0))
                trade_scale = float(dynamic_scalars.get("trade_frequency_penalty", 1.0))

            fitness -= (
                float(fitness_weights.get("ulcer_index_penalty", 0.0))
                * ulcer_scale
                * ulcer_index
            )
            fitness -= (
                float(fitness_weights.get("trade_frequency_penalty", 0.0))
                * trade_scale
                * trade_penalty
            )

            return max(0.0, fitness)

        except (KeyError, TypeError, ValueError) as e:
            logger.error(f"フィットネス計算エラー: {e}", exc_info=True)
            return config.constraint_violation_penalty

    def _calculate_balance_score_fast(self, backtest_result: Dict[str, SerializableValue]) -> float:
        """ロング・ショートバランススコア計算（最適化版）。"""
        trade_history = backtest_result.get("trade_history")
        if not isinstance(trade_history, list):
            return 0.5

        try:
            n_trades = len(trade_history)
            if n_trades < 50:
                return self._calculate_balance_score_inline(cast(list[dict[str, Any]], trade_history))
            return self._calculate_balance_score_numpy(cast(list[dict[str, Any]], trade_history))
        except Exception as e:
            logger.debug(f"バランススコア計算エラー: {e}")
            return 0.5

    def _calculate_balance_score_inline(self, trade_history: list[dict[str, Any]]) -> float:
        """小規模データ向けのインライン計算。"""
        long_count = 0
        short_count = 0
        long_pnl = 0.0
        short_pnl = 0.0

        for trade in trade_history:
            size = trade.get("size", 0.0)
            pnl = trade.get("pnl", 0.0)
            if size > 0:
                long_count += 1
                long_pnl += pnl
            elif size < 0:
                short_count += 1
                short_pnl += pnl

        total_count = long_count + short_count
        if total_count == 0:
            return 0.5

        long_ratio = long_count / total_count
        short_ratio = short_count / total_count
        trade_balance = 1.0 - abs(long_ratio - short_ratio)

        total_pnl = long_pnl + short_pnl
        if total_pnl > 0:
            if long_pnl > 0 and short_pnl > 0:
                profit_balance = 1.0
            elif long_pnl > 0 or short_pnl > 0:
                profit_balance = 0.7
            else:
                profit_balance = 0.5
        elif long_pnl < 0 and short_pnl < 0:
            profit_balance = 0.1
        else:
            profit_balance = 0.3

        balance_score = 0.6 * trade_balance + 0.4 * profit_balance
        return float(max(0.0, min(1.0, balance_score)))

    def _calculate_balance_score_numpy(self, trade_history: list[dict[str, Any]]) -> float:
        """大規模データ向けの NumPy 版計算。"""
        trades_array = np.array(
            [
                (trade.get("size", 0.0), trade.get("pnl", 0.0))
                for trade in trade_history
            ],
            dtype=np.float64,
        )

        if len(trades_array) == 0:
            return 0.5

        sizes = trades_array[:, 0]
        pnls = trades_array[:, 1]

        long_mask = sizes > 0
        short_mask = sizes < 0

        long_count = np.sum(long_mask)
        short_count = np.sum(short_mask)
        total_count = long_count + short_count

        if total_count == 0:
            return 0.5

        long_ratio = long_count / total_count
        short_ratio = short_count / total_count
        trade_balance = 1.0 - abs(long_ratio - short_ratio)

        long_pnl = np.sum(pnls[long_mask]) if long_count > 0 else 0.0
        short_pnl = np.sum(pnls[short_mask]) if short_count > 0 else 0.0
        total_pnl = long_pnl + short_pnl

        if total_pnl > 0:
            if long_pnl > 0 and short_pnl > 0:
                profit_balance = 1.0
            elif long_pnl > 0 or short_pnl > 0:
                profit_balance = 0.7
            else:
                profit_balance = 0.5
        elif long_pnl < 0 and short_pnl < 0:
            profit_balance = 0.1
        else:
            profit_balance = 0.3

        balance_score = float(0.6 * trade_balance + 0.4 * profit_balance)
        return max(0.0, min(1.0, balance_score))

    def calculate_long_short_balance(self, backtest_result: Dict[str, SerializableValue]) -> float:
        """
        ロング・ショートバランススコアを計算

        Args:
            backtest_result: バックテスト結果

        Returns:
            バランススコア（0.0-1.0）
        """
        try:
            return self._calculate_balance_score_fast(backtest_result)

        except (KeyError, TypeError, ValueError) as e:
            logger.error(f"ロング・ショートバランス計算エラー: {e}", exc_info=True)
            return 0.5

    def calculate_multi_objective_fitness(
        self, backtest_result: Dict[str, SerializableValue], config: GAConfig, **kwargs
    ) -> Tuple[float, ...]:
        """
        複数の目的関数（利益、リスク、安定性等）に基づいて個体の多次元適応度を計算します。

        このメソッドは、NSGA-II 等の多目的最適化アルゴリズムで使用される適応度タプルを生成します：
        1. **メトリクスの抽出**: バックテスト結果から収益率、シャープレシオ、最大ドローダウン等の統計データを取得。
        2. **制約チェック**: 取引回数が設定された最小回数（`min_trades`）を下回る場合、即座にペナルティ値を返却。
        3. **目的関数の構成**: `config.objectives` に指定された順序で各指標（weighted_score, total_return, sharpe_ratio 等）を抽出。
        4. **動的スケーリング**: `objective_dynamic_scalars` が存在する場合、集団の統計に基づいて各指標のスケールを調整。
        5. **正規化**: 最小化すべき指標（ドローダウン等）は、DEAPの仕様に合わせて適切に処理されます。

        Args:
            backtest_result (Dict[str, SerializableValue]): シミュレーターから出力された全統計データ。
            config (GAConfig): 最適化対象とする目的関数リスト、制約、重みを含む統合設定。
            **kwargs: 追加の評価コンテキスト。

        Returns:
            Tuple[float, ...]: 各目的関数に対応する評価値のタプル。
                DEAPの `Fitness` クラスにより、この値を用いてパレート優位性が判定されます。
        """
        try:
            metrics = self.extract_performance_metrics(backtest_result)
            total_trades = int(metrics["total_trades"])

            min_trades_req = int(config.fitness_constraints.get("min_trades", 0))
            if total_trades < min_trades_req:
                return self.get_penalty_values(config)

            fitness_values = []

            for objective in config.objectives:
                if objective == "weighted_score":
                    value = self.calculate_fitness(backtest_result, config, **kwargs)
                elif objective == "total_return":
                    value = metrics["total_return"]
                elif objective == "sharpe_ratio":
                    value = metrics["sharpe_ratio"]
                elif objective == "max_drawdown":
                    value = metrics["max_drawdown"]
                elif objective == "win_rate":
                    value = metrics["win_rate"]
                elif objective == "profit_factor":
                    value = metrics["profit_factor"]
                elif objective == "sortino_ratio":
                    value = metrics["sortino_ratio"]
                elif objective == "calmar_ratio":
                    value = metrics["calmar_ratio"]
                elif objective == "balance_score":
                    value = self.calculate_long_short_balance(backtest_result)
                elif objective == "ulcer_index":
                    value = metrics.get("ulcer_index", 0.0)
                elif objective == "trade_frequency_penalty":
                    value = metrics.get("trade_frequency_penalty", 0.0)
                else:
                    logger.warning(f"未知の目的: {objective}")
                    penalty = self.get_penalty_values(config)
                    penalty_idx = list(config.objectives).index(objective)
                    value = penalty[penalty_idx] if penalty_idx < len(penalty) else 0.0

                dynamic_scalars = getattr(config, "objective_dynamic_scalars", {})
                scale = float(dynamic_scalars.get(objective, 1.0))
                fitness_values.append(float(value) * scale)

            return tuple(fitness_values)

        except (KeyError, TypeError, ValueError) as e:
            logger.error(f"多目的フィットネス計算エラー: {e}", exc_info=True)
            return self.get_penalty_values(config)
