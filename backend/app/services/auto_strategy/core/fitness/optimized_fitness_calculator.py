"""
最適化されたフィットネス計算モジュール

パフォーマンス最適化版のフィットネス計算を提供します。
"""

import logging
import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

from app.services.auto_strategy.config.ga import GAConfig
from ..evaluation.evaluation_metrics import calculate_trade_frequency_penalty, calculate_ulcer_index

logger = logging.getLogger(__name__)


class OptimizedFitnessCalculator:
    """
    最適化されたフィットネス計算クラス

    主な最適化ポイント:
    1. メトリクスの事前計算とキャッシュ
    2. ベクトル化されたロング・ショートバランス計算
    3. インライン化による関数呼び出しオーバーヘッド削減
    4. 早期リターンによる不要な計算の回避
    """

    # 最小化目的（ペナルティ値は最大化）と最大化目的（ペナルティ値は最小化）の分類
    _MINIMIZE_OBJECTIVES = frozenset({"max_drawdown", "ulcer_index", "trade_frequency_penalty"})

    # デフォルトフィットネス重み（辞書アクセスを避けるためタプルで保持）
    _DEFAULT_WEIGHTS = (0.3, 0.4, 0.2, 0.1, 0.1)  # total_return, sharpe, max_dd, win_rate, balance

    def __init__(self) -> None:
        # 計算済みメトリクスのキャッシュ
        self._metrics_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_enabled = True

    def clear_cache(self):
        """キャッシュをクリア"""
        self._metrics_cache.clear()

    def get_penalty_values(self, config: "GAConfig") -> Tuple[float, ...]:
        """一貫したペナルティ値のタプルを返す。"""
        penalty_values = []
        for obj in config.objectives:
            if obj in self._MINIMIZE_OBJECTIVES:
                penalty_values.append(float("inf"))
            else:
                penalty_values.append(-float("inf"))
        return tuple(penalty_values)

    def extract_performance_metrics(
        self, backtest_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        バックテスト結果からパフォーマンスメトリクスを抽出（最適化版）

        最適化:
        - 直接的な辞書アクセス（getattr削減）
        - 早期リターンによる不要な計算の回避
        - NumPy配列の活用
        """
        # キャッシュチェック
        cache_key: str = str(id(backtest_result))
        if self._cache_enabled and cache_key in self._metrics_cache:
            return self._metrics_cache[cache_key]

        # performance_metricsを直接取得（毎回get()を呼ばない）
        pm = backtest_result.get("performance_metrics")
        if pm is None:
            pm = {}

        # メトリクスを直接取得（getattr削減）
        total_return = pm.get("total_return", 0.0)
        sharpe_ratio = pm.get("sharpe_ratio", 0.0)
        max_drawdown = pm.get("max_drawdown", 1.0)
        win_rate = pm.get("win_rate", 0.0)
        profit_factor = pm.get("profit_factor", 0.0)
        sortino_ratio = pm.get("sortino_ratio", 0.0)
        calmar_ratio = pm.get("calmar_ratio", 0.0)
        total_trades = pm.get("total_trades", 0)

        # 値のバリデーションを最適化（個別チェック削減）
        if total_return is None or not isinstance(total_return, (int, float)):
            total_return = 0.0
        elif isinstance(total_return, float) and not math.isfinite(total_return):
            total_return = 0.0

        if sharpe_ratio is None or not isinstance(sharpe_ratio, (int, float)):
            sharpe_ratio = 0.0
        elif isinstance(sharpe_ratio, float) and not math.isfinite(sharpe_ratio):
            sharpe_ratio = 0.0

        if max_drawdown is None or not isinstance(max_drawdown, (int, float)):
            max_drawdown = 1.0
        elif isinstance(max_drawdown, float) and not math.isfinite(max_drawdown):
            max_drawdown = 1.0
        elif max_drawdown < 0:
            max_drawdown = 0.0

        if win_rate is None or not isinstance(win_rate, (int, float)):
            win_rate = 0.0
        elif isinstance(win_rate, float) and not math.isfinite(win_rate):
            win_rate = 0.0

        if total_trades is None or not isinstance(total_trades, (int, float)):
            total_trades = 0
        total_trades = int(total_trades)  # int型に変換

        # Ulcer Index計算（ equity_curveが空でない場合のみ）
        equity_curve = backtest_result.get("equity_curve")
        if equity_curve:
            ulcer_index = calculate_ulcer_index(equity_curve)
        else:
            ulcer_index = 0.0

        # 取引頻度ペナルティ計算
        trade_history = backtest_result.get("trade_history")
        if trade_history:
            trade_penalty = calculate_trade_frequency_penalty(
                total_trades=int(total_trades),
                start_date=backtest_result.get("start_date"),
                end_date=backtest_result.get("end_date"),
                trade_history=trade_history,
            )
        else:
            trade_penalty = 0.0

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

        # キャッシュに保存
        if self._cache_enabled:
            self._metrics_cache[str(cache_key)] = metrics

        return metrics

    def calculate_fitness(
        self, backtest_result: Dict[str, Any], config: GAConfig, **kwargs
    ) -> float:
        """
        フィットネス計算（最適化版）

        最適化:
        - 早期リターンによる不要な計算の回避
        - 辞書アクセスの最小化
        - インライン化による関数呼び出し削減
        """
        try:
            # メトリクス抽出
            metrics = self.extract_performance_metrics(backtest_result)

            total_trades = metrics["total_trades"]

            # 早期リターン: 取引回数チェック
            if total_trades == 0:
                return config.zero_trades_penalty

            min_trades_req = config.fitness_constraints.get("min_trades", 0)
            if total_trades < min_trades_req:
                return config.constraint_violation_penalty

            # ドローダウン制約チェック
            max_drawdown = metrics["max_drawdown"]
            max_dd_limit = config.fitness_constraints.get("max_drawdown_limit")
            if max_dd_limit is not None and max_drawdown > max_dd_limit:
                return config.constraint_violation_penalty

            # リターン制約チェック
            total_return = metrics["total_return"]
            if total_return < 0:
                return config.constraint_violation_penalty

            # シャープレシオ制約チェック
            sharpe_ratio = metrics["sharpe_ratio"]
            min_sharpe = config.fitness_constraints.get("min_sharpe_ratio", 0)
            if sharpe_ratio < min_sharpe:
                return config.constraint_violation_penalty

            # バランススコア計算（インライン化）
            balance_score = self._calculate_balance_score_fast(backtest_result)

            # フィットネス重み取得（コピーを避ける）
            fw = config.fitness_weights

            # フィットネス計算（インライン化）
            fitness = (
                fw.get("total_return", 0.3) * total_return
                + fw.get("sharpe_ratio", 0.4) * sharpe_ratio
                + fw.get("max_drawdown", 0.2) * (1.0 - max_drawdown)
                + fw.get("win_rate", 0.1) * metrics["win_rate"]
                + fw.get("balance_score", 0.1) * balance_score
            )

            # ペナルティ適用
            ulcer_scale = 1.0
            trade_scale = 1.0
            if config.dynamic_objective_reweighting:
                dynamic_scalars = config.objective_dynamic_scalars
                ulcer_scale = dynamic_scalars.get("ulcer_index", 1.0)
                trade_scale = dynamic_scalars.get("trade_frequency_penalty", 1.0)

            ulcer_penalty_weight = fw.get("ulcer_index_penalty", 0.0)
            if ulcer_penalty_weight > 0:
                fitness -= ulcer_penalty_weight * ulcer_scale * metrics["ulcer_index"]

            trade_penalty_weight = fw.get("trade_frequency_penalty", 0.0)
            if trade_penalty_weight > 0:
                fitness -= trade_penalty_weight * trade_scale * metrics["trade_frequency_penalty"]

            return max(0.0, fitness)

        except (KeyError, TypeError, ValueError) as e:
            logger.error(f"フィットネス計算エラー: {e}", exc_info=True)
            return config.constraint_violation_penalty

    def _calculate_balance_score_fast(self, backtest_result: Dict[str, Any]) -> float:
        """
        ロング・ショートバランススコア計算（最適化版）

        最適化:
        - 小規模データ: インライン化による高速化
        - 大規模データ: NumPy配列の活用
        - 早期リターン
        """
        trade_history = backtest_result.get("trade_history")
        if not trade_history:
            return 0.5

        try:
            n_trades = len(trade_history)

            # 小規模データ（50件未満）の場合はインライン計算の方が高速
            if n_trades < 50:
                return self._calculate_balance_score_inline(trade_history)

            # 大規模データの場合はNumPy配列を使用
            return self._calculate_balance_score_numpy(trade_history)

        except Exception as e:
            logger.debug(f"バランススコア計算エラー: {e}")
            return 0.5

    def _calculate_balance_score_inline(self, trade_history: list) -> float:
        """インライン計算版（小規模データ向け）"""
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

    def _calculate_balance_score_numpy(self, trade_history: list) -> float:
        """NumPy版（大規模データ向け）"""
        # NumPy配列に変換
        trades_array = np.array([
            (trade.get("size", 0.0), trade.get("pnl", 0.0))
            for trade in trade_history
        ], dtype=np.float64)

        if len(trades_array) == 0:
            return 0.5

        sizes = trades_array[:, 0]
        pnls = trades_array[:, 1]

        # ロング・ショートの分類
        long_mask = sizes > 0
        short_mask = sizes < 0

        long_count = np.sum(long_mask)
        short_count = np.sum(short_mask)
        total_count = long_count + short_count

        if total_count == 0:
            return 0.5

        # 取引バランス計算
        long_ratio = long_count / total_count
        short_ratio = short_count / total_count
        trade_balance = 1.0 - abs(long_ratio - short_ratio)

        # 損益バランス計算
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

    def calculate_multi_objective_fitness(
        self, backtest_result: Dict[str, Any], config: GAConfig, **kwargs
    ) -> Tuple[float, ...]:
        """
        多目的適応度の計算（最適化版）

        最適化:
        - メトリクスの再利用
        - 目的関数の事前マッピング
        - 早期リターン
        """
        try:
            metrics = self.extract_performance_metrics(backtest_result)
            total_trades = metrics["total_trades"]

            min_trades_req = config.fitness_constraints.get("min_trades", 0)
            if total_trades < min_trades_req:
                return self.get_penalty_values(config)

            # 目的関数の値を事前計算（マッピングテーブル使用）
            objective_values = {
                "total_return": metrics["total_return"],
                "sharpe_ratio": metrics["sharpe_ratio"],
                "max_drawdown": metrics["max_drawdown"],
                "win_rate": metrics["win_rate"],
                "profit_factor": metrics["profit_factor"],
                "sortino_ratio": metrics["sortino_ratio"],
                "calmar_ratio": metrics["calmar_ratio"],
                "ulcer_index": metrics["ulcer_index"],
                "trade_frequency_penalty": metrics["trade_frequency_penalty"],
            }

            dynamic_scalars = config.objective_dynamic_scalars

            fitness_values = []
            for objective in config.objectives:
                if objective == "weighted_score":
                    value = self.calculate_fitness(backtest_result, config, **kwargs)
                elif objective == "balance_score":
                    value = self._calculate_balance_score_fast(backtest_result)
                else:
                    value = objective_values.get(objective, 0.0)

                scale = dynamic_scalars.get(objective, 1.0)
                fitness_values.append(float(value) * scale)

            return tuple(fitness_values)

        except (KeyError, TypeError, ValueError) as e:
            logger.error(f"多目的フィットネス計算エラー: {e}", exc_info=True)
            return self.get_penalty_values(config)

    def calculate_long_short_balance(self, backtest_result: Dict[str, Any]) -> float:
        """
        ロング・ショートバランススコア計算（公開メソッド）

        Args:
            backtest_result: バックテスト結果辞書

        Returns:
            バランススコア（0.0～1.0）
        """
        return self._calculate_balance_score_fast(backtest_result)
