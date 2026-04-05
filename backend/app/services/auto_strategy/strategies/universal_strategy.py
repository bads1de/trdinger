"""
汎用自動生成戦略クラス

GAから生成されたStrategyGeneを受け取り、その定義に基づいて動的に振る舞う
backtesting.py互換の戦略クラスです。
Pickle化可能にするため、filesのトップレベルで定義されています。
"""

import logging
from math import ceil
from typing import Any, List, Optional, Tuple, Union, cast

import pandas as pd
from backtesting import Strategy

from ..config.ml_filter_settings import resolve_ml_gate_settings
from ..core.evaluation.condition_evaluator import ConditionEvaluator
from ..genes import (
    Condition,
    ConditionGroup,
    IndicatorGene,
    TPSLGene,
    TPSLMethod,
)
from ..genes.conditions import StateTracker
from ..genes.entry import EntryGene
from ..positions.entry_executor import EntryExecutor
from ..positions.lower_tf_simulator import LowerTimeframeSimulator
from ..positions.position_sizing_service import PositionSizingService
from ..services.indicator_service import IndicatorCalculator
from ..tpsl.tpsl_service import TPSLService
from .entry_decision_engine import EntryDecisionEngine
from .ml_filter import MLFilter
from .order_manager import OrderManager
from .position_exit_engine import PositionExitEngine
from .position_manager import PositionManager
from .runtime_state import StrategyRuntimeState
from .stateful_conditions import StatefulConditionsEvaluator

logger = logging.getLogger(__name__)


class StrategyEarlyTermination(RuntimeError):
    """戦略が早期打ち切り条件に達したことを示す例外。"""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class UniversalStrategy(Strategy):
    """
    GA生成汎用戦略クラス

    StrategyFactoryで動的にクラスを生成する代わりに、
    パラメータとしてStrategyGeneを受け取り、その振る舞いを動的に変更します。
    これにより、multiprocessingでのPickle化が可能になります。
    """

    # backtesting.pyの要件: パラメータはクラス変数として定義する必要がある
    # ここではデフォルト値をNoneとし、実行時にparams辞書で上書きされることを期待する
    strategy_gene = None
    minute_data = None
    timeframe = "1h"
    evaluation_start = None
    ml_predictor = None  # MLフィルター用予測器
    volatility_gate_enabled = False
    volatility_model_path = None
    ml_filter_threshold = 0.5  # 旧互換の非推奨パラメータ。現行 gate 判定では未使用。
    enable_early_termination = False
    early_termination_max_drawdown = None
    early_termination_min_trades = None
    early_termination_min_trade_check_progress = 0.5
    early_termination_trade_pace_tolerance = 0.5
    early_termination_min_expectancy = None
    early_termination_expectancy_min_trades = 5
    early_termination_expectancy_progress = 0.6

    @property
    def _sl_price(self) -> float | None:
        """ストップロス価格を取得する。"""
        return self.runtime_state.sl_price

    @_sl_price.setter
    def _sl_price(self, value: float | None) -> None:
        self.runtime_state.sl_price = value

    @property
    def _tp_price(self) -> float | None:
        """テイクプロフィット価格を取得する。"""
        return self.runtime_state.tp_price

    @_tp_price.setter
    def _tp_price(self, value: float | None) -> None:
        self.runtime_state.tp_price = value

    @property
    def _entry_price(self) -> float | None:
        """エントリー価格を取得する。"""
        return self.runtime_state.entry_price

    @_entry_price.setter
    def _entry_price(self, value: float | None) -> None:
        self.runtime_state.entry_price = value

    @property
    def _position_direction(self) -> float:
        """ポジション方向を取得する。"""
        return self.runtime_state.position_direction

    @_position_direction.setter
    def _position_direction(self, value: float) -> None:
        self.runtime_state.position_direction = value

    @property
    def _tp_reached(self) -> bool:
        """TP到達フラグを取得する。"""
        return self.runtime_state.tp_reached

    @_tp_reached.setter
    def _tp_reached(self, value: bool) -> None:
        self.runtime_state.tp_reached = value

    @property
    def _trailing_tp_sl(self) -> float | None:
        """トレーリングTP/SL価格を取得する。"""
        return self.runtime_state.trailing_tp_sl

    @_trailing_tp_sl.setter
    def _trailing_tp_sl(self, value: float | None) -> None:
        self.runtime_state.trailing_tp_sl = value

    def __init__(self, broker, data, params):
        """
        初期化

        Args:
            broker: Brokerインスタンス
            data: Dataインスタンス
            params: パラメータ辞書（'strategy_gene'を含む必要がある）
        """
        self.condition_evaluator = ConditionEvaluator()
        self.tpsl_service = TPSLService()
        self.position_sizing_service = PositionSizingService()
        self.entry_executor = EntryExecutor()  # エントリー注文実行サービス
        self.lower_tf_simulator = LowerTimeframeSimulator()  # 1分足シミュレーター
        self.state_tracker = StateTracker()  # ステートフル条件用
        self.runtime_state = StrategyRuntimeState()
        self._current_bar_index = 0  # バーインデックストラッカー

        # 注文管理マネージャーの初期化
        self.order_manager = OrderManager(self, self.lower_tf_simulator)

        # ヘルパークラスの初期化
        self.position_manager = PositionManager(self)
        self.position_exit_engine = PositionExitEngine(self)
        self.stateful_conditions_evaluator = StatefulConditionsEvaluator(self)
        self.ml_filter = MLFilter(self)
        self.entry_decision_engine = EntryDecisionEngine(self)

        self._minute_data = None  # 1分足DataFrame（パラメータから取得）

        # パラメータの検証と設定
        if params is None:
            params = {}

        super().__init__(broker, data, params)

        # パラメータから遺伝子を取得
        if "strategy_gene" in params:
            self.strategy_gene = params["strategy_gene"]
            self.gene = params["strategy_gene"]
        elif self.strategy_gene is not None:
            # クラス変数から取得（フォールバック）
            self.gene = self.strategy_gene
        else:
            # 安全のためデフォルトの空遺伝子またはエラー
            raise ValueError("UniversalStrategy requires 'strategy_gene' in params")

        # ベースタイムフレーム（パラメータから取得、デフォルトは1h）
        self.base_timeframe = params.get("timeframe", "1h")
        self.evaluation_start = params.get("evaluation_start")
        self._evaluation_start = self._normalize_evaluation_start(self.evaluation_start)
        self.enable_early_termination = bool(
            params.get("enable_early_termination", False)
        )
        self.early_termination_max_drawdown = params.get(
            "early_termination_max_drawdown"
        )
        self.early_termination_min_trades = params.get(
            "early_termination_min_trades"
        )
        self.early_termination_min_trade_check_progress = float(
            params.get("early_termination_min_trade_check_progress", 0.5) or 0.5
        )
        self.early_termination_trade_pace_tolerance = float(
            params.get("early_termination_trade_pace_tolerance", 0.5) or 0.5
        )
        self.early_termination_min_expectancy = params.get(
            "early_termination_min_expectancy"
        )
        self.early_termination_expectancy_min_trades = int(
            params.get("early_termination_expectancy_min_trades", 5) or 5
        )
        self.early_termination_expectancy_progress = float(
            params.get("early_termination_expectancy_progress", 0.6) or 0.6
        )

        # 1分足データの取得（1分足シミュレーション用）
        self._minute_data = params.get("minute_data")
        self._total_bars = max(1, len(data)) if hasattr(data, "__len__") else 1
        (
            self._evaluation_index,
            self._evaluation_start_index,
            self._evaluation_total_bars,
        ) = self._initialize_evaluation_progress_bounds(data)
        self._starting_equity = self._get_current_equity(default=100000.0)
        self._max_equity_seen = self._starting_equity

        # MTFデータプロバイダーの初期化（MTF指標が存在する場合のみ）
        self.mtf_data_provider = None
        if self._has_mtf_indicators():
            from ..services.mtf_data_provider import MultiTimeframeDataProvider

            self.mtf_data_provider = MultiTimeframeDataProvider(
                base_data=data,
                base_timeframe=self.base_timeframe,
            )
            # logger.debug(
            #     f"MTFデータプロバイダー初期化: base_timeframe={self.base_timeframe}"
            # )

        # IndicatorCalculatorの初期化（MTFデータプロバイダー付き）
        self.indicator_calculator = IndicatorCalculator(
            mtf_data_provider=self.mtf_data_provider
        )

        self.indicators = {}

        # === ML フィルター設定 ===
        # HybridPredictor インスタンス（オプション）
        self.ml_predictor = params.get("ml_predictor")
        ml_gate_settings = resolve_ml_gate_settings(params)
        self.volatility_gate_enabled = ml_gate_settings.enabled
        self.volatility_model_path = ml_gate_settings.model_path
        self.ml_filter_enabled = ml_gate_settings.enabled
        if "ml_filter_threshold" in params:
            logger.warning(
                "ml_filter_threshold は非推奨のため無視されます。volatility gate は学習済み cut-off で判定します"
            )
        # 旧互換フィールド。volatility gate 化後は参照しない。
        self.ml_filter_threshold = params.get("ml_filter_threshold", 0.5)

        # ベクトル化評価結果のキャッシュ
        self._precomputed_signals = {}

    def _has_mtf_indicators(self) -> bool:
        """MTF指標が存在するかチェック"""
        if not self.gene or not self.gene.indicators:
            return False
        return any(
            getattr(ind, "timeframe", None) is not None
            for ind in self.gene.indicators
            if ind.enabled
        )

    def _get_effective_sub_gene(self, direction: float, gene_type: str) -> Any:
        """
        方向とタイプに応じた有効なサブ遺伝子を取得（統合版）

        Args:
            direction: 1.0 (Long) or -1.0 (Short)
            gene_type: 'tpsl' or 'entry'

        Returns:
            有効なサブ遺伝子またはNone
        """
        if not self.gene:
            return None

        # フィールド名の構築（例: long_tpsl_gene）
        prefix = "long" if direction > 0 else "short"
        specific_field = f"{prefix}_{gene_type}_gene"
        common_field = f"{gene_type}_gene"

        # 1. 方向別設定を優先
        target_gene = getattr(self.gene, specific_field, None)
        if target_gene and getattr(target_gene, "enabled", True):
            return target_gene

        # 2. フォールバック: 共通設定
        common_gene = getattr(self.gene, common_field, None)
        if common_gene and getattr(common_gene, "enabled", True):
            return common_gene

        return None

    def _get_effective_tpsl_gene(self, direction: float) -> Optional[TPSLGene]:
        """有効なTPSL遺伝子を取得（方向別設定を優先し、共通設定にフォールバック）"""
        target = self._get_effective_sub_gene(direction, "tpsl")
        return cast(Optional[TPSLGene], target)

    def _get_effective_entry_gene(self, direction: float) -> Optional[EntryGene]:
        """有効なエントリー遺伝子を取得（方向別設定を優先し、共通設定にフォールバック）"""
        target = self._get_effective_sub_gene(direction, "entry")
        return cast(Optional[EntryGene], target)

    def _normalize_evaluation_start(self, value: Any) -> Optional[pd.Timestamp]:
        """評価開始時刻を pandas.Timestamp に正規化する。"""
        if value is None or value == "":
            return None

        try:
            return pd.Timestamp(value)
        except Exception:
            logger.warning(f"evaluation_start の解析に失敗しました: {value}")
            return None

    def _is_evaluation_bar(self) -> bool:
        """現在バーが評価開始時刻以降かを返す。"""
        if self._evaluation_start is None:
            return True

        if not hasattr(self.data, "index") or len(self.data.index) == 0:
            return True

        current_time = pd.Timestamp(self.data.index[-1])
        if current_time.tzinfo is not None and self._evaluation_start.tzinfo is None:
            evaluation_start = self._evaluation_start.tz_localize(current_time.tzinfo)
        elif current_time.tzinfo is None and self._evaluation_start.tzinfo is not None:
            evaluation_start = self._evaluation_start.tz_localize(None)
        elif current_time.tzinfo != self._evaluation_start.tzinfo:
            evaluation_start = self._evaluation_start.tz_convert(current_time.tzinfo)
        else:
            evaluation_start = self._evaluation_start

        return current_time >= evaluation_start

    def _initialize_evaluation_progress_bounds(
        self,
        data: Any,
    ) -> tuple[Optional[pd.DatetimeIndex], int, int]:
        """評価進捗計算に使う評価窓の境界を初期化する。"""
        raw_index = getattr(data, "index", None)
        if raw_index is None or len(raw_index) == 0:
            return None, 0, self._total_bars

        try:
            full_index = pd.DatetimeIndex(raw_index)
        except Exception:
            return None, 0, self._total_bars

        start_index = 0
        if self._evaluation_start is not None:
            evaluation_start = self._align_timestamp_to_index_tz(
                self._evaluation_start,
                full_index,
            )
            start_index = int(full_index.searchsorted(evaluation_start, side="left"))

        total_bars = max(1, len(full_index) - start_index)
        return full_index, start_index, total_bars

    @staticmethod
    def _align_timestamp_to_index_tz(
        value: pd.Timestamp,
        index: pd.DatetimeIndex,
    ) -> pd.Timestamp:
        """DatetimeIndex に合わせて Timestamp の timezone をそろえる。"""
        if len(index) == 0:
            return value

        first_index_value = pd.Timestamp(index[0])
        if first_index_value.tzinfo is not None and value.tzinfo is None:
            return value.tz_localize(first_index_value.tzinfo)
        if first_index_value.tzinfo is None and value.tzinfo is not None:
            return value.tz_localize(None)
        if first_index_value.tzinfo != value.tzinfo:
            return value.tz_convert(first_index_value.tzinfo)
        return value

    def _get_current_equity(self, default: float = 0.0) -> float:
        """現在資産を安全に取得する。"""
        try:
            return float(getattr(self, "equity", default) or default)
        except Exception:
            return float(default)

    def _get_progress_ratio(self) -> float:
        """現在までの評価進捗を返す。"""
        evaluation_index = getattr(self, "_evaluation_index", None)
        if isinstance(evaluation_index, pd.DatetimeIndex) and len(evaluation_index) > 0:
            current_index = getattr(self.data, "index", None)
            if current_index is not None and len(current_index) > 0:
                try:
                    current_time = self._align_timestamp_to_index_tz(
                        pd.Timestamp(current_index[-1]),
                        evaluation_index,
                    )
                    current_position = int(
                        evaluation_index.searchsorted(current_time, side="right")
                    )
                    evaluation_start_index = int(
                        getattr(self, "_evaluation_start_index", 0) or 0
                    )
                    evaluation_total_bars = max(
                        1,
                        int(getattr(self, "_evaluation_total_bars", 1) or 1),
                    )
                    evaluated_bars = max(0, current_position - evaluation_start_index)
                    return min(1.0, evaluated_bars / evaluation_total_bars)
                except Exception:
                    logger.debug("評価窓ベースの進捗計算に失敗したためフォールバックします")

        total_bars = max(1, int(getattr(self, "_total_bars", 1) or 1))
        current_bar = max(0, int(getattr(self, "_current_bar_index", 0) or 0))
        return min(1.0, current_bar / total_bars)

    def _calculate_closed_trade_expectancy(self) -> Optional[float]:
        """クローズ済みトレードの平均期待値を返す。"""
        try:
            trades = list(getattr(self, "closed_trades", []) or [])
        except Exception:
            return None

        if not trades:
            return None

        values = []
        for trade in trades:
            for attr_name in ("pl_pct", "pl", "pnl", "return_pct"):
                value = getattr(trade, attr_name, None)
                if value is None:
                    continue
                try:
                    values.append(float(value))
                    break
                except Exception:
                    continue

        if not values:
            return None

        return float(sum(values) / len(values))

    def _should_terminate_early(self) -> Optional[str]:
        """早期打ち切りすべき理由を返す。"""
        if not self.enable_early_termination:
            return None

        current_equity = self._get_current_equity(default=self._starting_equity)
        self._max_equity_seen = max(self._max_equity_seen, current_equity)

        if self.early_termination_max_drawdown is not None and self._max_equity_seen > 0:
            drawdown = max(
                0.0,
                (self._max_equity_seen - current_equity) / self._max_equity_seen,
            )
            if drawdown >= float(self.early_termination_max_drawdown):
                return "max_drawdown"

        progress = self._get_progress_ratio()

        min_trades = self.early_termination_min_trades
        if (
            min_trades is not None
            and progress >= self.early_termination_min_trade_check_progress
        ):
            closed_trade_count = len(getattr(self, "closed_trades", []) or [])
            required_trade_count = max(
                1,
                int(
                    ceil(
                        float(min_trades)
                        * progress
                        * self.early_termination_trade_pace_tolerance
                    )
                ),
            )
            if closed_trade_count < required_trade_count:
                return "trade_pace"

        if (
            self.early_termination_min_expectancy is not None
            and progress >= self.early_termination_expectancy_progress
        ):
            closed_trade_count = len(getattr(self, "closed_trades", []) or [])
            if closed_trade_count >= self.early_termination_expectancy_min_trades:
                expectancy = self._calculate_closed_trade_expectancy()
                if (
                    expectancy is not None
                    and expectancy < float(self.early_termination_min_expectancy)
                ):
                    return "expectancy"

        return None

    def _check_early_termination(self) -> None:
        """早期打ち切り条件を満たした場合に例外を送出する。"""
        reason = self._should_terminate_early()
        if reason:
            raise StrategyEarlyTermination(reason)

    def _check_entry_conditions(self, direction: float) -> bool:
        """指定された方向のエントリー条件をチェック"""
        # ベクトル化済みのシグナルがあればそれを使用 (O(1))
        if direction in self._precomputed_signals:
            signals = self._precomputed_signals[direction]
            if signals is not None:
                # len(self.data) - 1 が現在の足のインデックス
                idx = len(self.data) - 1
                # 配列範囲外アクセスのガード
                if 0 <= idx < len(signals):
                    return bool(signals[idx])

        field_name = (
            "long_entry_conditions" if direction > 0 else "short_entry_conditions"
        )
        conditions = cast(
            List[Union[Condition, ConditionGroup]], getattr(self.gene, field_name, [])
        )

        if not conditions:
            return False

        return self.condition_evaluator.evaluate_conditions(conditions, self)

    def _calculate_position_size(self) -> float:
        """ポジションサイズを計算"""
        try:
            # PositionSizingGeneが有効な場合
            if (
                self.gene.position_sizing_gene
                and self.gene.position_sizing_gene.enabled
            ):
                current_price = (
                    self.data.Close[-1]
                    if hasattr(self, "data") and len(self.data.Close) > 0
                    else 50000.0
                )
                account_balance = getattr(self, "equity", 100000.0)

                # ATR Calculation for market_data
                market_data = {}
                try:
                    # ベクトル化されたATRを使用（高速）
                    if (
                        hasattr(self, "_precomputed_atr")
                        and self._precomputed_atr is not None
                    ):
                        idx = len(self.data) - 1
                        if 0 <= idx < len(self._precomputed_atr):
                            atr = self._precomputed_atr[idx]
                            import numpy as np

                            if not np.isnan(atr) and current_price > 0:
                                market_data["atr_pct"] = atr / current_price

                    # フォールバック（低速）
                    if "atr_pct" not in market_data:
                        lookback = getattr(
                            self.gene.position_sizing_gene, "lookback_period", 14
                        )
                        # Need lookback + 1 for previous close
                        if len(self.data) > lookback + 1:
                            # TR = max(high-low, abs(high-prev_close), abs(low-prev_close))
                            high = self.data.High[-lookback:]
                            low = self.data.Low[-lookback:]
                            prev_close = self.data.Close[-lookback - 1 : -1]

                            import numpy as np

                            # backtesting.pyのデータオブジェクトをnumpy配列に変換して安全に計算
                            high = np.array(self.data.High[-lookback:])
                            low = np.array(self.data.Low[-lookback:])
                            prev_close = np.array(self.data.Close[-lookback - 1 : -1])

                            tr1 = high - low
                            tr2 = np.abs(high - prev_close)
                            tr3 = np.abs(low - prev_close)
                            tr = np.maximum(tr1, np.maximum(tr2, tr3))

                            atr = np.mean(tr)
                            if current_price > 0:
                                market_data["atr_pct"] = atr / current_price
                except Exception:
                    # logger.warning(f"ATR calculation failed in position sizing: {e}")
                    pass

                position_size = (
                    self.position_sizing_service.calculate_position_size_fast(
                        gene=self.gene.position_sizing_gene,
                        account_balance=account_balance,
                        current_price=current_price,
                        market_data=market_data,
                    )
                )
                return max(0.001, min(0.2, float(position_size)))
            return 0.01
        except Exception as e:
            logger.warning(f"ポジションサイズ計算エラー、フォールバック使用: {e}")
            return 0.01

    def init(self):
        """指標の初期化"""
        try:
            if not self.gene:
                return

            # 1. 各指標を初期化
            enabled_indicators = [ind for ind in self.gene.indicators if ind.enabled]

            for indicator_gene in enabled_indicators:
                self._init_indicator(indicator_gene)

            # 2. MLフィルター用の特徴量事前計算
            if self.volatility_gate_enabled and self.ml_predictor:
                self.ml_filter.precompute_ml_features()

            # 3. エントリー条件のベクトル化事前計算
            try:
                # Long
                long_conds = getattr(self.gene, "long_entry_conditions", [])
                if long_conds:
                    self._precomputed_signals[1.0] = (
                        self.condition_evaluator.calculate_conditions_vectorized(
                            long_conds, self
                        )
                    )

                # Short
                short_conds = getattr(self.gene, "short_entry_conditions", [])
                if short_conds:
                    self._precomputed_signals[-1.0] = (
                        self.condition_evaluator.calculate_conditions_vectorized(
                            short_conds, self
                        )
                    )
            except Exception as e:
                # 失敗してもフォールバックするのでログのみ
                logger.debug(f"ベクトル化事前計算失敗（フォールバック使用）: {e}")

            # 4. ポジションサイジング用のATR事前計算
            self._precomputed_atr = None
            if (
                self.gene.position_sizing_gene
                and self.gene.position_sizing_gene.enabled
            ):
                try:
                    lookback = getattr(
                        self.gene.position_sizing_gene, "lookback_period", 14
                    )
                    # pandas-taを使って一括計算
                    # self.data.df がない場合でも一時的にDFを作成して計算
                    try:
                        import pandas_ta_classic as ta

                        # 高速化のため必要なカラムのみでDataFrameを構築
                        # backtesting.pyのデータ配列はnumpy array
                        high = self.data.High
                        low = self.data.Low
                        close = self.data.Close

                        # pandas-taはSeriesを期待するためDF化
                        temp_df = pd.DataFrame(
                            {"high": high, "low": low, "close": close}
                        )

                        # pandas-taで計算
                        atr_series = ta.atr(
                            temp_df["high"],
                            temp_df["low"],
                            temp_df["close"],
                            length=lookback,
                        )
                        if atr_series is not None:
                            self._precomputed_atr = cast(pd.Series, atr_series).values
                            logger.debug("ATR事前計算完了")
                        else:
                            logger.warning(
                                "ATR事前計算失敗: ta.atr が None を返しました"
                            )

                    except ImportError:
                        # pandas-taがない場合などはフォールバック
                        logger.warning(
                            "pandas-taが見つからないためATR事前計算をスキップ"
                        )
                        pass
                    except Exception as e:
                        logger.debug(
                            f"ATR事前計算中のエラー（フォールバック使用）: {e}"
                        )
                except Exception as e:
                    logger.debug(f"ATR事前計算失敗: {e}")

            # 5. TP/SL用のATR事前計算（Long/Shortで異なる場合がある）
            self._precomputed_tpsl_atr = {}
            for direction in [1.0, -1.0]:
                tpsl_gene = self._get_effective_tpsl_gene(direction)
                if tpsl_gene and getattr(tpsl_gene, "method", None) in (
                    TPSLMethod.VOLATILITY_BASED,
                    TPSLMethod.ADAPTIVE,
                    TPSLMethod.STATISTICAL,
                ):
                    try:
                        atr_period = getattr(tpsl_gene, "atr_period", 14)
                        # 同じ期間の計算は一度だけ行う
                        if atr_period not in self._precomputed_tpsl_atr:
                            if hasattr(self.data, "df"):
                                import pandas_ta_classic as ta

                                high = self.data.df["High"]
                                low = self.data.df["Low"]
                                close = self.data.df["Close"]
                                # pandas-taのATR計算
                                atr_result = ta.atr(high, low, close, length=atr_period)
                                if atr_result is not None:
                                    self._precomputed_tpsl_atr[atr_period] = cast(
                                        pd.Series, atr_result
                                    ).values
                    except Exception as e:
                        logger.debug(f"TP/SL ATR事前計算失敗: {e}")

        except Exception as e:
            logger.error(f"戦略初期化エラー: {e}", exc_info=True)
            raise

    def _init_indicator(self, indicator_gene: IndicatorGene):
        """単一指標の初期化"""
        try:
            # 指標計算器を使用して初期化
            self.indicator_calculator.init_indicator(indicator_gene, self)
        except Exception as e:
            logger.error(f"指標初期化エラー {indicator_gene.type}: {e}", exc_info=True)
            # エラーを再発生させて上位で適切に処理
            raise

    def _calculate_effective_tpsl_prices(
        self, direction: float, current_price: float
    ) -> Tuple[Optional[float], Optional[float]]:
        """有効なTP/SL価格を計算"""
        active_tpsl_gene = self._get_effective_tpsl_gene(direction)
        if not active_tpsl_gene:
            return None, None

        market_data = {}
        tpsl_method = active_tpsl_gene.method  # type: ignore[attr-defined]

        if tpsl_method in (
            TPSLMethod.VOLATILITY_BASED,
            TPSLMethod.ADAPTIVE,
            TPSLMethod.STATISTICAL,
        ):
            atr_period = getattr(active_tpsl_gene, "atr_period", 14)

            # 1. 事前計算されたATRを使用（高速）
            if (
                hasattr(self, "_precomputed_tpsl_atr")
                and atr_period in self._precomputed_tpsl_atr
            ):
                idx = len(self.data) - 1
                atr_array = self._precomputed_tpsl_atr[atr_period]
                if 0 <= idx < len(atr_array):
                    val = atr_array[idx]
                    import numpy as np

                    if not np.isnan(val):
                        market_data["atr"] = val

            # 2. フォールバック（事前計算がない場合のみ従来の重い処理）
            if "atr" not in market_data:
                required_slice_size = atr_period + 1
                if len(self.data) > required_slice_size:
                    highs = self.data.High[-required_slice_size:]
                    lows = self.data.Low[-required_slice_size:]
                    closes = self.data.Close[-required_slice_size:]
                    market_data["ohlc_data"] = [
                        {"high": h, "low": low_val, "close": c}
                        for h, low_val, c in zip(highs, lows, closes)
                    ]

        return self.tpsl_service.calculate_tpsl_prices(
            current_price=current_price,
            tpsl_gene=cast(Optional[TPSLGene], active_tpsl_gene),
            position_direction=direction,
            market_data=market_data,
        )

    def next(self):
        """各バーでの戦略実行"""
        try:
            self._current_bar_index += 1

            if not self._is_evaluation_bar():
                return

            # 1. 保留注文とステートフルトリガーの処理
            if self._minute_data is not None:
                self.order_manager.check_pending_order_fills(
                    cast(pd.DataFrame, self._minute_data),
                    self.data.index[-1],
                    self._current_bar_index,
                )
            self.order_manager.expire_pending_orders(self._current_bar_index)
            self.stateful_conditions_evaluator.process_stateful_triggers()

            # 2. 既存ポジションの悲観的決済チェック
            handled_open_position = self.position_exit_engine.handle_open_position()
            self._check_early_termination()
            if handled_open_position:
                return

            # 3. 新規エントリー判定（ノーポジション時）
            if not self.position:
                direction = self.entry_decision_engine.determine_entry_direction()
                if direction == 0.0:
                    return
                self.entry_decision_engine.execute_entry(direction)

        except StrategyEarlyTermination:
            raise
        except Exception as e:
            logger.error(f"戦略実行エラー: {e}")

    # ===== ML フィルターメソッド =====

    def _ml_allows_entry(self, direction: float) -> bool:
        """
        MLフィルターがエントリーを許可するかチェック

        Args:
            direction: 1.0 (Long) or -1.0 (Short)

        Returns:
            True: エントリー許可, False: エントリーブロック
        """
        return self.ml_filter.ml_allows_entry(direction)

    def _prepare_current_features(self) -> Optional[pd.DataFrame]:
        """
        MLフィルター用の現在の特徴量を準備

        Returns:
            特徴量DataFrame、準備できない場合はNone
        """
        return self.ml_filter.prepare_current_features()

    def _process_stateful_triggers(self):
        """ステートフルトリガーを処理"""
        self.stateful_conditions_evaluator.process_stateful_triggers()

    def _get_stateful_entry_direction(self) -> Optional[float]:
        """
        ステートフルエントリーの方向を取得

        Returns:
            1.0 (Long), -1.0 (Short), または None
        """
        return self.stateful_conditions_evaluator.get_stateful_entry_direction()

    # ===== ツールフィルターメソッド =====

    def _tools_block_entry(self) -> bool:
        """
        ツールがエントリーをブロックするかチェック

        tool_genes に設定されたすべてのツールを評価し、
        いずれかがエントリーをスキップすべきと判断した場合 True を返します。

        Returns:
            True: エントリーをブロック（スキップすべき）
            False: エントリーを許可
        """
        # 遺伝子が設定されていない場合はエントリーを許可
        if not self.gene:
            return False
        if not hasattr(self.gene, "tool_genes") or not self.gene.tool_genes:
            return False

        # ツールレジストリをインポート
        from ..tools import ToolContext, tool_registry

        try:
            # 現在のバーのタイムスタンプを取得
            current_timestamp = self.data.index[-1]

            # コンテキストを作成
            context = ToolContext(
                timestamp=current_timestamp,
                current_price=float(self.data.Close[-1]),
                current_high=float(self.data.High[-1]),
                current_low=float(self.data.Low[-1]),
                current_volume=(
                    float(self.data.Volume[-1]) if hasattr(self.data, "Volume") else 0.0
                ),
            )

            # すべての有効なツールをチェック
            for tool_gene in self.gene.tool_genes:
                if not tool_gene.enabled:
                    continue

                # レジストリからツールを取得
                tool = tool_registry.get(tool_gene.tool_name)
                if tool is None:
                    logger.warning(
                        f"ツール '{tool_gene.tool_name}' がレジストリに見つかりません"
                    )
                    continue

                # ツールでフィルタリング
                if tool.should_skip_entry(context, tool_gene.params):
                    # logger.debug(
                    #     f"ツール '{tool_gene.tool_name}' がエントリーをブロック "
                    #     f"(params={tool_gene.params})"
                    # )
                    return True

            return False

        except Exception as e:
            # エラー時はエントリーを許可（フェイルセーフ）
            logger.warning(f"ツールフィルターエラー（フェイルセーフ適用）: {e}")
            return False
