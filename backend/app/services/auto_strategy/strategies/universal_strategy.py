"""
汎用自動生成戦略クラス

GAから生成されたStrategyGeneを受け取り、その定義に基づいて動的に振る舞う
backtesting.py互換の戦略クラスです。
Pickle化可能にするため、filesのトップレベルで定義されています。
"""

import logging
from typing import Any, List, Optional, Tuple, Union, cast

import pandas as pd
from backtesting import Strategy

from ..config.ml_filter_settings import resolve_ml_gate_settings
from ..config.sub_configs import resolve_early_termination_settings
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
from .early_termination import (
    StrategyEarlyTermination,
    StrategyEarlyTerminationController,
)
from .execution_cycle import StrategyExecutionCycle
from .ml_filter import MLFilter
from .order_manager import OrderManager
from .position_manager import PositionManager
from .runtime_state import StrategyRuntimeState
from .stateful_conditions import StatefulConditionsEvaluator
from .strategy_initializer import StrategyInitializer

logger = logging.getLogger(__name__)

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
    early_termination_settings = None
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
        self.stateful_conditions_evaluator = StatefulConditionsEvaluator(self)
        self.early_termination_controller = StrategyEarlyTerminationController(self)
        self.ml_filter = MLFilter(self)
        self.entry_decision_engine = EntryDecisionEngine(self)
        self.strategy_initializer = StrategyInitializer(self)
        self.execution_cycle = StrategyExecutionCycle(self)

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
        resolved_early_termination_settings = resolve_early_termination_settings(params)
        if params.get("early_termination_settings") is not None:
            self.early_termination_settings = resolved_early_termination_settings
        else:
            self.early_termination_settings = None
        early_termination_params = resolved_early_termination_settings.to_strategy_params()
        self.enable_early_termination = bool(
            early_termination_params["enable_early_termination"]
        )
        self.early_termination_max_drawdown = early_termination_params[
            "early_termination_max_drawdown"
        ]
        self.early_termination_min_trades = early_termination_params[
            "early_termination_min_trades"
        ]
        self.early_termination_min_trade_check_progress = float(
            early_termination_params["early_termination_min_trade_check_progress"]
        )
        self.early_termination_trade_pace_tolerance = float(
            early_termination_params["early_termination_trade_pace_tolerance"]
        )
        self.early_termination_min_expectancy = early_termination_params[
            "early_termination_min_expectancy"
        ]
        self.early_termination_expectancy_min_trades = int(
            early_termination_params["early_termination_expectancy_min_trades"]
        )
        self.early_termination_expectancy_progress = float(
            early_termination_params["early_termination_expectancy_progress"]
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
        return self.early_termination_controller.normalize_evaluation_start(value)

    def _is_evaluation_bar(self) -> bool:
        """現在バーが評価開始時刻以降かを返す。"""
        return self.early_termination_controller.is_evaluation_bar()

    def _initialize_evaluation_progress_bounds(
        self,
        data: Any,
    ) -> tuple[Optional[pd.DatetimeIndex], int, int]:
        """評価進捗計算に使う評価窓の境界を初期化する。"""
        return self.early_termination_controller.initialize_evaluation_progress_bounds(
            data
        )

    @staticmethod
    def _align_timestamp_to_index_tz(
        value: pd.Timestamp,
        index: pd.DatetimeIndex,
    ) -> pd.Timestamp:
        """DatetimeIndex に合わせて Timestamp の timezone をそろえる。"""
        return StrategyEarlyTerminationController.align_timestamp_to_index_tz(
            value,
            index,
        )

    def _get_current_equity(self, default: float = 0.0) -> float:
        """現在資産を安全に取得する。"""
        return self.early_termination_controller.get_current_equity(default)

    def _get_progress_ratio(self) -> float:
        """現在までの評価進捗を返す。"""
        return self.early_termination_controller.get_progress_ratio()

    def _calculate_closed_trade_expectancy(self) -> Optional[float]:
        """クローズ済みトレードの平均期待値を返す。"""
        return self.early_termination_controller.calculate_closed_trade_expectancy()

    def _should_terminate_early(self) -> Optional[str]:
        """早期打ち切りすべき理由を返す。"""
        return self.early_termination_controller.should_terminate_early()

    def _check_early_termination(self) -> None:
        """早期打ち切り条件を満たした場合に例外を送出する。"""
        reason = self._should_terminate_early()
        if reason:
            raise StrategyEarlyTermination(reason)

    def _check_entry_conditions(self, direction: float) -> bool:
        """指定された方向のエントリー条件をチェック"""
        return self.entry_decision_engine.check_entry_conditions(direction)

    def _calculate_position_size(self) -> float:
        """ポジションサイズを計算"""
        return self.entry_decision_engine.calculate_position_size()

    def init(self):
        """指標の初期化"""
        self.strategy_initializer.initialize()

    def _init_indicator(self, indicator_gene: IndicatorGene):
        """単一指標の初期化"""
        self.strategy_initializer.init_indicator(indicator_gene)

    def _calculate_effective_tpsl_prices(
        self, direction: float, current_price: float
    ) -> Tuple[Optional[float], Optional[float]]:
        """有効なTP/SL価格を計算"""
        return self.entry_decision_engine.calculate_effective_tpsl_prices(
            direction,
            current_price,
        )

    def next(self):
        """各バーでの戦略実行"""
        try:
            self._current_bar_index += 1
            self.execution_cycle.run_current_bar()

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
        return self.entry_decision_engine.tools_block_entry()
