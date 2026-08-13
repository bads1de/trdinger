"""
汎用自動生成戦略クラス

GAから生成されたStrategyGeneを受け取り、その定義に基づいて動的に振る舞う
backtesting.py互換の戦略クラスです。
Pickle化可能にするため、filesのトップレベルで定義されています。
"""

import logging
from typing import Any, cast

import pandas as pd
from backtesting import Strategy

from ..config.ga.nested_configs import EarlyTerminationSettings
from ..core.evaluation.condition_evaluator import ConditionEvaluator
from ..genes import ExitGene, IndicatorGene, TPSLGene
from ..genes.conditions import StateTracker
from ..genes.entry import EntryGene
from ..positions.entry_executor import EntryExecutor
from ..positions.lower_tf_simulator import LowerTimeframeSimulator
from ..positions.position_sizing_service import PositionSizingService
from ..services.indicator_service import IndicatorCalculator
from ..tpsl.tpsl_service import TPSLService
from .early_termination import (
    StrategyEarlyTermination,
    StrategyEarlyTerminationController,
)
from .entry_decision_engine import EntryDecisionEngine
from .execution_cycle import StrategyExecutionCycle
from .exit_decision_engine import ExitDecisionEngine
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
    early_termination_settings = None

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

    def __init__(self, broker: Any, data: Any, params: Any) -> None:
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
        self.entry_decision_engine = EntryDecisionEngine(self)
        self.exit_decision_engine = ExitDecisionEngine(self)
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
        early_termination_settings = params.get("early_termination_settings")
        if early_termination_settings is None:
            self.early_termination_settings = EarlyTerminationSettings()
        elif isinstance(early_termination_settings, EarlyTerminationSettings):
            self.early_termination_settings = early_termination_settings
        else:
            self.early_termination_settings = EarlyTerminationSettings.from_source(
                early_termination_settings
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

        self.indicators: dict[str, Any] = {}

        # ベクトル化評価結果のキャッシュ
        self._precomputed_signals: dict[float, Any] = {}
        self._precomputed_exit_signals: dict[float, Any] = {}

    def _has_mtf_indicators(self) -> bool:
        """MTF指標が存在するかチェック"""
        if not self.gene or not self.gene.indicators:
            return False
        return any(
            getattr(ind, "timeframe", None) is not None
            for ind in self.gene.indicators
            if ind.enabled
        )

    def _get_effective_sub_gene(self, direction: float, gene_type: str) -> object:
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

    def _get_effective_tpsl_gene(self, direction: float) -> TPSLGene | None:
        """有効なTPSL遺伝子を取得（方向別設定を優先し、共通設定にフォールバック）"""
        target = self._get_effective_sub_gene(direction, "tpsl")
        return cast(TPSLGene | None, target)

    def _get_effective_entry_gene(self, direction: float) -> EntryGene | None:
        """有効なエントリー遺伝子を取得（方向別設定を優先し、共通設定にフォールバック）"""
        target = self._get_effective_sub_gene(direction, "entry")
        return cast(EntryGene | None, target)

    def _get_effective_exit_gene(self, direction: float) -> ExitGene | None:
        """有効なイグジット遺伝子を取得（方向別設定を優先し、共通設定にフォールバック）"""
        target = self._get_effective_sub_gene(direction, "exit")
        return cast(ExitGene | None, target)

    def _normalize_evaluation_start(
        self, value: str | pd.Timestamp | None
    ) -> pd.Timestamp | None:
        """評価開始時刻を pandas.Timestamp に正規化する。"""
        return self.early_termination_controller.normalize_evaluation_start(value)

    def _is_evaluation_bar(self) -> bool:
        """現在バーが評価開始時刻以降かを返す。"""
        return self.early_termination_controller.is_evaluation_bar()

    def _initialize_evaluation_progress_bounds(
        self,
        data: object,
    ) -> tuple[pd.DatetimeIndex | None, int, int]:
        """評価進捗計算に使う評価窓の境界を初期化する。"""
        return self.early_termination_controller.initialize_evaluation_progress_bounds(
            data
        )

    def _get_current_equity(self, default: float = 0.0) -> float:
        """現在資産を安全に取得する。"""
        return self.early_termination_controller.get_current_equity(default)

    def _should_terminate_early(self) -> str | None:
        """早期打ち切りすべき理由を返す。"""
        return self.early_termination_controller.should_terminate_early()

    def _check_early_termination(self) -> None:
        """早期打ち切り条件を満たした場合に例外を送出する。"""
        reason = self._should_terminate_early()
        if reason:
            raise StrategyEarlyTermination(reason)

    def init(self) -> None:
        """
        戦略の初期化フェーズ（`backtesting.py` のライフサイクル）。

        このメソッドは、バックテスト開始前に一度だけ呼び出されます。
        `StrategyInitializer` を使用して、以下の初期化処理を実行します：
        1. `StrategyGene` に定義されたテクニカル指標の計算とキャッシュ。
        2. エントリー条件、決済条件、およびステートフルトリガーのセットアップ。
        3. ポジション管理（利確・損切り）および注文管理コンポーネントの初期化。
        4. ボラティリティゲート等の最終チェック準備。

        Note:
            `backtesting.py` では `init()` 内で全てのインジケータ（`I()`関数を使用）を
            宣言する必要があります。本クラスでは `StrategyInitializer` がこの役割を担います。
        """
        self.strategy_initializer.initialize()

    def _init_indicator(self, indicator_gene: IndicatorGene) -> None:
        """
        単一のテクニカル指標を初期化します。

        Args:
            indicator_gene (IndicatorGene): 指標の定義情報（タイプ、パラメータ、時間軸等）。
        """
        self.strategy_initializer.init_indicator(indicator_gene)

    def next(self) -> None:
        """
        メインの戦略実行ループ（`backtesting.py` のライフサイクル）。

        各バー（ローソク足）が確定するたびに呼び出され、以下のサイクルを実行します（`StrategyExecutionCycle` に委譲）：
        1. インデックスの更新と現在の市場データの取得。
        2. 保有中ポジションの状態確認とトレールストップ等の更新。
        3. 決済条件（Exit Conditions）の評価と、合致する場合の成行決済。
        4. 新規エントリー条件（Entry Conditions）の評価。
        5. エントリー許可時、ボラティリティゲートによる最終チェック。
        6. 全てのフィルターを通過した場合、適切なロットサイズでの注文執行。
        7. 早期終了条件（Early Termination）のチェック。

        Raises:
            StrategyEarlyTermination: ドローダウン制限などの早期終了条件に合致した場合に発生し、
                バックテスト全体を中断させます。
            Exception: 実行中に予期しないエラーが発生した場合。エラーはログに記録され、次のバーへ進みます。
        """
        try:
            self._current_bar_index += 1
            self.execution_cycle.run_current_bar()

        except StrategyEarlyTermination:
            raise
        except Exception as e:
            logger.error(
                f"戦略実行エラー (bar={self._current_bar_index}): {e}",
                exc_info=True,
            )
            # エラー発生時は安全な状態にリセットし、次のバーで継続できるようにする
            self.position_manager.reset_position_state()
