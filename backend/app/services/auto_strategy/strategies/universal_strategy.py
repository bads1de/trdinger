"""
汎用自動生成戦略クラス

GAから生成されたStrategyGeneを受け取り、その定義に基づいて動的に振る舞う
backtesting.py互換の戦略クラスです。
Pickle化可能にするため、filesのトップレベルで定義されています。
"""

import logging
from typing import Any, Dict, List, Union, cast

import numpy as np

import pandas as pd
from backtesting import Strategy

from ..core.condition_evaluator import ConditionEvaluator
from ..models.entry_gene import EntryGene
from ..config.enums import EntryType
from ..models.pending_order import PendingOrder
from ..models.conditions import StateTracker
from ..models import (
    Condition,
    ConditionGroup,
    IndicatorGene,
    TPSLMethod,
)
from ..positions.entry_executor import EntryExecutor
from ..positions.lower_tf_simulator import LowerTimeframeSimulator
from ..positions.position_sizing_service import PositionSizingService
from ..services.indicator_service import IndicatorCalculator
from ..tpsl.tpsl_service import TPSLService

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
    ml_predictor = None  # MLフィルター用予測器
    ml_filter_threshold = (
        0.5  # MLフィルター閾値（is_valid >= threshold でエントリー許可）
    )

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
        self._current_bar_index = 0  # バーインデックストラッカー

        # 保留注文管理用
        self._pending_orders: list[PendingOrder] = []
        self._minute_data = None  # 1分足DataFrame（パラメータから取得）

        # 悲観的約定ロジック用: SL/TP管理変数
        self._sl_price: float | None = None
        self._tp_price: float | None = None
        self._entry_price: float | None = None
        self._position_direction: float = 0.0  # 1.0=Long, -1.0=Short, 0.0=No position
        self._tp_reached: bool = False  # トレーリングTP用: TP到達後フラグ
        self._trailing_tp_sl: float | None = None  # トレーリングTP用: 利益確保ライン

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

        # 1分足データの取得（1分足シミュレーション用）
        self._minute_data = params.get("minute_data")

        # MTFデータプロバイダーの初期化（MTF指標が存在する場合のみ）
        self.mtf_data_provider = None
        if self._has_mtf_indicators():
            from ..services.mtf_data_provider import MultiTimeframeDataProvider

            self.mtf_data_provider = MultiTimeframeDataProvider(
                base_data=data,
                base_timeframe=self.base_timeframe,
            )
            logger.debug(
                f"MTFデータプロバイダー初期化: base_timeframe={self.base_timeframe}"
            )

        # IndicatorCalculatorの初期化（MTFデータプロバイダー付き）
        self.indicator_calculator = IndicatorCalculator(
            mtf_data_provider=self.mtf_data_provider
        )

        self.indicators = {}

        # === ML フィルター設定 ===
        # HybridPredictor インスタンス（オプション）
        self.ml_predictor = params.get("ml_predictor")
        # ML フィルター閾値（エントリー許可のための方向スコア差分）
        self.ml_filter_threshold = params.get("ml_filter_threshold", 0.5)

    def _has_mtf_indicators(self) -> bool:
        """MTF指標が存在するかチェック"""
        if not self.gene or not self.gene.indicators:
            return False
        return any(
            getattr(ind, "timeframe", None) is not None
            for ind in self.gene.indicators
            if ind.enabled
        )

    def _get_effective_tpsl_gene(self, direction: float) -> Union[None, object]:
        """
        方向に応じた有効なTPSL遺伝子を取得

        Args:
            direction: 1.0 (Long) or -1.0 (Short)

        Returns:
            有効なTPSLGeneまたはNone
        """
        if not self.gene:
            return None

        # 方向別設定を優先確認
        target_gene = None
        if direction > 0:  # Long
            if hasattr(self.gene, "long_tpsl_gene") and self.gene.long_tpsl_gene:
                target_gene = self.gene.long_tpsl_gene
        elif direction < 0:  # Short
            if hasattr(self.gene, "short_tpsl_gene") and self.gene.short_tpsl_gene:
                target_gene = self.gene.short_tpsl_gene

        # 有効な個別設定があれば返す
        if target_gene and target_gene.enabled:
            return target_gene

        # フォールバック: 共通設定
        if self.gene.tpsl_gene and self.gene.tpsl_gene.enabled:
            return self.gene.tpsl_gene

        return None

    def _get_effective_entry_gene(self, direction: float) -> Union[None, EntryGene]:
        """
        方向に応じた有効なエントリー遺伝子を取得

        Args:
            direction: 1.0 (Long) or -1.0 (Short)

        Returns:
            有効なEntryGeneまたはNone
        """
        if not self.gene:
            return None

        # 方向別設定を優先確認
        target_gene = None
        if direction > 0:  # Long
            if hasattr(self.gene, "long_entry_gene") and self.gene.long_entry_gene:
                target_gene = self.gene.long_entry_gene
        elif direction < 0:  # Short
            if hasattr(self.gene, "short_entry_gene") and self.gene.short_entry_gene:
                target_gene = self.gene.short_entry_gene

        # 有効な個別設定があれば返す
        if target_gene and target_gene.enabled:
            return target_gene

        # フォールバック: 共通設定
        if (
            hasattr(self.gene, "entry_gene")
            and self.gene.entry_gene
            and self.gene.entry_gene.enabled
        ):
            return self.gene.entry_gene

        return None

    def init(self):
        """指標の初期化"""
        try:
            if not self.gene:
                return

            # 各指標を初期化
            enabled_indicators = [ind for ind in self.gene.indicators if ind.enabled]

            for indicator_gene in enabled_indicators:
                self._init_indicator(indicator_gene)

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

    def _check_long_entry_conditions(self) -> bool:
        """ロングエントリー条件をチェック"""
        long_conditions = cast(
            List[Union[Condition, ConditionGroup]],
            self.gene.get_effective_long_conditions(),
        )

        if not long_conditions:
            # 条件が空の場合は、entry_conditionsを使用
            if self.gene.entry_conditions:
                entry_conditions = cast(
                    List[Union[Condition, ConditionGroup]],
                    self.gene.entry_conditions,
                )
                return self.condition_evaluator.evaluate_conditions(
                    entry_conditions, self
                )
            return False

        return self.condition_evaluator.evaluate_conditions(long_conditions, self)

    def _check_short_entry_conditions(self) -> bool:
        """ショートエントリー条件をチェック"""
        short_conditions = cast(
            List[Union[Condition, ConditionGroup]],
            self.gene.get_effective_short_conditions(),
        )

        if not short_conditions:
            # ショート条件が空の場合は、entry_conditionsを使用
            if self.gene.entry_conditions:
                entry_conditions = cast(
                    List[Union[Condition, ConditionGroup]],
                    self.gene.entry_conditions,
                )
                return self.condition_evaluator.evaluate_conditions(
                    entry_conditions, self
                )
            return False

        return self.condition_evaluator.evaluate_conditions(short_conditions, self)

    def _check_exit_conditions(self) -> bool:
        """イグジット条件をチェック"""
        # TP/SL遺伝子が存在し有効な場合はイグジット条件をスキップ
        if self.gene.tpsl_gene and self.gene.tpsl_gene.enabled:
            return False

        # 通常のイグジット条件をチェック
        exit_conditions = cast(
            List[Union[Condition, ConditionGroup]], self.gene.exit_conditions
        )
        if not exit_conditions:
            return False

        return self.condition_evaluator.evaluate_conditions(exit_conditions, self)

    def _calculate_position_size(self) -> float:
        """
        ポジションサイズを計算

        Note: キャッシュを使用しない。エントリー時に毎回計算することで、
        口座残高の変動（複利効果）に対応する。
        """
        try:
            # PositionSizingGeneが有効な場合
            if (
                self.gene.position_sizing_gene
                and self.gene.position_sizing_gene.enabled
            ):
                # 現在の市場データ（該当するものがなければデフォルト値を使用）
                current_price = (
                    self.data.Close[-1]
                    if hasattr(self, "data") and len(self.data.Close) > 0
                    else 50000.0
                )

                # 現在の口座残高を取得（backtesting.py の equity 属性）
                # これにより複利効果が機能する
                account_balance = getattr(
                    self, "equity", 100000.0
                )  # デフォルト口座残高

                # 高速計算メソッドを使用（VaR等の重い計算をスキップ）
                position_size = (
                    self.position_sizing_service.calculate_position_size_fast(
                        gene=self.gene.position_sizing_gene,
                        account_balance=account_balance,
                        current_price=current_price,
                    )
                )

                # 結果を安全範囲に制限
                return max(0.001, min(0.2, float(position_size)))
            else:
                # デフォルトサイズを使用
                return 0.01

        except Exception as e:
            logger.warning(f"ポジションサイズ計算エラー、フォールバック使用: {e}")
            # エラー時はデフォルトサイズを使用
            return 0.01

    def next(self):
        """各バーでの戦略実行"""
        try:
            # バーインデックスを更新
            self._current_bar_index += 1

            # === 保留注文の約定チェック（1分足シミュレーション） ===
            self._check_pending_order_fills()
            self._expire_pending_orders()

            # ステートフル条件のトリガーをチェック・記録
            self._process_stateful_triggers()

            # === 悲観的約定ロジック (Pessimistic Exit) ===
            # ポジションがあり、SL/TPが設定されている場合、SL優先で決済判定
            if self.position and self._sl_price is not None:
                exited = self._check_pessimistic_exit()
                if exited:
                    return  # 決済したのでこの足ではこれ以上処理しない

            # ポジションがない場合のエントリー判定
            if not self.position:
                # === ツールフィルター判定 ===
                # 週末フィルターなど、登録されたツールでエントリーをフィルタリング
                if self._tools_block_entry():
                    return  # いずれかのツールがエントリーをブロック

                long_signal = self._check_long_entry_conditions()
                short_signal = self._check_short_entry_conditions()

                # ステートフル条件も評価（方向情報付き）
                stateful_direction = self._get_stateful_entry_direction()

                # 通常条件またはステートフル条件いずれかでエントリー
                if long_signal or short_signal or stateful_direction is not None:
                    # ポジションサイズを決定
                    position_size = self._calculate_position_size()
                    current_price = self.data.Close[-1]

                    # エントリー方向を決定
                    # 優先順位: long_signal > short_signal > stateful_direction
                    direction = 0.0
                    if long_signal:
                        direction = 1.0
                    elif short_signal:
                        direction = -1.0
                    elif stateful_direction is not None:
                        direction = stateful_direction

                    # === ML フィルター判定 ===
                    # エントリー方向が決まった後、MLが許可するかチェック
                    if direction != 0.0 and self.ml_predictor is not None:
                        if not self._ml_allows_entry(direction):
                            # MLがエントリーを拒否した場合、スキップ
                            return

                    # TP/SL価格を計算
                    sl_price = None
                    tp_price = None

                    if direction != 0.0:
                        active_tpsl_gene = self._get_effective_tpsl_gene(direction)

                        if active_tpsl_gene:
                            # 市場データの準備（必要な場合のみ）
                            market_data = {}
                            tpsl_method = active_tpsl_gene.method

                            # ボラティリティベース、適応型、または統計的手法の場合、OHLCデータを作成
                            if tpsl_method in (
                                TPSLMethod.VOLATILITY_BASED,
                                TPSLMethod.ADAPTIVE,
                                TPSLMethod.STATISTICAL,
                            ):
                                # atr_period に基づいて必要なスライスサイズを決定
                                # True Range 計算には (atr_period + 1) 本のデータが必要
                                atr_period = getattr(active_tpsl_gene, "atr_period", 14)
                                required_slice_size = atr_period + 1

                                if len(self.data) > required_slice_size:
                                    # 必要な期間のみスライス（動的に決定）
                                    highs = self.data.High[-required_slice_size:]
                                    lows = self.data.Low[-required_slice_size:]
                                    closes = self.data.Close[-required_slice_size:]

                                    # VolatilityCalculatorが期待する形式（list of dicts）に変換
                                    market_data["ohlc_data"] = [
                                        {"high": h, "low": low_val, "close": c}
                                        for h, low_val, c in zip(highs, lows, closes)
                                    ]

                            # TPSLServiceを使用して価格を計算
                            sl_price, tp_price = (
                                self.tpsl_service.calculate_tpsl_prices(
                                    current_price=current_price,
                                    tpsl_gene=active_tpsl_gene,
                                    position_direction=direction,
                                    market_data=market_data,
                                )
                            )

                    # エントリー注文パラメータを計算
                    entry_gene = self._get_effective_entry_gene(direction)
                    entry_params = self.entry_executor.calculate_entry_params(
                        entry_gene, current_price, direction
                    )

                    # 注文タイプを判定
                    is_market_order = (
                        entry_gene is None
                        or not entry_gene.enabled
                        or entry_gene.entry_type == EntryType.MARKET
                    )

                    if is_market_order:
                        # 成行注文: 即時実行
                        if direction > 0:  # Long
                            self.buy(size=position_size)
                            self._entry_price = current_price
                            self._sl_price = sl_price
                            self._tp_price = tp_price
                            self._position_direction = 1.0
                        elif direction < 0:  # Short
                            self.sell(size=position_size)
                            self._entry_price = current_price
                            self._sl_price = sl_price
                            self._tp_price = tp_price
                            self._position_direction = -1.0
                    else:
                        # 指値/逆指値注文: 保留リストに追加
                        self._create_pending_order(
                            direction=direction,
                            size=position_size,
                            entry_params=entry_params,
                            sl_price=sl_price,
                            tp_price=tp_price,
                            entry_gene=entry_gene,
                        )

            # ポジションがある場合のイグジット判定
            elif self.position and self._check_exit_conditions():
                self.position.close()

        except Exception as e:
            logger.error(f"戦略実行エラー: {e}")

    def _check_pessimistic_exit(self) -> bool:
        """
        悲観的約定ロジックによるSL/TP判定

        同一足内でSLとTPの両方に達した場合、SLを優先して決済します。
        これにより「幻の利益」を防ぎ、バックテスト結果を安全側に倒します。

        Returns:
            True: 決済が実行された場合
            False: 決済が実行されなかった場合
        """
        if self._sl_price is None:
            return False

        current_low = self.data.Low[-1]
        current_high = self.data.High[-1]

        # ロングポジションの場合
        if self._position_direction > 0:
            # トレーリングTP到達後モード: 利益確保ラインで決済判定
            if self._tp_reached and self._trailing_tp_sl is not None:
                if current_low <= self._trailing_tp_sl:
                    self.position.close()
                    self._reset_position_state()
                    return True
                # 利益確保ラインを更新（さらに上昇した場合）
                self._update_trailing_tp_sl()
                return False

            # 1. SL判定 [最優先]: Low <= SL価格
            if current_low <= self._sl_price:
                self.position.close()
                self._reset_position_state()
                return True

            # 2. TP判定 [次点]: High >= TP価格
            if self._tp_price is not None and current_high >= self._tp_price:
                # トレーリングTPが有効な場合は即時決済せず、利益確保モードへ
                if self._is_trailing_tp_enabled():
                    self._tp_reached = True
                    # 初期利益確保ライン = TP価格（ここから追従開始）
                    self._trailing_tp_sl = self._tp_price
                    self._update_trailing_tp_sl()
                    return False
                else:
                    self.position.close()
                    self._reset_position_state()
                    return True

        # ショートポジションの場合
        elif self._position_direction < 0:
            # トレーリングTP到達後モード: 利益確保ラインで決済判定
            if self._tp_reached and self._trailing_tp_sl is not None:
                if current_high >= self._trailing_tp_sl:
                    self.position.close()
                    self._reset_position_state()
                    return True
                # 利益確保ラインを更新（さらに下落した場合）
                self._update_trailing_tp_sl()
                return False

            # 1. SL判定 [最優先]: High >= SL価格 (ショートはSLが上側)
            if current_high >= self._sl_price:
                self.position.close()
                self._reset_position_state()
                return True

            # 2. TP判定 [次点]: Low <= TP価格 (ショートはTPが下側)
            if self._tp_price is not None and current_low <= self._tp_price:
                # トレーリングTPが有効な場合は即時決済せず、利益確保モードへ
                if self._is_trailing_tp_enabled():
                    self._tp_reached = True
                    # 初期利益確保ライン = TP価格（ここから追従開始）
                    self._trailing_tp_sl = self._tp_price
                    self._update_trailing_tp_sl()
                    return False
                else:
                    self.position.close()
                    self._reset_position_state()
                    return True

        # === トレーリングストップ更新 ===
        # 決済条件に達しなかった場合、トレーリングが有効ならSLを更新
        self._update_trailing_stop()

        return False

    def _reset_position_state(self) -> None:
        """ポジション決済後に内部状態をリセット"""
        self._sl_price = None
        self._tp_price = None
        self._entry_price = None
        self._position_direction = 0.0
        self._tp_reached = False
        self._trailing_tp_sl = None

    def _is_trailing_tp_enabled(self) -> bool:
        """トレーリングTPが有効かどうかを確認"""
        active_tpsl_gene = self._get_effective_tpsl_gene(self._position_direction)
        if not active_tpsl_gene:
            return False
        return getattr(active_tpsl_gene, "trailing_take_profit", False)

    def _update_trailing_tp_sl(self) -> None:
        """
        トレーリングTP用の利益確保ラインを更新

        TP到達後、価格がさらに有利な方向に動いた場合、
        利益確保ライン（実質的なSL）を追従させます。
        """
        if not self._tp_reached or self._trailing_tp_sl is None:
            return

        active_tpsl_gene = self._get_effective_tpsl_gene(self._position_direction)
        if not active_tpsl_gene:
            return

        trailing_step = getattr(active_tpsl_gene, "trailing_step_pct", 0.01)
        current_close = self.data.Close[-1]

        # ロングポジションの場合: 終値ベースで新しい利益確保ラインを計算
        if self._position_direction > 0:
            new_trailing_sl = current_close * (1.0 - trailing_step)
            if new_trailing_sl > self._trailing_tp_sl:
                self._trailing_tp_sl = new_trailing_sl
                logger.debug(
                    f"トレーリングTP利益確保ライン更新 (Long): {self._trailing_tp_sl:.2f}"
                )

        # ショートポジションの場合
        elif self._position_direction < 0:
            new_trailing_sl = current_close * (1.0 + trailing_step)
            if new_trailing_sl < self._trailing_tp_sl:
                self._trailing_tp_sl = new_trailing_sl
                logger.debug(
                    f"トレーリングTP利益確保ライン更新 (Short): {self._trailing_tp_sl:.2f}"
                )

    def _update_trailing_stop(self) -> None:
        """
        トレーリングストップの更新

        価格が有利な方向に動いた場合、SLを追従させます。
        SLは有利な方向にのみ移動し、不利な方向には絶対に戻しません。
        """
        # トレーリングが有効か確認
        active_tpsl_gene = self._get_effective_tpsl_gene(self._position_direction)
        if not active_tpsl_gene:
            return
        if not getattr(active_tpsl_gene, "trailing_stop", False):
            return
        if self._sl_price is None:
            return

        trailing_step = getattr(active_tpsl_gene, "trailing_step_pct", 0.01)
        current_close = self.data.Close[-1]

        # ロングポジションの場合: 終値ベースで新SLを計算し、現在SLより高ければ更新
        if self._position_direction > 0:
            new_sl = current_close * (1.0 - trailing_step)
            if new_sl > self._sl_price:
                self._sl_price = new_sl
                logger.debug(f"トレーリングSL更新 (Long): {self._sl_price:.2f}")

        # ショートポジションの場合: 終値ベースで新SLを計算し、現在SLより低ければ更新
        elif self._position_direction < 0:
            new_sl = current_close * (1.0 + trailing_step)
            if new_sl < self._sl_price:
                self._sl_price = new_sl
                logger.debug(f"トレーリングSL更新 (Short): {self._sl_price:.2f}")

    def _process_stateful_triggers(self) -> None:
        """
        ステートフル条件のトリガーをチェックし、StateTrackerに記録

        各バーで呼ばれ、すべてのStatefulConditionのトリガー条件を評価します。
        成立していれば、StateTrackerにイベントとして記録します。
        """
        if not self.gene or not hasattr(self.gene, "stateful_conditions"):
            return

        for stateful_cond in self.gene.stateful_conditions:
            if stateful_cond.enabled:
                self.condition_evaluator.check_and_record_trigger(
                    stateful_cond,
                    self,
                    self.state_tracker,
                    self._current_bar_index,
                )

    def _check_stateful_conditions(self) -> bool:
        """
        ステートフル条件を評価

        いずれかのステートフル条件が成立していればTrueを返します。

        Returns:
            ステートフル条件成立ならTrue
        """
        if not self.gene or not hasattr(self.gene, "stateful_conditions"):
            return False

        for stateful_cond in self.gene.stateful_conditions:
            if stateful_cond.enabled:
                result = self.condition_evaluator.evaluate_stateful_condition(
                    stateful_cond,
                    self,
                    self.state_tracker,
                    self._current_bar_index,
                )
                if result:
                    return True

        return False

    def _get_stateful_entry_direction(self) -> "float | None":
        """
        成立したステートフル条件からエントリー方向を取得

        いずれかのステートフル条件が成立していれば、その条件に設定された
        direction を元にエントリー方向を返します。

        Returns:
            1.0 (Long), -1.0 (Short), または None（条件不成立時）
        """
        if not self.gene or not hasattr(self.gene, "stateful_conditions"):
            return None

        for stateful_cond in self.gene.stateful_conditions:
            if stateful_cond.enabled:
                result = self.condition_evaluator.evaluate_stateful_condition(
                    stateful_cond,
                    self,
                    self.state_tracker,
                    self._current_bar_index,
                )
                if result:
                    # direction フィールドに基づいてエントリー方向を返す
                    direction = getattr(stateful_cond, "direction", "long")
                    return 1.0 if direction == "long" else -1.0

        return None

    # ===== ML フィルターメソッド =====

    def _ml_allows_entry(self, direction: float) -> bool:
        """
        MLがエントリーを許可するかチェック

        MLフィルター（ダマシ予測モデル）が設定されている場合、
        予測結果に基づいてエントリーの可否を判断します。

        ダマシ予測モデルは「このエントリーシグナルが有効かどうか」を
        0-1の確率で出力します。is_valid が閾値以上であればエントリーを許可。

        Args:
            direction: 取引方向 (1.0=Long, -1.0=Short) ※現在は方向に関係なく判定

        Returns:
            True: エントリー許可, False: エントリー拒否
        """
        # ML予測器が設定されていない場合はエントリーを許可
        if self.ml_predictor is None:
            return True

        # ML予測器が学習済みでない場合はエントリーを許可
        try:
            if hasattr(self.ml_predictor, "is_trained"):
                if not self.ml_predictor.is_trained():
                    logger.debug("ML予測器未学習: エントリー許可")
                    return True
        except Exception as e:
            logger.warning(f"ML学習状態チェックエラー: {e}")
            return True

        try:
            # 現在の特徴量を準備
            features = self._prepare_current_features()

            # ML予測を実行
            prediction = self.ml_predictor.predict(features)

            # ダマシ予測モデルの判定
            # is_valid: エントリーが有効である確率 (0.0-1.0)
            # 閾値以上であればエントリーを許可
            is_valid = prediction.get("is_valid", 0.5)
            allowed = is_valid >= self.ml_filter_threshold

            if not allowed:
                logger.debug(
                    f"MLフィルター拒否: direction={direction}, "
                    f"is_valid={is_valid:.3f}, threshold={self.ml_filter_threshold}"
                )

            return allowed

        except Exception as e:
            # 予測エラー時はエントリーを許可（フェイルセーフ）
            logger.warning(f"ML予測エラー（フェイルセーフ適用）: {e}")
            return True

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
        from ..tools import tool_registry, ToolContext

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
                    logger.debug(
                        f"ツール '{tool_gene.tool_name}' がエントリーをブロック "
                        f"(params={tool_gene.params})"
                    )
                    return True

            return False

        except Exception as e:
            # エラー時はエントリーを許可（フェイルセーフ）
            logger.warning(f"ツールフィルターエラー（フェイルセーフ適用）: {e}")
            return False

    def _prepare_current_features(self) -> pd.DataFrame:
        """
        現在のバーからML用特徴量を準備

        OHLCVデータを使用して、HybridPredictorが期待する形式の
        特徴量DataFrameを生成します。

        Returns:
            特徴量DataFrame
        """
        try:
            # 過去N本分のデータを取得（特徴量計算に必要な期間）
            lookback = 20  # デフォルトルックバック期間

            # データが十分にあるか確認
            data_length = len(self.data) if hasattr(self.data, "__len__") else 0
            actual_lookback = min(lookback, max(data_length - 1, 1))

            # OHLCVデータを抽出
            closes = list(self.data.Close[-actual_lookback:])
            highs = list(self.data.High[-actual_lookback:])
            lows = list(self.data.Low[-actual_lookback:])

            # 基本特徴量を計算
            features: Dict[str, Any] = {}

            # 価格関連
            current_close = closes[-1] if closes else 0.0
            features["close"] = current_close
            features["high"] = highs[-1] if highs else current_close
            features["low"] = lows[-1] if lows else current_close

            # リターン系特徴量
            if len(closes) >= 2:
                features["close_return_1"] = (
                    (closes[-1] - closes[-2]) / closes[-2] if closes[-2] != 0 else 0
                )
            else:
                features["close_return_1"] = 0

            if len(closes) >= 6:
                features["close_return_5"] = (
                    (closes[-1] - closes[-6]) / closes[-6] if closes[-6] != 0 else 0
                )
            else:
                features["close_return_5"] = 0

            # ローリング統計量
            if len(closes) >= 5:
                arr = np.array(closes[-5:])
                features["close_rolling_mean_5"] = float(np.mean(arr))
                features["close_rolling_std_5"] = float(np.std(arr))
            else:
                features["close_rolling_mean_5"] = current_close
                features["close_rolling_std_5"] = 0

            # 戦略構造特徴量（StrategyGeneから抽出）
            if self.gene:
                features["indicator_count"] = len(
                    [ind for ind in self.gene.indicators if ind.enabled]
                )
                features["condition_count"] = len(self.gene.entry_conditions or [])
                features["has_tpsl"] = (
                    1 if self.gene.tpsl_gene and self.gene.tpsl_gene.enabled else 0
                )

                if self.gene.tpsl_gene and self.gene.tpsl_gene.enabled:
                    features["take_profit_ratio"] = getattr(
                        self.gene.tpsl_gene, "take_profit_pct", 0.02
                    )
                    features["stop_loss_ratio"] = getattr(
                        self.gene.tpsl_gene, "stop_loss_pct", 0.01
                    )
                else:
                    features["take_profit_ratio"] = 0.02
                    features["stop_loss_ratio"] = 0.01

            return pd.DataFrame([features])

        except Exception as e:
            logger.error(f"特徴量準備エラー: {e}")
            # エラー時はデフォルト特徴量を返す
            return pd.DataFrame(
                [
                    {
                        "close": 0,
                        "close_return_1": 0,
                        "close_return_5": 0,
                        "indicator_count": 1,
                    }
                ]
            )

    # ===== 保留注文管理メソッド =====

    def _check_pending_order_fills(self) -> None:
        """
        保留注文の約定をチェック

        1分足データを使用して、各保留注文が約定したかを判定します。
        約定した場合は即座に取引を実行し、保留リストから削除します。
        """
        if not self._pending_orders or self._minute_data is None:
            return

        if self.position:
            # 既にポジションがある場合は保留注文をキャンセル
            self._pending_orders.clear()
            return

        # 現在のバーの時刻範囲を取得
        # Note: backtesting.py の data.index[-1] は現在のバーの開始時刻を指すと仮定
        # (Pandas の resample/date_range の標準的な挙動)
        current_bar_time = self.data.index[-1]
        bar_duration = self._get_bar_duration()

        if bar_duration is None:
            return

        # 期間: [OpenTime, OpenTime + Duration)
        # つまり、今確定したバーの期間データを取得する
        bar_start = current_bar_time
        bar_end = current_bar_time + bar_duration

        # 該当期間の1分足を抽出
        minute_bars = self.lower_tf_simulator.get_minute_data_for_bar(
            self._minute_data, bar_start, bar_end
        )

        if minute_bars.empty:
            return

        filled_orders = []

        for order in self._pending_orders:
            filled, fill_price = self.lower_tf_simulator.check_order_fill(
                order, minute_bars
            )

            if filled and fill_price is not None:
                # 約定実行
                self._execute_filled_order(order, fill_price)
                filled_orders.append(order)
                break  # 1バーで1注文のみ約定

        # 約定した注文を削除
        for order in filled_orders:
            self._pending_orders.remove(order)

    def _expire_pending_orders(self) -> None:
        """期限切れの保留注文を削除"""
        self._pending_orders = [
            order
            for order in self._pending_orders
            if not order.is_expired(self._current_bar_index)
        ]

    def _create_pending_order(
        self,
        direction: float,
        size: float,
        entry_params: dict,
        sl_price: float | None,
        tp_price: float | None,
        entry_gene: EntryGene,
    ) -> None:
        """
        保留注文を作成

        Args:
            direction: 取引方向 (1.0=Long, -1.0=Short)
            size: ポジションサイズ
            entry_params: エントリーパラメータ (limit, stop)
            sl_price: ストップロス価格
            tp_price: テイクプロフィット価格
            entry_gene: エントリー遺伝子
        """
        order = PendingOrder(
            order_type=entry_gene.entry_type,
            direction=direction,
            limit_price=entry_params.get("limit"),
            stop_price=entry_params.get("stop"),
            size=size,
            created_bar_index=self._current_bar_index,
            validity_bars=entry_gene.order_validity_bars,
            sl_price=sl_price,
            tp_price=tp_price,
        )
        self._pending_orders.append(order)

    def _execute_filled_order(self, order: PendingOrder, fill_price: float) -> None:
        """
        約定した注文を実行

        Args:
            order: 約定した保留注文
            fill_price: 約定価格
        """
        if order.is_long():
            self.buy(size=order.size)
        else:
            self.sell(size=order.size)

        # 内部状態を設定
        self._entry_price = fill_price
        self._sl_price = order.sl_price
        self._tp_price = order.tp_price
        self._position_direction = order.direction

    def _get_bar_duration(self):
        """
        現在のタイムフレームのバー期間を取得

        Returns:
            pd.Timedelta または None
        """
        timeframe_map = {
            "1m": pd.Timedelta(minutes=1),
            "5m": pd.Timedelta(minutes=5),
            "15m": pd.Timedelta(minutes=15),
            "30m": pd.Timedelta(minutes=30),
            "1h": pd.Timedelta(hours=1),
            "4h": pd.Timedelta(hours=4),
            "1d": pd.Timedelta(days=1),
        }
        return timeframe_map.get(self.base_timeframe)


