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

from ..core.condition_evaluator import ConditionEvaluator
from ..genes.entry import EntryGene
from ..config.constants import EntryType
from ..genes.conditions import StateTracker
from ..genes import (
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
from .order_manager import OrderManager

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

        # 注文管理マネージャーの初期化
        self.order_manager = OrderManager(self, self.lower_tf_simulator)

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
        # ML フィルター閾値（エントリー許可のための方向スコア差分）
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

    def _get_effective_tpsl_gene(self, direction: float) -> Union[None, object]:
        """旧互換用: 有効なTPSL遺伝子を取得"""
        return self._get_effective_sub_gene(direction, "tpsl")

    def _get_effective_entry_gene(self, direction: float) -> Union[None, EntryGene]:
        """旧互換用: 有効なエントリー遺伝子を取得"""
        return self._get_effective_sub_gene(direction, "entry")

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
                except Exception as e:
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
            if self.ml_predictor:
                self._precompute_ml_features()

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

        except Exception as e:
            logger.error(f"戦略初期化エラー: {e}", exc_info=True)
            raise

    def _precompute_ml_features(self):
        """ML予測に必要な全期間の特徴量を一括計算してキャッシュする"""
        try:
            from ..core.hybrid_feature_adapter import HybridFeatureAdapter

            # アダプターの初期化
            self.feature_adapter = HybridFeatureAdapter()

            # 全期間のデータを準備
            full_ohlcv = self.data.df.copy()
            full_ohlcv.columns = [c.lower() for c in full_ohlcv.columns]

            # 一括変換実行
            self._precomputed_features = self.feature_adapter.gene_to_features(
                gene=self.gene, ohlcv_data=full_ohlcv, apply_preprocessing=False
            )
            # logger.info(f"ML特徴量の一括計算完了: {len(self._precomputed_features)}行")
        except Exception as e:
            logger.error(f"ML特徴量事前計算エラー: {e}")
            self._precomputed_features = None

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
        tpsl_method = active_tpsl_gene.method

        if tpsl_method in (
            TPSLMethod.VOLATILITY_BASED,
            TPSLMethod.ADAPTIVE,
            TPSLMethod.STATISTICAL,
        ):
            atr_period = getattr(active_tpsl_gene, "atr_period", 14)
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
            tpsl_gene=active_tpsl_gene,
            position_direction=direction,
            market_data=market_data,
        )

    def next(self):
        """各バーでの戦略実行"""
        try:
            self._current_bar_index += 1

            # 1. 保留注文とステートフルトリガーの処理
            self.order_manager.check_pending_order_fills(
                self._minute_data, self.data.index[-1], self._current_bar_index
            )
            self.order_manager.expire_pending_orders(self._current_bar_index)
            self._process_stateful_triggers()

            # 2. 既存ポジションの悲観的決済チェック
            if self.position and self._sl_price is not None:
                if self._check_pessimistic_exit():
                    return

            # 3. 新規エントリー判定（ノーポジション時）
            if not self.position:
                if self._tools_block_entry():
                    return

                # 方向の決定（優先順位: 通常ロング > 通常ショート > ステートフル）
                direction = 0.0
                if self._check_entry_conditions(1.0):
                    direction = 1.0
                elif self._check_entry_conditions(-1.0):
                    direction = -1.0
                else:
                    stateful_dir = self._get_stateful_entry_direction()
                    if stateful_dir is not None:
                        direction = stateful_dir

                if direction == 0.0:
                    return

                # 4. MLフィルター判定
                if self.ml_predictor and not self._ml_allows_entry(direction):
                    return

                # 5. TP/SLおよびエントリーパラメータの計算
                current_price = self.data.Close[-1]
                sl_price, tp_price = self._calculate_effective_tpsl_prices(
                    direction, current_price
                )

                entry_gene = self._get_effective_entry_gene(direction)
                entry_params = self.entry_executor.calculate_entry_params(
                    entry_gene, current_price, direction
                )
                position_size = self._calculate_position_size()

                # 6. 注文実行
                is_market = (
                    entry_gene is None
                    or not entry_gene.enabled
                    or entry_gene.entry_type == EntryType.MARKET
                )
                if is_market:
                    if direction > 0:
                        self.buy(size=position_size)
                    else:
                        self.sell(size=position_size)

                    self._entry_price, self._sl_price, self._tp_price = (
                        current_price,
                        sl_price,
                        tp_price,
                    )
                    self._position_direction = direction
                else:
                    self.order_manager.create_pending_order(
                        direction=direction,
                        size=position_size,
                        entry_params=entry_params,
                        sl_price=sl_price,
                        tp_price=tp_price,
                        entry_gene=entry_gene,
                        current_bar_index=self._current_bar_index,
                    )

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
                # logger.debug(
                #     f"トレーリングTP利益確保ライン更新 (Long): {self._trailing_tp_sl:.2f}"
                # )

        # ショートポジションの場合
        elif self._position_direction < 0:
            new_trailing_sl = current_close * (1.0 + trailing_step)
            if new_trailing_sl < self._trailing_tp_sl:
                self._trailing_tp_sl = new_trailing_sl
                # logger.debug(
                #     f"トレーリングTP利益確保ライン更新 (Short): {self._trailing_tp_sl:.2f}"
                # )

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
                # logger.debug(f"トレーリングSL更新 (Long): {self._sl_price:.2f}")

        # ショートポジションの場合: 終値ベースで新SLを計算し、現在SLより低ければ更新
        elif self._position_direction < 0:
            new_sl = current_close * (1.0 + trailing_step)
            if new_sl < self._sl_price:
                self._sl_price = new_sl
                # logger.debug(f"トレーリングSL更新 (Short): {self._sl_price:.2f}")

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
            # 1. 事前計算済みの特徴量から現在の行を取得
            current_time = self.data.index[-1]
            features = None

            if (
                hasattr(self, "_precomputed_features")
                and self._precomputed_features is not None
            ):
                if current_time in self._precomputed_features.index:
                    # インデックスで高速検索
                    features = self._precomputed_features.loc[[current_time]]

            # 2. キャッシュがない場合はフォールバック（低速）
            if features is None:
                features = self._prepare_current_features()

            # 3. ML予測を実行
            prediction = self.ml_predictor.predict(features)

            # ダマシ予測モデルの判定
            # is_valid: エントリーが有効である確率 (0.0-1.0)
            # 閾値以上であればエントリーを許可
            is_valid = prediction.get("is_valid", 0.5)
            allowed = is_valid >= self.ml_filter_threshold

            if not allowed:
                # logger.debug(
                #     f"MLフィルター拒否: direction={direction}, "
                #     f"is_valid={is_valid:.3f}, threshold={self.ml_filter_threshold}"
                # )
                pass

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

    def _prepare_current_features(self) -> pd.DataFrame:
        """
        現在のバーからML用特徴量を準備
        HybridFeatureAdapterに委譲して一貫性を確保します。
        """
        try:
            from ..core.hybrid_feature_adapter import HybridFeatureAdapter

            # アダプターの初期化（まだ存在しない場合）
            if not hasattr(self, "feature_adapter"):
                self.feature_adapter = HybridFeatureAdapter()

            # 現在のバーのOHLCVデータを取得
            # backtesting.pyのdataオブジェクトをDataFrameに変換（直近のみ）
            lookback = 30  # 特徴量計算に必要な最低限のルックバック
            data_len = len(self.data)
            actual_lookback = min(lookback, data_len)

            # 効率のため必要な分だけスライス
            subset = self.data.df.iloc[-actual_lookback:].copy()

            # 既存のカラム名を小文字に統一（アダプタの期待に合わせる）
            subset.columns = [c.lower() for c in subset.columns]

            # アダプタを使用して特徴量変換
            features_df = self.feature_adapter.gene_to_features(
                gene=self.gene,
                ohlcv_data=subset,
                apply_preprocessing=False,  # 推論時は基本的なクリーニングのみ
            )

            # 直近の1行のみを返す
            return features_df.iloc[[-1]]

        except Exception as e:
            logger.error(f"特徴量準備エラー (Adapter使用): {e}")
            # フォールバック（最小限の構造を持つDataFrame）
            return pd.DataFrame([{"close": self.data.Close[-1], "indicator_count": 1}])
