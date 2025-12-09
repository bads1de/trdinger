"""
汎用自動生成戦略クラス

GAから生成されたStrategyGeneを受け取り、その定義に基づいて動的に振る舞う
backtesting.py互換の戦略クラスです。
Pickle化可能にするため、filesのトップレベルで定義されています。
"""

import logging
from typing import List, Union, cast

from backtesting import Strategy

from ..core.condition_evaluator import ConditionEvaluator
from ..models.stateful_condition import StateTracker
from ..models.strategy_models import (
    Condition,
    ConditionGroup,
    IndicatorGene,
    TPSLMethod,
)
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

    def __init__(self, broker, data, params):
        """
        初期化

        Args:
            broker: Brokerインスタンス
            data: Dataインスタンス
            params: パラメータ辞書（'strategy_gene'を含む必要がある）
        """
        # サービスの初期化（クラスレベルで持つべきだが、状態を持たないのでインスタンスごとでも可）
        # 注意: multiprocessing時はここで初期化することが重要
        self.condition_evaluator = ConditionEvaluator()
        self.tpsl_service = TPSLService()
        self.position_sizing_service = PositionSizingService()
        self.state_tracker = StateTracker()  # ステートフル条件用
        self._current_bar_index = 0  # バーインデックストラッカー

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
        """ポジションサイズを計算（キャッシュ付き高速版）"""
        try:
            # PositionSizingGeneが有効な場合
            if (
                self.gene.position_sizing_gene
                and self.gene.position_sizing_gene.enabled
            ):
                # キャッシュチェック：同一遺伝子設定では再計算しない
                if hasattr(self, "_cached_position_size"):
                    return self._cached_position_size

                # 現在の市場データ（該当するものがなければデフォルト値を使用）
                current_price = (
                    self.data.Close[-1]
                    if hasattr(self, "data") and len(self.data.Close) > 0
                    else 50000.0
                )
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

                # 結果を安全範囲に制限してキャッシュ
                self._cached_position_size = max(0.001, min(0.2, float(position_size)))
                return self._cached_position_size
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

            # ステートフル条件のトリガーをチェック・記録
            self._process_stateful_triggers()

            # ポジションがない場合のエントリー判定
            if not self.position:
                long_signal = self._check_long_entry_conditions()
                short_signal = self._check_short_entry_conditions()

                # ステートフル条件も評価
                stateful_signal = self._check_stateful_conditions()

                # 通常条件またはステートフル条件いずれかでエントリー
                if long_signal or short_signal or stateful_signal:
                    # ポジションサイズを決定
                    position_size = self._calculate_position_size()
                    current_price = self.data.Close[-1]

                    # エントリー方向を決定
                    direction = 0.0
                    if long_signal:
                        direction = 1.0
                    elif short_signal:
                        direction = -1.0
                    # Note: stateful_signalのみの場合は、方向が不明確なためスキップされる
                    # (将来的にStatefulConditionに方向性を持たせる修正が必要)

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
                            if (
                                tpsl_method
                                in (
                                    TPSLMethod.VOLATILITY_BASED,
                                    TPSLMethod.ADAPTIVE,
                                    TPSLMethod.STATISTICAL,
                                )
                                and len(self.data) > 30
                            ):
                                # 過去30本のデータを取得（ATR計算用）
                                # Note: パフォーマンス最適化のため、必要な期間のみスライス
                                highs = self.data.High[-30:]
                                lows = self.data.Low[-30:]
                                closes = self.data.Close[-30:]

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

                    # 取引実行
                    if long_signal:
                        if sl_price and tp_price:
                            self.buy(size=position_size, sl=sl_price, tp=tp_price)
                        else:
                            self.buy(size=position_size)

                    elif short_signal:
                        if sl_price and tp_price:
                            self.sell(size=position_size, sl=sl_price, tp=tp_price)
                        else:
                            self.sell(size=position_size)

            # ポジションがある場合のイグジット判定
            elif self.position and self._check_exit_conditions():
                self.position.close()

        except Exception as e:
            logger.error(f"戦略実行エラー: {e}")

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
