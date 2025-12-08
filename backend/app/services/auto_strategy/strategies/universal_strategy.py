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
from ..models.strategy_models import (
    Condition,
    ConditionGroup,
    IndicatorGene,
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
        self.indicator_calculator = IndicatorCalculator()
        self.tpsl_service = TPSLService()
        self.position_sizing_service = PositionSizingService()

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

        self.indicators = {}

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
        """ポジションサイズを計算"""
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
                account_balance = getattr(
                    self, "equity", 100000.0
                )  # デフォルト口座残高

                # PositionSizingServiceを使用して計算
                result = self.position_sizing_service.calculate_position_size(
                    gene=self.gene.position_sizing_gene,
                    account_balance=account_balance,
                    current_price=current_price,
                )

                # 結果を返却（安全範囲に制限）
                position_size = result.position_size
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
            # ポジションがない場合のエントリー判定
            if not self.position:
                long_signal = self._check_long_entry_conditions()
                short_signal = self._check_short_entry_conditions()

                if long_signal or short_signal:
                    # ポジションサイズを決定
                    position_size = self._calculate_position_size()
                    current_price = self.data.Close[-1]

                    # TP/SL価格を計算
                    sl_price = None
                    tp_price = None

                    if self.gene.tpsl_gene and self.gene.tpsl_gene.enabled:
                        if long_signal:
                            sl_price = current_price * (
                                1 - self.gene.tpsl_gene.stop_loss_pct
                            )
                            tp_price = current_price * (
                                1 + self.gene.tpsl_gene.take_profit_pct
                            )
                        elif short_signal:
                            sl_price = current_price * (
                                1 + self.gene.tpsl_gene.stop_loss_pct
                            )
                            tp_price = current_price * (
                                1 - self.gene.tpsl_gene.take_profit_pct
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
