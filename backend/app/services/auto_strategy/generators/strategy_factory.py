"""
戦略ファクトリー

StrategyGeneから動的にbacktesting.py互換のStrategy継承クラスを生成します。
"""

from ..models.strategy_models import Condition
import logging
from typing import List, Tuple, Type, Union, cast
import os
from ..models.strategy_models import Condition
import logging
from typing import List, Tuple, Type, Union, cast

from backtesting import Strategy

from ..services.indicator_service import IndicatorCalculator
from ..services.position_sizing_service import PositionSizingService
from ..services.tpsl_service import TPSLService
from ..core.condition_evaluator import ConditionEvaluator
from ..models.strategy_models import IndicatorGene, StrategyGene, ConditionGroup
from ..utils.decorators import auto_strategy_operation

logger = logging.getLogger(__name__)


def _debug_log(message: str, level: str = "debug", force: bool = False):
    """
    条件付きデバッグログ出力ヘルパー

    Args:
        message: ログメッセージ
        level: ログレベル ('debug', 'info', 'warning')
        force: 強制出力する場合True
    """
    # 環境変数でデバッグログを制御
    debug_enabled = force or (
        hasattr(logging, '_debug_strategy_factory') or
        os.getenv('AUTO_STRATEGY_DEBUG', '').lower() in ('true', '1')
    )

    if debug_enabled:
        log_func = getattr(logger, level, logger.debug)
        log_func(f"🏭 [StrategyFactory] {message}")


# 環境変数でデバッグフラグを設定
if os.getenv('AUTO_STRATEGY_DEBUG', '').lower() in ('true', '1'):
    logging._debug_strategy_factory = True


class StrategyFactory:
    """
    戦略ファクトリー

    StrategyGeneから動的にStrategy継承クラスを生成します。
    """

    def __init__(self):
        """初期化"""
        self.condition_evaluator = ConditionEvaluator()
        self.indicator_calculator = IndicatorCalculator()
        self.tpsl_service = TPSLService()
        self.position_sizing_service = PositionSizingService()

    def create_strategy_class(self, gene: StrategyGene) -> Type[Strategy]:
        """
        遺伝子から動的にStrategy継承クラスを生成

        Args:
            gene: 戦略遺伝子

        Returns:
            backtesting.py互換のStrategy継承クラス

        Raises:
            ValueError: 遺伝子が無効な場合
        """
        _debug_log(f"戦略クラス作成開始: 指標数={len(gene.indicators)}")
        _debug_log(f"戦略遺伝子詳細: {[ind.type for ind in gene.indicators]}")

        # 遺伝子の妥当性検証
        is_valid, errors = gene.validate()
        if not is_valid:
            raise ValueError(f"Invalid strategy gene: {', '.join(errors)}")

        _debug_log("戦略遺伝子検証成功")

        # ファクトリー参照を保存
        factory = self

        _debug_log("動的クラス生成開始")

        # 動的クラス生成
        class GeneratedStrategy(Strategy):
            """動的生成された戦略クラス"""

            # backtesting.pyがパラメータを認識できるようにクラス変数として定義
            strategy_gene = gene  # デフォルト値として元のgeneを設定

            def __init__(self, broker=None, data=None, params=None):
                _debug_log(
                    f"戦略初期化: broker={broker is not None}, data={data is not None}"
                )

                # paramsがNoneの場合は空辞書を設定
                if params is None:
                    params = {}

                # super().__init__は渡されたparamsを検証し、インスタンス変数として設定する
                super().__init__(broker, data, params)

                # 戦略遺伝子を設定（backtesting.pyからパラメータとして渡される場合もある）
                if params and "strategy_gene" in params:
                    self.strategy_gene = params["strategy_gene"]
                    self.gene = params["strategy_gene"]
                    _debug_log(
                        f"戦略遺伝子をparamsから設定: {self.strategy_gene.indicators[0].type if self.strategy_gene.indicators else 'なし'}"
                    )
                else:
                    # デフォルトとして元のgeneを使用
                    self.strategy_gene = gene
                    self.gene = gene
                    _debug_log(
                        f"戦略遺伝子をデフォルトから設定: {gene.indicators[0].type if gene.indicators else 'なし'}"
                    )

                self.indicators = {}
                self.factory = factory  # ファクトリーへの参照

            def init(self):
                """指標の初期化"""
                _debug_log("戦略初期化開始")
                _debug_log(f"指標数: {len(self.strategy_gene.indicators)}")

                try:
                    # 各指標を初期化
                    enabled_indicators = [ind for ind in self.strategy_gene.indicators if ind.enabled]
                    _debug_log(f"有効な指標数: {len(enabled_indicators)}")

                    for indicator_gene in enabled_indicators:
                        self._init_indicator(indicator_gene)

                    _debug_log("戦略初期化完了")
                except Exception as e:
                    logger.error(f"戦略初期化エラー: {e}", exc_info=True)
                    raise

            def _init_indicator(self, indicator_gene: IndicatorGene):
                """単一指標の初期化（統合版）"""
                try:
                    # 指標計算器を使用して初期化
                    try:
                        factory.indicator_calculator.init_indicator(
                            indicator_gene, self
                        )
                        return
                    except Exception as e:
                        logger.warning(f"指標初期化失敗 {indicator_gene.type}: {e}")

                        # フォールバック: SMAの最小構成でリカバーを試みる
                        if indicator_gene.type not in ("SMA", "RSI"):
                            from ..models.strategy_models import IndicatorGene as IG

                            period = indicator_gene.parameters.get("period", 14)
                            if period <= 0:
                                period = 14

                            try:
                                fb = IG(
                                    type="SMA", parameters={"period": period}, enabled=True
                                )
                                factory.indicator_calculator.init_indicator(fb, self)
                                _debug_log(f"フォールバック適用: {indicator_gene.type} -> SMA({period})")
                                return
                            except Exception as fb_e:
                                logger.warning(f"フォールバック失敗: {fb_e}")
                        else:
                            # 最後の手段: RSI(14)
                            try:
                                from ..models.strategy_models import IndicatorGene as IG
                                fb2 = IG(
                                    type="RSI", parameters={"period": 14}, enabled=True
                                )
                                factory.indicator_calculator.init_indicator(fb2, self)
                                _debug_log("最終フォールバック適用: RSI(14)")
                                return
                            except Exception as fb2_e:
                                logger.error(f"最終フォールバック失敗: {fb2_e}")

                except Exception as e:
                    logger.error(
                        f"指標初期化エラー {indicator_gene.type}: {e}", exc_info=True
                    )
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
                        return factory.condition_evaluator.evaluate_conditions(
                            entry_conditions, self
                        )
                    return False

                return factory.condition_evaluator.evaluate_conditions(
                    long_conditions, self
                )

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
                        return factory.condition_evaluator.evaluate_conditions(
                            entry_conditions, self
                        )
                    return False

                return factory.condition_evaluator.evaluate_conditions(
                    short_conditions, self
                )

            def _check_exit_conditions(self) -> bool:
                """イグジット条件をチェック（統合版）"""
                # TP/SL遺伝子が存在し有効な場合はイグジット条件をスキップ
                if self.gene.tpsl_gene and self.gene.tpsl_gene.enabled:
                    return False

                # 通常のイグジット条件をチェック
                exit_conditions = cast(
                    List[Union[Condition, ConditionGroup]], self.gene.exit_conditions
                )
                if not exit_conditions:
                    return False

                return factory.condition_evaluator.evaluate_conditions(
                    exit_conditions, self
                )

            def _calculate_position_size(self) -> float:
                """ポジションサイズを計算（PositionSizingService使用）"""
                try:
                    # PositionSizingGeneが有効な場合
                    if (
                        hasattr(self, "gene")
                        and self.gene.position_sizing_gene
                        and self.gene.position_sizing_gene.enabled
                    ):
                        # 現在の市場データ（該当するものがなければデフォルト値を使用）
                        current_price = self.data.Close[-1] if hasattr(self, 'data') and len(self.data.Close) > 0 else 50000.0
                        account_balance = getattr(self, 'equity', 100000.0)  # デフォルト口座残高

                        # PositionSizingServiceを使用して計算
                        result = factory.position_sizing_service.calculate_position_size(
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
                    # デバッグ時はスタックトレースも出力
                    if logging.getLogger().isEnabledFor(logging.DEBUG):
                        import traceback
                        traceback.print_exc()

        # クラス名を設定
        short_id = str(gene.id).split("-")[0] if gene.id else "Unknown"
        GeneratedStrategy.__name__ = f"GS_{short_id}"
        GeneratedStrategy.__qualname__ = GeneratedStrategy.__name__

        _debug_log(f"戦略クラス作成完了: {GeneratedStrategy.__name__}")
        logger.info(f"戦略クラス生成成功: {GeneratedStrategy.__name__}")

        return GeneratedStrategy


    def validate_gene(self, gene: StrategyGene) -> Tuple[bool, list]:
        """
        戦略遺伝子の妥当性を検証

        Args:
            gene: 検証する戦略遺伝子

        Returns:
            (is_valid, error_messages)
        """
        try:
            return gene.validate()
        except Exception as e:
            logger.error(f"遺伝子検証エラー: {e}")
            return False, [f"検証エラー: {str(e)}"]
