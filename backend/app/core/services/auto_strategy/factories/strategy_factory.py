"""
戦略ファクトリー

StrategyGeneから動的にbacktesting.py互換のStrategy継承クラスを生成します。
"""

import logging
from typing import Type, Tuple

from backtesting import Strategy
from ..models.strategy_gene import StrategyGene, IndicatorGene
from ..evaluators.condition_evaluator import ConditionEvaluator
from ..calculators.indicator_calculator import IndicatorCalculator
from ..calculators.tpsl_calculator import TPSLCalculator
from ..calculators.position_sizing_helper import PositionSizingHelper

logger = logging.getLogger(__name__)


class StrategyFactory:
    """
    戦略ファクトリー

    StrategyGeneから動的にStrategy継承クラスを生成します。
    """

    def __init__(self):
        """初期化"""
        self.condition_evaluator = ConditionEvaluator()
        self.indicator_calculator = IndicatorCalculator()
        self.tpsl_calculator = TPSLCalculator()
        self.position_sizing_helper = PositionSizingHelper()

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
        # 遺伝子の妥当性検証
        is_valid, errors = gene.validate()
        if not is_valid:
            raise ValueError(f"Invalid strategy gene: {', '.join(errors)}")

        # ファクトリー参照を保存
        factory = self

        # 動的クラス生成
        class GeneratedStrategy(Strategy):
            """動的生成された戦略クラス"""

            # backtesting.pyがパラメータを認識できるようにクラス変数として定義
            strategy_gene = None

            def _check_params(self, params):
                # backtesting.pyの厳格なパラメータチェックを回避するため、
                # 親の親である_Strategyのメソッドを模倣する。
                # これにより、クラス変数として定義されていないパラメータも受け入れられる。
                checked_params = dict(params)

                # _get_params()のロジックを再実装して静的解析エラーを回避
                defined_params = {}
                for key, value in type(self).__dict__.items():
                    if not key.startswith("_") and not callable(value):
                        defined_params[key] = value

                for key, value in defined_params.items():
                    checked_params.setdefault(key, value)
                return checked_params

            def __init__(self, broker=None, data=None, params=None):
                # paramsがNoneの場合は空辞書を設定
                if params is None:
                    params = {}

                # super().__init__は渡されたparamsを検証し、インスタンス変数として設定する
                super().__init__(broker, data, params)

                # self.strategy_geneはbacktesting.pyによって設定される
                # 渡されていればそれを使用し、なければ元のgeneを使用する
                current_gene = getattr(self, "strategy_gene", None)
                self.gene = current_gene if current_gene is not None else gene

                self.indicators = {}
                self.factory = factory  # ファクトリーへの参照

            def init(self):
                """指標の初期化"""
                try:
                    # 各指標を初期化
                    for i, indicator_gene in enumerate(gene.indicators):
                        if indicator_gene.enabled:
                            self._init_indicator(indicator_gene)
                        else:
                            pass

                except Exception as e:
                    logger.error(f"戦略初期化エラー: {e}", exc_info=True)
                    pass
                    raise

            def next(self):
                """売買ロジック"""
                try:
                    # リスク管理設定を取得
                    risk_management = self.gene.risk_management
                    stop_loss_pct = risk_management.get("stop_loss")
                    take_profit_pct = risk_management.get("take_profit")

                    # デバッグログ: 取引量設定の詳細
                    current_price = self.data.Close[-1]
                    current_equity = getattr(self, "equity", "N/A")

                    # ストップロスとテイクプロフィットの価格を計算
                    sl_price, tp_price = factory.tpsl_calculator.calculate_tpsl_prices(
                        current_price,
                        stop_loss_pct,
                        take_profit_pct,
                        risk_management,
                        gene,
                    )

                    # ロング・ショートエントリー条件チェック
                    long_entry_result = self._check_long_entry_conditions()
                    short_entry_result = self._check_short_entry_conditions()

                    if not self.position and (long_entry_result or short_entry_result):
                        # backtesting.pyのマージン問題を回避するため、非常に小さな固定サイズを使用
                        current_price = self.data.Close[-1]

                        # 現在の資産を取得
                        current_equity = getattr(self, "equity", 100000.0)
                        available_cash = getattr(self, "cash", current_equity)

                        # ポジション方向を決定
                        if long_entry_result and short_entry_result:
                            # 両方の条件が満たされた場合はロングを優先
                            position_direction = 1.0  # ロング
                        elif long_entry_result:
                            position_direction = 1.0  # ロング
                        elif short_entry_result:
                            position_direction = -1.0  # ショート
                        else:
                            position_direction = 1.0  # デフォルトはロング

                        # 動的ポジションサイズ計算
                        calculated_size = (
                            factory.position_sizing_helper.calculate_position_size(
                                gene, current_equity, current_price, self.data
                            )
                        )

                        # ポジション方向を適用
                        final_size = calculated_size * position_direction

                        # 利用可能現金で購入可能かチェック（ショートの場合も絶対値で計算）
                        required_cash = abs(final_size) * current_price
                        if required_cash <= available_cash * 0.99:
                            logger.info(
                                f"  必要資金: {required_cash:.2f}, 利用可能現金: {available_cash:.2f}"
                            )

                            # 計算されたサイズで注文を実行（SL/TPなし）
                            # ショートの場合は負のサイズでbuy()を呼び出す
                            self.buy(size=final_size)
                        else:
                            pass
                    # イグジット条件チェック
                    elif self.position:
                        exit_result = self._check_exit_conditions()
                        if exit_result:
                            self.position.close()

                except Exception as e:
                    logger.error(f"売買ロジックエラー: {e}", exc_info=True)
                    pass

            def _init_indicator(self, indicator_gene: IndicatorGene):
                """単一指標の初期化（統合版）"""
                try:
                    logger.info(
                        f"指標初期化開始: {indicator_gene.type}, パラメータ: {indicator_gene.parameters}"
                    )
                    # 指標計算器を使用して初期化
                    factory.indicator_calculator.init_indicator(indicator_gene, self)
                    logger.info(f"指標初期化完了: {indicator_gene.type}")

                except Exception as e:
                    logger.error(
                        f"指標初期化エラー {indicator_gene.type}: {e}", exc_info=True
                    )
                    # エラーを再発生させて上位で適切に処理
                    raise

            def _check_entry_conditions(self) -> bool:
                """エントリー条件をチェック（後方互換性のため保持）"""
                return factory.condition_evaluator.evaluate_conditions(
                    self.gene.entry_conditions, self
                )

            def _check_long_entry_conditions(self) -> bool:
                """ロングエントリー条件をチェック"""
                long_conditions = self.gene.get_effective_long_conditions()
                if not long_conditions:
                    # 条件が空の場合は、戦略設定に依存
                    # 後方互換性のため、entry_conditionsがある場合は有効とする
                    return bool(self.gene.entry_conditions)
                return factory.condition_evaluator.evaluate_conditions(
                    long_conditions, self
                )

            def _check_short_entry_conditions(self) -> bool:
                """ショートエントリー条件をチェック"""
                short_conditions = self.gene.get_effective_short_conditions()
                if not short_conditions:
                    # ショート条件が明示的に設定されていない場合は無効
                    # ただし、ロング・ショート分離がされていない場合は後方互換性を考慮
                    if (
                        not self.gene.has_long_short_separation()
                        and self.gene.entry_conditions
                    ):
                        return bool(self.gene.entry_conditions)
                    return False
                return factory.condition_evaluator.evaluate_conditions(
                    short_conditions, self
                )

            def _check_exit_conditions(self) -> bool:
                """イグジット条件をチェック（統合版）"""
                return factory.condition_evaluator.evaluate_conditions(
                    self.gene.exit_conditions, self
                )

        # クラス名を設定
        GeneratedStrategy.__name__ = f"GeneratedStrategy_{gene.id}"
        GeneratedStrategy.__qualname__ = GeneratedStrategy.__name__

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
            # logger.error(f"遺伝子検証エラー: {e}")
            return False, [f"検証エラー: {str(e)}"]
