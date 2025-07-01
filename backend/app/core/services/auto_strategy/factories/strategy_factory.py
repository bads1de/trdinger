"""
戦略ファクトリー

StrategyGeneから動的にbacktesting.py互換のStrategy継承クラスを生成します。
責任を分離し、各機能を専用モジュールに委譲します。
"""

from typing import Type, Tuple
import logging
from backtesting import Strategy

from ..models.strategy_gene import StrategyGene, IndicatorGene
from .indicator_initializer import IndicatorInitializer
from .condition_evaluator import ConditionEvaluator
from .data_converter import DataConverter


logger = logging.getLogger(__name__)


class StrategyFactory:
    """
    戦略ファクトリー

    StrategyGeneから動的にStrategy継承クラスを生成し、
    既存のTALibAdapterシステムと統合します。
    """

    def __init__(self):
        """初期化"""
        # 分離されたコンポーネント
        self.indicator_initializer = IndicatorInitializer()
        self.condition_evaluator = ConditionEvaluator()
        self.data_converter = DataConverter()

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
                    print(f"🔧 戦略初期化開始: {len(gene.indicators)}個の指標")

                    # 各指標を初期化
                    for i, indicator_gene in enumerate(gene.indicators):
                        print(
                            f"  指標 {i+1}: {indicator_gene.type}, enabled={indicator_gene.enabled}"
                        )
                        if indicator_gene.enabled:
                            print(f"    → 初期化実行中...")
                            self._init_indicator(indicator_gene)
                            print(f"    → 初期化完了")
                        else:
                            print(f"    → スキップ（無効）")

                    print(f"🔧 戦略初期化完了: {len(self.indicators)}個の指標")
                    print(f"  登録された指標: {list(self.indicators.keys())}")
                    logger.info(f"戦略初期化完了: {len(self.indicators)}個の指標")

                except Exception as e:
                    print(f"❌ 戦略初期化エラー: {e}")
                    logger.error(f"戦略初期化エラー: {e}")
                    import traceback

                    traceback.print_exc()
                    raise

            def next(self):
                """売買ロジック"""
                try:
                    # エントリー条件チェック
                    if not self.position and self._check_entry_conditions():
                        self.buy()

                    # イグジット条件チェック
                    elif self.position and self._check_exit_conditions():
                        self.sell()

                except Exception as e:
                    logger.error(f"売買ロジックエラー: {e}")
                    # エラーが発生してもバックテストを継続

            def _init_indicator(self, indicator_gene: IndicatorGene):
                """単一指標の初期化"""
                # IndicatorInitializerに委譲
                indicator_name = (
                    self.factory.indicator_initializer.initialize_indicator(
                        indicator_gene, self.data, self
                    )
                )
                if indicator_name:
                    pass
                else:
                    logger.warning(f"指標初期化失敗: {indicator_gene.type}")

            def _convert_to_series(self, bt_array):
                """backtesting.pyの_ArrayをPandas Seriesに変換"""
                return self.factory.data_converter.convert_to_series(bt_array)

            def _check_entry_conditions(self) -> bool:
                """エントリー条件をチェック"""
                return self.factory.condition_evaluator.check_entry_conditions(
                    gene.entry_conditions, self
                )

            def _check_exit_conditions(self) -> bool:
                """イグジット条件をチェック"""
                return self.factory.condition_evaluator.check_exit_conditions(
                    gene.exit_conditions, self
                )

            def _evaluate_condition(self, condition):
                """単一条件を評価"""
                return self.factory.condition_evaluator.evaluate_condition(
                    condition, self
                )

            def _get_condition_value(self, operand):
                """条件のオペランドから値を取得"""
                return self.factory.condition_evaluator.get_condition_value(
                    operand, self
                )

        # クラス名を設定
        GeneratedStrategy.__name__ = f"GeneratedStrategy_{gene.id}"
        GeneratedStrategy.__qualname__ = GeneratedStrategy.__name__

        return GeneratedStrategy

    def validate_gene(self, gene: StrategyGene) -> Tuple[bool, list]:
        """
        遺伝子の妥当性を詳細に検証

        Args:
            gene: 検証する戦略遺伝子

        Returns:
            (is_valid, error_messages)
        """
        errors = []

        # 基本的な妥当性チェック
        is_valid, basic_errors = gene.validate()
        errors.extend(basic_errors)

        # 指標の対応状況チェック
        for indicator in gene.indicators:
            if (
                indicator.enabled
                and not self.indicator_initializer.is_supported_indicator(
                    indicator.type
                )
            ):
                errors.append(f"未対応の指標: {indicator.type}")

        # 演算子の対応状況チェック
        for condition in gene.entry_conditions + gene.exit_conditions:
            if not self.condition_evaluator.is_supported_operator(condition.operator):
                errors.append(f"未対応の演算子: {condition.operator}")

        return len(errors) == 0, errors
