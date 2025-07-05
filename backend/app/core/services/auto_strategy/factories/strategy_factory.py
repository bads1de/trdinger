"""
戦略ファクトリー

StrategyGeneから動的にbacktesting.py互換のStrategy継承クラスを生成します。
"""

import logging
import numpy as np
from typing import Type, Dict, Any, List, Union, Tuple

from backtesting import Strategy
from app.core.services.indicators import TechnicalIndicatorService
from ..models.strategy_gene import StrategyGene, IndicatorGene, Condition

logger = logging.getLogger(__name__)


class StrategyFactory:
    """
    戦略ファクトリー

    StrategyGeneから動的にStrategy継承クラスを生成します。
    """

    def __init__(self):
        """初期化"""
        self.technical_indicator_service = TechnicalIndicatorService()

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
                    logger.debug(f"戦略初期化開始: {len(gene.indicators)}個の指標")

                    # 各指標を初期化
                    for i, indicator_gene in enumerate(gene.indicators):
                        logger.debug(
                            f"  指標 {i+1}: {indicator_gene.type}, enabled={indicator_gene.enabled}"
                        )
                        if indicator_gene.enabled:
                            logger.debug("    → 初期化実行中...")
                            self._init_indicator(indicator_gene)
                            logger.debug("    → 初期化完了")
                        else:
                            logger.debug("    → スキップ（無効）")

                    logger.info(
                        f"戦略初期化完了: {len(self.indicators)}個の指標. "
                        f"登録された指標: {list(self.indicators.keys())}"
                    )

                except Exception as e:
                    logger.error(f"戦略初期化エラー: {e}", exc_info=True)
                    raise

            def next(self):
                """売買ロジック"""
                try:
                    # リスク管理設定を取得
                    risk_management = self.gene.risk_management
                    position_size = risk_management.get(
                        "position_size", 0.1
                    )  # デフォルトは10%
                    stop_loss_pct = risk_management.get("stop_loss")
                    take_profit_pct = risk_management.get("take_profit")

                    # ストップロスとテイクプロフィットの価格を計算
                    sl_price = (
                        self.data.Close[-1] * (1 - stop_loss_pct)
                        if stop_loss_pct
                        else None
                    )
                    tp_price = (
                        self.data.Close[-1] * (1 + take_profit_pct)
                        if take_profit_pct
                        else None
                    )

                    # エントリー条件チェック
                    if not self.position and self._check_entry_conditions():
                        self.buy(size=position_size, sl=sl_price, tp=tp_price)

                    # イグジット条件チェック
                    elif self.position and self._check_exit_conditions():
                        self.position.close()

                except Exception as e:
                    logger.error(f"売買ロジックエラー: {e}", exc_info=True)

            def _init_indicator(self, indicator_gene: IndicatorGene):
                """単一指標の初期化（統合版）"""
                try:
                    # 指標計算を直接実行
                    result = self.factory._calculate_indicator(
                        indicator_gene.type, indicator_gene.parameters, self.data
                    )

                    if result is not None:
                        # 指標をstrategy.I()で登録
                        if isinstance(result, tuple):
                            # 複数の出力がある指標（MACD等）
                            for i, output in enumerate(result):
                                indicator_name = f"{indicator_gene.type}_{i}"
                                setattr(
                                    self, indicator_name, self.I(lambda x=output: x)
                                )
                        else:
                            # 単一出力の指標
                            setattr(
                                self, indicator_gene.type, self.I(lambda x=result: x)
                            )

                        logger.debug(f"指標初期化成功: {indicator_gene.type}")
                    else:
                        logger.warning(f"指標計算失敗: {indicator_gene.type}")

                except Exception as e:
                    logger.error(f"指標初期化エラー {indicator_gene.type}: {e}")

            def _check_entry_conditions(self) -> bool:
                """エントリー条件をチェック（統合版）"""
                return self.factory._evaluate_conditions(
                    self.gene.entry_conditions, self
                )

            def _check_exit_conditions(self) -> bool:
                """イグジット条件をチェック（統合版）"""
                return self.factory._evaluate_conditions(
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
            logger.error(f"遺伝子検証エラー: {e}")
            return False, [f"検証エラー: {str(e)}"]

    def _calculate_indicator(
        self, indicator_type: str, parameters: Dict[str, Any], data
    ) -> Union[np.ndarray, Tuple[np.ndarray, ...], None]:
        """
        指標計算（統合版）

        Args:
            indicator_type: 指標タイプ
            parameters: パラメータ
            data: backtesting.pyのデータオブジェクト

        Returns:
            計算結果（numpy配列）
        """
        try:
            # backtesting.pyのデータオブジェクトをDataFrameに変換
            df = data.df

            # TechnicalIndicatorServiceを使用して計算
            result = self.technical_indicator_service.calculate_indicator(
                df, indicator_type, parameters
            )

            return result

        except Exception as e:
            logger.error(f"指標計算エラー {indicator_type}: {e}")
            return None

    def _evaluate_conditions(
        self, conditions: List[Condition], strategy_instance
    ) -> bool:
        """
        条件評価（統合版）

        Args:
            conditions: 評価する条件リスト
            strategy_instance: 戦略インスタンス

        Returns:
            全条件がTrueかどうか
        """
        try:
            if not conditions:
                return True

            for condition in conditions:
                if not self._evaluate_single_condition(condition, strategy_instance):
                    return False
            return True

        except Exception as e:
            logger.error(f"条件評価エラー: {e}")
            return False

    def _evaluate_single_condition(
        self, condition: Condition, strategy_instance
    ) -> bool:
        """
        単一条件の評価（統合版）

        Args:
            condition: 評価する条件
            strategy_instance: 戦略インスタンス

        Returns:
            条件の評価結果
        """
        try:
            # 左オペランドの値を取得
            left_value = self._get_condition_value(
                condition.left_operand, strategy_instance
            )
            right_value = self._get_condition_value(
                condition.right_operand, strategy_instance
            )

            # 両方の値が数値であることを確認
            if not isinstance(left_value, (int, float)) or not isinstance(
                right_value, (int, float)
            ):
                logger.warning(
                    f"比較できない値です: left={left_value}({type(left_value)}), "
                    f"right={right_value}({type(right_value)})"
                )
                return False

            # 条件評価
            if condition.operator == ">":
                return left_value > right_value
            elif condition.operator == "<":
                return left_value < right_value
            elif condition.operator == ">=":
                return left_value >= right_value
            elif condition.operator == "<=":
                return left_value <= right_value
            elif condition.operator == "==":
                return bool(np.isclose(left_value, right_value))
            elif condition.operator == "!=":
                return not bool(np.isclose(left_value, right_value))
            else:
                logger.warning(f"未対応の演算子: {condition.operator}")
                return False

        except Exception as e:
            logger.error(f"条件評価エラー: {e}")
            return False

    def _get_condition_value(
        self, operand: Union[Dict[str, Any], str, int, float], strategy_instance
    ) -> float:
        """
        条件オペランドから値を取得（統合版）

        Args:
            operand: オペランド（辞書、文字列、数値）
            strategy_instance: 戦略インスタンス

        Returns:
            オペランドの値
        """
        try:
            # 辞書の場合（指標を表す）
            if isinstance(operand, dict):
                indicator_name = operand.get("indicator")
                if indicator_name and hasattr(strategy_instance, indicator_name):
                    indicator_value = getattr(strategy_instance, indicator_name)
                    if hasattr(indicator_value, "__getitem__"):
                        return float(indicator_value[-1])
                    return float(indicator_value)

            # 数値の場合
            if isinstance(operand, (int, float)):
                return float(operand)

            # 文字列の場合
            if isinstance(operand, str):
                # 数値文字列の場合
                if operand.replace(".", "").replace("-", "").isdigit():
                    return float(operand)

                # 価格データの場合
                if operand.lower() in ["close", "high", "low", "open"]:
                    price_data = getattr(strategy_instance.data, operand.capitalize())
                    return float(price_data[-1])

                # 指標の場合
                if hasattr(strategy_instance, operand):
                    indicator_value = getattr(strategy_instance, operand)
                    if hasattr(indicator_value, "__getitem__"):
                        return float(indicator_value[-1])
                    return float(indicator_value)

            logger.warning(f"未対応のオペランド: {operand}")
            return 0.0

        except Exception as e:
            logger.error(f"オペランド値取得エラー: {e}")
            return 0.0
