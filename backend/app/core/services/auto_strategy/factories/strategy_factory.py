"""
戦略ファクトリー

StrategyGeneから動的にbacktesting.py互換のStrategy継承クラスを生成します。
"""

# import logging
import numpy as np
from typing import Type, Dict, Any, List, Union, Tuple, Optional

from backtesting import Strategy
from app.core.services.indicators import TechnicalIndicatorService
from ..models.strategy_gene import StrategyGene, IndicatorGene, Condition

# logger = logging.getLogger(__name__)


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
                    # 各指標を初期化
                    for i, indicator_gene in enumerate(gene.indicators):
                        if indicator_gene.enabled:
                            self._init_indicator(indicator_gene)
                        else:
                            pass

                except Exception as e:
                    # logger.error(f"戦略初期化エラー: {e}", exc_info=True)
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
                    sl_price, tp_price = self.factory._calculate_tpsl_prices(
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

                        # 非常に小さな固定サイズ（1単位）を使用
                        # これによりマージン不足エラーを回避
                        fixed_size = 1.0 * position_direction

                        # 利用可能現金で購入可能かチェック（ショートの場合も絶対値で計算）
                        required_cash = abs(fixed_size) * current_price
                        if required_cash <= available_cash * 0.99:
                            # logger.info(
                            #     f"{'ロング' if fixed_size > 0 else 'ショート'}注文実行 - 固定size: {fixed_size}, 価格: {current_price}"
                            # )
                            # logger.info(
                            #     f"  必要資金: {required_cash:.2f}, 利用可能現金: {available_cash:.2f}"
                            # )

                            # 固定サイズで注文を実行（SL/TPなし）
                            # ショートの場合は負のサイズでbuy()を呼び出す
                            self.buy(size=fixed_size)
                        else:
                            pass
                    # イグジット条件チェック
                    elif self.position:
                        exit_result = self._check_exit_conditions()
                        if exit_result:
                            self.position.close()

                except Exception as e:
                    # logger.error(f"売買ロジックエラー: {e}", exc_info=True)
                    pass

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
                    else:
                        pass

                except Exception as e:
                    # logger.error(f"指標初期化エラー {indicator_gene.type}: {e}")
                    pass

            def _check_entry_conditions(self) -> bool:
                """エントリー条件をチェック（後方互換性のため保持）"""
                return self.factory._evaluate_conditions(
                    self.gene.entry_conditions, self
                )

            def _check_long_entry_conditions(self) -> bool:
                """ロングエントリー条件をチェック"""
                long_conditions = self.gene.get_effective_long_conditions()
                if not long_conditions:
                    # 条件が空の場合は、戦略設定に依存
                    # 後方互換性のため、entry_conditionsがある場合は有効とする
                    return bool(self.gene.entry_conditions)
                return self.factory._evaluate_conditions(long_conditions, self)

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
                return self.factory._evaluate_conditions(short_conditions, self)

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
            # logger.error(f"遺伝子検証エラー: {e}")
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
            # logger.error(f"指標計算エラー {indicator_type}: {e}")
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
            # logger.error(f"条件評価エラー: {e}")
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
                # logger.warning(
                #     f"比較できない値です: left={left_value}({type(left_value)}), "
                #     f"right={right_value}({type(right_value)})"
                # )
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
                # logger.warning(f"未対応の演算子: {condition.operator}")
                return False

        except Exception as e:
            # logger.error(f"条件評価エラー: {e}")
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

                # 指標の場合（indicatorsディクショナリから取得）
                if (
                    hasattr(strategy_instance, "indicators")
                    and operand in strategy_instance.indicators
                ):
                    indicator_value = strategy_instance.indicators[operand]
                    if hasattr(indicator_value, "__getitem__"):
                        return float(indicator_value[-1])
                    return float(indicator_value)

                # 指標の場合（直接属性から取得）
                if hasattr(strategy_instance, operand):
                    indicator_value = getattr(strategy_instance, operand)
                    if hasattr(indicator_value, "__getitem__"):
                        return float(indicator_value[-1])
                    return float(indicator_value)

            # logger.warning(f"未対応のオペランド: {operand}")
            return 0.0

        except Exception as e:
            # logger.error(f"オペランド値取得エラー: {e}")
            return 0.0

    def _calculate_tpsl_prices(
        self,
        current_price: float,
        stop_loss_pct: Optional[float],
        take_profit_pct: Optional[float],
        risk_management: Dict[str, Any],
        gene: Optional[Any] = None,
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        TP/SL価格を計算（従来方式と新方式の両方をサポート）

        Args:
            current_price: 現在価格
            stop_loss_pct: ストップロス割合
            take_profit_pct: テイクプロフィット割合
            risk_management: リスク管理設定

        Returns:
            (SL価格, TP価格)のタプル
        """
        try:
            # TP/SL遺伝子が利用可能かチェック（GA最適化対象）
            if gene and hasattr(gene, "tpsl_gene") and gene.tpsl_gene:
                return self._calculate_tpsl_from_gene(current_price, gene.tpsl_gene)
            # 新しいTP/SL計算方式が使用されているかチェック（従来の高度機能）
            elif self._is_advanced_tpsl_used(risk_management):
                return self._calculate_advanced_tpsl_prices(
                    current_price, stop_loss_pct, take_profit_pct, risk_management
                )
            else:
                # 従来の固定割合ベース計算
                return self._calculate_legacy_tpsl_prices(
                    current_price, stop_loss_pct, take_profit_pct
                )

        except Exception as e:
            # logger.error(f"TP/SL価格計算エラー: {e}")
            # フォールバック: 従来方式
            return self._calculate_legacy_tpsl_prices(
                current_price, stop_loss_pct, take_profit_pct
            )

    def _is_advanced_tpsl_used(self, risk_management: Dict[str, Any]) -> bool:
        """高度なTP/SL機能が使用されているかチェック"""
        return any(key.startswith("_tpsl_") for key in risk_management.keys())

    def _calculate_legacy_tpsl_prices(
        self,
        current_price: float,
        stop_loss_pct: Optional[float],
        take_profit_pct: Optional[float],
    ) -> Tuple[Optional[float], Optional[float]]:
        """従来の固定割合ベースTP/SL価格計算"""
        sl_price = current_price * (1 - stop_loss_pct) if stop_loss_pct else None
        tp_price = current_price * (1 + take_profit_pct) if take_profit_pct else None
        return sl_price, tp_price

    def _calculate_advanced_tpsl_prices(
        self,
        current_price: float,
        stop_loss_pct: Optional[float],
        take_profit_pct: Optional[float],
        risk_management: Dict[str, Any],
    ) -> Tuple[Optional[float], Optional[float]]:
        """高度なTP/SL価格計算（リスクリワード比、ボラティリティベースなど）"""
        try:
            # 使用された戦略を取得
            strategy_used = risk_management.get("_tpsl_strategy", "unknown")

            # 基本的な価格計算
            sl_price = current_price * (1 - stop_loss_pct) if stop_loss_pct else None
            tp_price = (
                current_price * (1 + take_profit_pct) if take_profit_pct else None
            )

            # 戦略固有の調整
            if strategy_used == "volatility_adaptive":
                # ボラティリティベースの場合、追加の調整を適用
                sl_price, tp_price = self._apply_volatility_adjustments(
                    current_price, sl_price, tp_price, risk_management
                )
            elif strategy_used == "risk_reward":
                # リスクリワード比ベースの場合、比率の整合性をチェック
                sl_price, tp_price = self._apply_risk_reward_adjustments(
                    current_price, sl_price, tp_price, risk_management
                )

            # メタデータをログ出力
            confidence_score = risk_management.get("_confidence_score", "N/A")
            rr_ratio = risk_management.get("_risk_reward_ratio", "N/A")

            # logger.info(
            #     f"高度なTP/SL計算: 戦略={strategy_used}, "
            #     f"RR比={rr_ratio}, 信頼度={confidence_score}"
            # )

            return sl_price, tp_price

        except Exception as e:
            # logger.error(f"高度なTP/SL価格計算エラー: {e}")
            # フォールバック
            return self._calculate_legacy_tpsl_prices(
                current_price, stop_loss_pct, take_profit_pct
            )

    def _apply_volatility_adjustments(
        self,
        current_price: float,
        sl_price: Optional[float],
        tp_price: Optional[float],
        risk_management: Dict[str, Any],
    ) -> Tuple[Optional[float], Optional[float]]:
        """ボラティリティベース調整を適用"""
        # 現在は基本実装のみ（将来的にATRベース調整を追加）
        return sl_price, tp_price

    def _apply_risk_reward_adjustments(
        self,
        current_price: float,
        sl_price: Optional[float],
        tp_price: Optional[float],
        risk_management: Dict[str, Any],
    ) -> Tuple[Optional[float], Optional[float]]:
        """リスクリワード比ベース調整を適用"""
        try:
            target_rr_ratio = risk_management.get("_risk_reward_ratio")

            if target_rr_ratio and sl_price:
                # SLが設定されている場合、RR比に基づいてTPを再計算
                sl_distance = current_price - sl_price
                tp_distance = sl_distance * target_rr_ratio
                adjusted_tp_price = current_price + tp_distance

                # logger.debug(
                #     f"RR比調整: 目標比率={target_rr_ratio}, "
                #     f"調整後TP={adjusted_tp_price}"
                # )

                return sl_price, adjusted_tp_price

            return sl_price, tp_price

        except Exception as e:
            # logger.error(f"RR比調整エラー: {e}")
            return sl_price, tp_price

    def _calculate_tpsl_from_gene(
        self, current_price: float, tpsl_gene
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        TP/SL遺伝子からTP/SL価格を計算（GA最適化対象）

        Args:
            current_price: 現在価格
            tpsl_gene: TP/SL遺伝子

        Returns:
            (SL価格, TP価格)のタプル
        """
        try:
            # TP/SL遺伝子から値を計算
            tpsl_values = tpsl_gene.calculate_tpsl_values()

            sl_pct = tpsl_values.get("stop_loss", 0.03)
            tp_pct = tpsl_values.get("take_profit", 0.06)

            # 価格に変換
            sl_price = current_price * (1 - sl_pct)
            tp_price = current_price * (1 + tp_pct)

            # logger.debug(
            #     f"TP/SL遺伝子計算: メソッド={tpsl_gene.method.value}, "
            #     f"SL={sl_pct:.3f}({sl_price:.2f}), TP={tp_pct:.3f}({tp_price:.2f})"
            # )

            return sl_price, tp_price

        except Exception as e:
            # logger.error(f"TP/SL遺伝子計算エラー: {e}")
            # フォールバック
            return current_price * 0.97, current_price * 1.06  # デフォルト3%SL, 6%TP
