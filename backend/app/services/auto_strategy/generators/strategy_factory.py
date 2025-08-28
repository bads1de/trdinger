"""
戦略ファクトリー

StrategyGeneから動的にbacktesting.py互換のStrategy継承クラスを生成します。
"""

from ..models.gene_strategy import Condition
import logging
from typing import List, Tuple, Type, Union, cast

from backtesting import Strategy

from ..services.indicator_service import IndicatorCalculator
from ..services.position_sizing_service import PositionSizingService
from ..services.tpsl_service import TPSLService
from ..core.condition_evaluator import ConditionEvaluator
from ..models.gene_strategy import IndicatorGene, StrategyGene
from ..models.condition_group import ConditionGroup

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
        logger.warning(f"🏭 戦略クラス作成開始: 指標数={len(gene.indicators)}")
        logger.warning(f"戦略遺伝子詳細: {[ind.type for ind in gene.indicators]}")

        # 遺伝子の妥当性検証
        is_valid, errors = gene.validate()
        if not is_valid:
            raise ValueError(f"Invalid strategy gene: {', '.join(errors)}")

        logger.warning("戦略遺伝子検証成功")

        # ファクトリー参照を保存
        factory = self

        logger.warning("動的クラス生成開始")

        # 動的クラス生成
        class GeneratedStrategy(Strategy):
            """動的生成された戦略クラス"""

            # backtesting.pyがパラメータを認識できるようにクラス変数として定義
            strategy_gene = gene  # デフォルト値として元のgeneを設定

            def __init__(self, broker=None, data=None, params=None):
                logger.warning(
                    f"戦略__init__開始: broker={broker is not None}, data={data is not None}, params={params}"
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
                    logger.warning(
                        f"戦略遺伝子をparamsから設定: {self.strategy_gene.indicators[0].type if self.strategy_gene.indicators else 'なし'}"
                    )
                else:
                    # デフォルトとして元のgeneを使用
                    self.strategy_gene = gene
                    self.gene = gene
                    logger.warning(
                        f"戦略遺伝子をデフォルトから設定: {gene.indicators[0].type if gene.indicators else 'なし'}"
                    )

                self.indicators = {}
                self.factory = factory  # ファクトリーへの参照

                logger.warning("戦略__init__完了")

            def init(self):
                """指標の初期化"""
                logger.warning("🚀 init()メソッド実行開始！")
                logger.warning(f"戦略遺伝子確認: {self.strategy_gene}")
                logger.warning(
                    f"戦略遺伝子指標数: {len(self.strategy_gene.indicators) if hasattr(self.strategy_gene, 'indicators') else 'なし'}"
                )

                try:
                    logger.warning(
                        f"戦略初期化開始: 指標数={len(self.strategy_gene.indicators)}"
                    )

                    # 各指標を初期化
                    for i, indicator_gene in enumerate(self.strategy_gene.indicators):
                        logger.warning(
                            f"指標処理 {i+1}/{len(self.strategy_gene.indicators)}: {indicator_gene.type}, enabled={indicator_gene.enabled}"
                        )

                        if indicator_gene.enabled:
                            logger.warning(f"指標初期化実行開始: {indicator_gene.type}")
                            self._init_indicator(indicator_gene)
                            logger.warning(f"指標初期化実行完了: {indicator_gene.type}")
                        else:
                            logger.warning(
                                f"指標スキップ（無効）: {indicator_gene.type}"
                            )

                    logger.warning("戦略初期化完了")
                except Exception as e:
                    logger.error(f"戦略初期化エラー: {e}", exc_info=True)
                    raise


            def _init_indicator(self, indicator_gene: IndicatorGene):
                """単一指標の初期化（統合版）"""
                try:
                    logger.warning(f"_init_indicator開始: {indicator_gene.type}")

                    # 指標計算器を使用して初期化
                    try:
                        logger.warning(
                            f"indicator_calculator.init_indicator呼び出し: {indicator_gene.type}"
                        )
                        factory.indicator_calculator.init_indicator(
                            indicator_gene, self
                        )
                        logger.warning(
                            f"indicator_calculator.init_indicator成功: {indicator_gene.type}"
                        )
                        return
                    except Exception as e:
                        logger.error(
                            f"indicator_calculator.init_indicator失敗: {indicator_gene.type}, エラー: {e}"
                        )

                        # フォールバック: SMA/RSIの最小構成でリカバーを試みる
                        fb = None
                        if indicator_gene.type not in ("SMA", "RSI"):
                            from ..models.gene_strategy import IndicatorGene as IG

                            period = indicator_gene.parameters.get("period", 14)
                            if period <= 0:
                                period = 14
                            # SMAを優先
                            fb = IG(
                                type="SMA", parameters={"period": period}, enabled=True
                            )
                            logger.warning(
                                f"フォールバック指標作成: {indicator_gene.type} -> SMA({period})"
                            )

                        if fb:
                            try:
                                logger.warning(
                                    f"フォールバック指標実行: SMA({fb.parameters['period']})"
                                )
                                factory.indicator_calculator.init_indicator(fb, self)
                                logger.warning(
                                    f"フォールバック指標を適用: {indicator_gene.type} -> SMA({fb.parameters['period']})"
                                )
                                return
                            except Exception as fb_e:
                                logger.error(f"フォールバック指標失敗: {fb_e}")

                        # 最後の手段: RSI(14)
                        try:
                            from ..models.gene_strategy import IndicatorGene as IG

                            fb2 = IG(
                                type="RSI", parameters={"period": 14}, enabled=True
                            )
                            logger.warning("最終フォールバック指標実行: RSI(14)")
                            factory.indicator_calculator.init_indicator(fb2, self)
                            logger.warning("フォールバック指標を適用: RSI(14)")
                            return
                        except Exception as fb2_e:
                            logger.error(f"最終フォールバック指標失敗: {fb2_e}")

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

                # デバッグログ: 条件の詳細
                if hasattr(self, "_debug_counter") and self._debug_counter % 50 == 0:
                    logger.info(f"[DEBUG] ロング条件数: {len(long_conditions)}")
                    for i, cond in enumerate(long_conditions):
                        if isinstance(cond, ConditionGroup):
                            logger.info(f"[DEBUG] ロング条件{i}: グループ条件({len(cond.conditions)}個)")
                        elif hasattr(cond, "left_operand") and hasattr(cond, "operator"):
                            logger.info(
                                f"[DEBUG] ロング条件{i}: {cond.left_operand} {cond.operator} {cond.right_operand}"
                            )
                            # 実際の値を取得してログ出力
                            try:
                                left_val = (
                                    factory.condition_evaluator.get_condition_value(
                                        cond.left_operand, self
                                    )
                                )
                                right_val = (
                                    factory.condition_evaluator.get_condition_value(
                                        cond.right_operand, self
                                    )
                                )
                                logger.info(
                                    f"[DEBUG] ロング条件{i}値: {left_val} {cond.operator} {right_val}"
                                )
                            except Exception as e:
                                logger.info(f"[DEBUG] ロング条件{i}値取得エラー: {e}")

                if not long_conditions:
                    # 条件が空の場合は、entry_conditionsを使用
                    if self.gene.entry_conditions:
                        entry_conditions = cast(
                            List[Union[Condition, ConditionGroup]],
                            self.gene.entry_conditions,
                        )

                        # デバッグログ: entry_conditionsの詳細
                        if (
                            hasattr(self, "_debug_counter")
                            and self._debug_counter % 50 == 0
                        ):
                            logger.info(
                                f"[DEBUG] entry_conditions使用: {len(entry_conditions)}件"
                            )
                            for i, cond in enumerate(entry_conditions):
                                if isinstance(cond, ConditionGroup):
                                    logger.info(f"[DEBUG] entry条件{i}: グループ条件({len(cond.conditions)}個)")
                                elif hasattr(cond, "left_operand") and hasattr(cond, "operator"):
                                    logger.info(
                                        f"[DEBUG] entry条件{i}: {cond.left_operand} {cond.operator} {cond.right_operand}"
                                    )
                                    try:
                                        left_val = factory.condition_evaluator.get_condition_value(
                                            cond.left_operand, self
                                        )
                                        right_val = factory.condition_evaluator.get_condition_value(
                                            cond.right_operand, self
                                        )
                                        logger.info(
                                            f"[DEBUG] entry条件{i}値: {left_val} {cond.operator} {right_val}"
                                        )
                                    except Exception as e:
                                        logger.info(
                                            f"[DEBUG] entry条件{i}値取得エラー: {e}"
                                        )

                        result = factory.condition_evaluator.evaluate_conditions(
                            entry_conditions, self
                        )
                        if (
                            hasattr(self, "_debug_counter")
                            and self._debug_counter % 50 == 0
                        ):
                            logger.info(f"[DEBUG] entry_conditions評価結果: {result}")
                        return result
                    return False

                result = factory.condition_evaluator.evaluate_conditions(
                    long_conditions, self
                )
                if hasattr(self, "_debug_counter") and self._debug_counter % 50 == 0:
                    logger.info(f"[DEBUG] ロング条件評価結果: {result}")
                return result

            def _check_short_entry_conditions(self) -> bool:
                """ショートエントリー条件をチェック"""
                short_conditions = cast(
                    List[Union[Condition, ConditionGroup]],
                    self.gene.get_effective_short_conditions(),
                )

                # デバッグログ: ショート条件の詳細
                if hasattr(self, "_debug_counter") and self._debug_counter % 50 == 0:
                    logger.info(f"[DEBUG] ショート条件数: {len(short_conditions)}")
                    for i, cond in enumerate(short_conditions):
                        if isinstance(cond, ConditionGroup):
                            logger.info(f"[DEBUG] ショート条件{i}: グループ条件({len(cond.conditions)}個)")
                        elif hasattr(cond, "left_operand") and hasattr(cond, "operator"):
                            logger.info(
                                f"[DEBUG] ショート条件{i}: {cond.left_operand} {cond.operator} {cond.right_operand}"
                            )
                            try:
                                left_val = (
                                    factory.condition_evaluator.get_condition_value(
                                        cond.left_operand, self
                                    )
                                )
                                right_val = (
                                    factory.condition_evaluator.get_condition_value(
                                        cond.right_operand, self
                                    )
                                )
                                logger.info(
                                    f"[DEBUG] ショート条件{i}値: {left_val} {cond.operator} {right_val}"
                                )
                            except Exception as e:
                                logger.info(f"[DEBUG] ショート条件{i}値取得エラー: {e}")

                if not short_conditions:
                    # ショート条件が空の場合は、entry_conditionsを使用
                    if self.gene.entry_conditions:
                        entry_conditions = cast(
                            List[Union[Condition, ConditionGroup]],
                            self.gene.entry_conditions,
                        )

                        # デバッグログ: entry_conditionsの詳細（ショート用）
                        if (
                            hasattr(self, "_debug_counter")
                            and self._debug_counter % 50 == 0
                        ):
                            logger.info(
                                f"[DEBUG] ショート用entry_conditions使用: {len(entry_conditions)}件"
                            )

                        result = factory.condition_evaluator.evaluate_conditions(
                            entry_conditions, self
                        )
                        if (
                            hasattr(self, "_debug_counter")
                            and self._debug_counter % 50 == 0
                        ):
                            logger.info(
                                f"[DEBUG] ショート用entry_conditions評価結果: {result}"
                            )
                        return result
                    return False

                result = factory.condition_evaluator.evaluate_conditions(
                    short_conditions, self
                )

                # デバッグログ: ショート条件評価結果
                if hasattr(self, "_debug_counter") and self._debug_counter % 50 == 0:
                    logger.info(f"[DEBUG] ショート条件評価結果: {result}")

                return result

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
                """ポジションサイズを計算"""
                # デフォルトのポジションサイズ
                default_size = 0.01

                # ポジションサイジング遺伝子が有効な場合
                if (
                    hasattr(self, "gene")
                    and self.gene.position_sizing_gene
                    and self.gene.position_sizing_gene.enabled
                ):
                    # 遺伝子に基づいてサイズを計算（実装は後で拡張）
                    pos = default_size
                else:
                    pos = default_size

                # 安全な範囲に制限
                return max(0.001, min(0.2, float(pos)))

            def next(self):
                """各バーでの戦略実行"""
                try:
                    # デバッグカウンターの初期化
                    if not hasattr(self, "_debug_counter"):
                        self._debug_counter = 0
                    self._debug_counter += 1

                    # ポジションがない場合のエントリー判定
                    if not self.position:
                        long_signal = self._check_long_entry_conditions()
                        short_signal = self._check_short_entry_conditions()

                        # デバッグログ
                        if self._debug_counter % 50 == 0:
                            logger.info(
                                f"[DEBUG] ロング条件: {long_signal}, ショート条件: {short_signal}"
                            )
                            logger.info(
                                f"[DEBUG] ロング条件数: {len(self.gene.get_effective_long_conditions())}"
                            )
                            logger.info(
                                f"[DEBUG] ショート条件数: {len(self.gene.get_effective_short_conditions())}"
                            )
                            logger.info(
                                f"[DEBUG] 現在価格: {self.data.Close[-1]}, 資産: {self.equity}"
                            )

                        # 詳細なデバッグログ
                        logger.info(
                            f"[DEBUG] ロング条件: {long_signal}, ショート条件: {short_signal}"
                        )
                        logger.info(
                            f"[DEBUG] ロング条件数: {len(self.gene.get_effective_long_conditions())}"
                        )
                        logger.info(
                            f"[DEBUG] ショート条件数: {len(self.gene.get_effective_short_conditions())}"
                        )
                        logger.info(
                            f"[DEBUG] 現在価格: {self.data.Close[-1]}, 資産: {self.equity}"
                        )

                        if long_signal or short_signal:
                            logger.info("[DEBUG] 取引条件が満たされました！")

                            # ポジションサイズを決定
                            position_size = self._calculate_position_size()

                            # TP/SL価格を計算
                            current_price = self.data.Close[-1]
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
                                logger.info(
                                    f"[DEBUG] 取引実行開始: position_direction={1.0}"
                                )
                                logger.info(f"[DEBUG] 取引サイズ決定: {position_size}")
                                logger.info(
                                    f"[DEBUG] ロング取引実行開始: size={position_size}"
                                )

                                if sl_price and tp_price:
                                    self.buy(
                                        size=position_size, sl=sl_price, tp=tp_price
                                    )
                                    logger.info(
                                        f"[DEBUG] ロング取引実行完了（SL/TP 付き）: size={position_size}"
                                    )
                                else:
                                    self.buy(size=position_size)
                                    logger.info(
                                        f"[DEBUG] ロング取引実行完了: size={position_size}"
                                    )

                            elif short_signal:
                                logger.info(
                                    f"[DEBUG] 取引実行開始: position_direction={-1.0}"
                                )
                                logger.info(f"[DEBUG] 取引サイズ決定: {position_size}")
                                logger.info(
                                    f"[DEBUG] ショート取引実行開始: size={position_size}"
                                )

                                if sl_price and tp_price:
                                    self.sell(
                                        size=position_size, sl=sl_price, tp=tp_price
                                    )
                                    logger.info(
                                        f"[DEBUG] ショート取引実行完了（SL/TP 付き）: size={position_size}"
                                    )
                                else:
                                    self.sell(size=position_size)
                                    logger.info(
                                        f"[DEBUG] ショート取引実行完了: size={position_size}"
                                    )

                    # ポジションがある場合のイグジット判定
                    elif self.position and self._check_exit_conditions():
                        logger.info("[DEBUG] イグジット条件が満たされました")
                        self.position.close()

                except Exception as e:
                    logger.error(f"next()メソッドでエラー: {e}")
                    import traceback

                    traceback.print_exc()

        # クラス名を設定
        short_id = str(gene.id).split("-")[0] if gene.id else "Unknown"
        GeneratedStrategy.__name__ = f"GS_{short_id}"
        GeneratedStrategy.__qualname__ = GeneratedStrategy.__name__

        logger.warning(f"✅ 戦略クラス作成完了: {GeneratedStrategy.__name__}")
        logger.warning(f"戦略クラス型: {type(GeneratedStrategy)}")
        logger.warning(f"戦略クラスMRO: {GeneratedStrategy.__mro__}")
        logger.warning(
            f"戦略クラス属性: {[attr for attr in dir(GeneratedStrategy) if not attr.startswith('_')]}"
        )

        return GeneratedStrategy

    def _calculate_position_size(self, gene: StrategyGene) -> float:
        """
        ポジションサイズを計算

        Args:
            gene: 戦略遺伝子

        Returns:
            float: ポジションサイズ（0.001-0.2の範囲）
        """
        # デフォルトのポジションサイズ
        default_size = 0.01

        # ポジションサイジング遺伝子が有効な場合
        if (
            gene.position_sizing_gene
            and gene.position_sizing_gene.enabled
        ):
            # 遺伝子に基づいてサイズを計算（実装は後で拡張）
            pos = default_size
        else:
            pos = default_size

        # 安全な範囲に制限
        return max(0.001, min(0.2, float(pos)))

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
