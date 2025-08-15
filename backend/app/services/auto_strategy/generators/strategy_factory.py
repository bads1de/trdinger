"""
戦略ファクトリー

StrategyGeneから動的にbacktesting.py互換のStrategy継承クラスを生成します。
"""

import logging
from typing import Tuple, Type

from backtesting import Strategy

from ..services.indicator_service import IndicatorCalculator
from ..services.position_sizing_service import PositionSizingService
from ..services.tpsl_service import TPSLService
from ..core.condition_evaluator import ConditionEvaluator
from ..models.gene_strategy import IndicatorGene, StrategyGene

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
                logger.warning("戦略遺伝子確認: {self.strategy_gene}")
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
                except Exception:
                    logger.error("戦略初期化エラー: {e}", exc_info=True)
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

                    # ロング・ショートエントリー条件チェック
                    long_entry_result = self._check_long_entry_conditions()
                    short_entry_result = self._check_short_entry_conditions()

                    # デバッグログ: ロング・ショート条件の評価結果
                    if hasattr(self, "_debug_counter"):
                        self._debug_counter += 1
                    else:
                        self._debug_counter = 1

                    # 100回に1回ログを出力（パフォーマンス考慮）
                    if self._debug_counter % 100 == 0:
                        logger.debug(
                            f"[DEBUG] ロング条件: {long_entry_result}, ショート条件: {short_entry_result}"
                        )
                        logger.debug(
                            f"[DEBUG] ロング条件数: {len(self.gene.get_effective_long_conditions())}"
                        )
                        logger.debug(
                            f"[DEBUG] ショート条件数: {len(self.gene.get_effective_short_conditions())}"
                        )

                    if not self.position and (long_entry_result or short_entry_result):
                        # backtesting.pyのマージン問題を回避するため、非常に小さな固定サイズを使用
                        current_price = self.data.Close[-1]

                        # 現在の資産を取得
                        current_equity = getattr(self, "equity", 100000.0)
                        available_cash = getattr(self, "cash", current_equity)

                        # ポジション方向を決定
                        if long_entry_result and short_entry_result:
                            # 両方の条件が満たされた場合はランダムに選択（より公平）
                            import random

                            position_direction = random.choice([1.0, -1.0])
                        elif long_entry_result:
                            position_direction = 1.0  # ロング
                        elif short_entry_result:
                            position_direction = -1.0  # ショート
                        else:
                            # どちらの条件も満たされない場合はエントリーしない
                            position_direction = None

                        # デバッグログ: ポジション方向決定
                        if self._debug_counter % 100 == 0:
                            logger.debug(
                                f"[DEBUG] ポジション方向: {position_direction} (ロング={long_entry_result}, ショート={short_entry_result})"
                            )

                        # ポリシーに委譲
                        if position_direction is None:
                            return
                        from app.services.auto_strategy.core.order_execution_policy import (
                            OrderExecutionPolicy,
                        )

                        sl_price, tp_price = OrderExecutionPolicy.compute_tpsl_prices(
                            factory,
                            current_price,
                            risk_management,
                            gene,
                            position_direction,
                        )

                        calculated_size = factory._calculate_position_size(
                            gene, current_equity, current_price, self.data
                        )
                        final_size = calculated_size * position_direction

                        if self._debug_counter % 100 == 0:
                            logger.debug(
                                f"[DEBUG] 計算サイズ: {calculated_size}, 最終サイズ: {final_size}"
                            )

                        final_size_bt = (
                            OrderExecutionPolicy.compute_final_position_size(
                                factory,
                                gene,
                                current_price=current_price,
                                current_equity=current_equity,
                                available_cash=available_cash,
                                data=self.data,
                                raw_size=final_size,
                            )
                        )
                        if final_size_bt == 0:
                            return

                        if self.gene.tpsl_gene and self.gene.tpsl_gene.enabled:
                            if final_size_bt > 0:
                                self.buy(size=final_size_bt, sl=sl_price, tp=tp_price)
                            else:
                                self.sell(
                                    size=abs(final_size_bt), sl=sl_price, tp=tp_price
                                )
                        else:
                            if final_size_bt > 0:
                                self.buy(size=final_size_bt)
                            else:
                                self.sell(size=abs(final_size_bt))
                    # イグジット条件チェック（TP/SL遺伝子が存在しない場合のみ）
                    elif self.position:
                        # TP/SL遺伝子が存在する場合はイグジット条件をスキップ
                        if not self.gene.tpsl_gene or not self.gene.tpsl_gene.enabled:
                            exit_result = self._check_exit_conditions()
                            if exit_result:
                                self.position.close()

                except Exception as e:
                    logger.error(f"売買ロジックエラー: {e}", exc_info=True)

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
                long_conditions = self.gene.get_effective_long_conditions()
                if not long_conditions:
                    # 条件が空の場合は、entry_conditionsを使用
                    if self.gene.entry_conditions:
                        return factory.condition_evaluator.evaluate_conditions(
                            self.gene.entry_conditions, self
                        )
                    return False
                return factory.condition_evaluator.evaluate_conditions(
                    long_conditions, self
                )

            def _check_short_entry_conditions(self) -> bool:
                """ショートエントリー条件をチェック"""
                short_conditions = self.gene.get_effective_short_conditions()

                # デバッグログ: ショート条件の詳細
                if hasattr(self, "_debug_counter") and self._debug_counter % 100 == 0:
                    logger.debug(
                        f"[DEBUG] ショート条件詳細: {[str(c.__dict__) for c in short_conditions]}"
                    )
                    logger.debug(
                        f"[DEBUG] ロング・ショート分離: {self.gene.has_long_short_separation()}"
                    )

                if not short_conditions:
                    # ショート条件が空の場合は、entry_conditionsを使用
                    if self.gene.entry_conditions:
                        return factory.condition_evaluator.evaluate_conditions(
                            self.gene.entry_conditions, self
                        )
                    return False

                result = factory.condition_evaluator.evaluate_conditions(
                    short_conditions, self
                )

                # デバッグログ: ショート条件評価結果
                if hasattr(self, "_debug_counter") and self._debug_counter % 100 == 0:
                    logger.debug(f"[DEBUG] ショート条件評価結果: {result}")

                return result

            def _check_exit_conditions(self) -> bool:
                """イグジット条件をチェック（統合版）"""
                # TP/SL遺伝子が存在し有効な場合はイグジット条件をスキップ
                if self.gene.tpsl_gene and self.gene.tpsl_gene.enabled:
                    return False

                return factory.condition_evaluator.evaluate_conditions(
                    self.gene.exit_conditions, self
                )

        # クラス名を設定
        # クラス名を短縮
        short_id = str(gene.id).split("-")[0]
        GeneratedStrategy.__name__ = f"GS_{short_id}"
        GeneratedStrategy.__qualname__ = GeneratedStrategy.__name__

        logger.warning(f"✅ 戦略クラス作成完了: {GeneratedStrategy.__name__}")
        logger.warning(f"戦略クラス型: {type(GeneratedStrategy)}")
        logger.warning(f"戦略クラスMRO: {GeneratedStrategy.__mro__}")
        logger.warning(
            f"戦略クラス属性: {[attr for attr in dir(GeneratedStrategy) if not attr.startswith('_')]}"
        )

        return GeneratedStrategy

    def _calculate_position_size(
        self, gene, account_balance: float, current_price: float, data
    ) -> float:
        """PositionSizingService を直接使用してサイズ計算"""
        try:
            # 市場データの準備（Helper の処理を内包）
            market_data = {}
            if (
                data is not None
                and hasattr(data, "High")
                and hasattr(data, "Low")
                and hasattr(data, "Close")
            ):
                # ATR の推定値（計算器側で必要なら補助的に使用可能）
                current_price_safe = (
                    current_price if current_price and current_price > 0 else 1.0
                )
                market_data["atr_pct"] = 0.04 if current_price_safe == 0 else 0.04
            trade_history = []

            calc_result = self.position_sizing_service.calculate_position_size(
                gene=getattr(gene, "position_sizing_gene", None)
                or getattr(gene, "position_sizing", None)
                or gene,
                account_balance=account_balance,
                current_price=current_price,
                symbol="BTCUSDT",
                market_data=market_data,
                trade_history=trade_history,
                use_cache=False,
            )
            return float(getattr(calc_result, "position_size", 0.0))
        except Exception:
            # フォールバック: 従来の risk_management.position_size（上限は広め）
            try:
                pos = getattr(gene, "risk_management", {}).get("position_size", 0.1)
            except Exception:
                pos = 0.1
            return max(0.01, min(50.0, float(pos)))

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
