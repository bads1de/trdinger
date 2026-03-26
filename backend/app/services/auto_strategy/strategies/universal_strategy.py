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

from ..core.evaluation.condition_evaluator import ConditionEvaluator
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
from .ml_filter import MLFilter
from .order_manager import OrderManager
from .position_manager import PositionManager
from .stateful_conditions import StatefulConditionsEvaluator

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

        # ヘルパークラスの初期化
        self.position_manager = PositionManager(self)
        self.stateful_conditions_evaluator = StatefulConditionsEvaluator(self)
        self.ml_filter = MLFilter(self)

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
        """有効なTPSL遺伝子を取得（方向別設定を優先し、共通設定にフォールバック）"""
        return self._get_effective_sub_gene(direction, "tpsl")

    def _get_effective_entry_gene(self, direction: float) -> Union[None, EntryGene]:
        """有効なエントリー遺伝子を取得（方向別設定を優先し、共通設定にフォールバック）"""
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
                    # ベクトル化されたATRを使用（高速）
                    if (
                        hasattr(self, "_precomputed_atr")
                        and self._precomputed_atr is not None
                    ):
                        idx = len(self.data) - 1
                        if 0 <= idx < len(self._precomputed_atr):
                            atr = self._precomputed_atr[idx]
                            import numpy as np

                            if not np.isnan(atr) and current_price > 0:
                                market_data["atr_pct"] = atr / current_price

                    # フォールバック（低速）
                    if "atr_pct" not in market_data:
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
                self.ml_filter.precompute_ml_features()

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

            # 4. ポジションサイジング用のATR事前計算
            self._precomputed_atr = None
            if (
                self.gene.position_sizing_gene
                and self.gene.position_sizing_gene.enabled
            ):
                try:
                    lookback = getattr(
                        self.gene.position_sizing_gene, "lookback_period", 14
                    )
                    # pandas-taを使って一括計算
                    # self.data.df がない場合でも一時的にDFを作成して計算
                    try:
                        import pandas_ta_classic as ta

                        # 高速化のため必要なカラムのみでDataFrameを構築
                        # backtesting.pyのデータ配列はnumpy array
                        high = self.data.High
                        low = self.data.Low
                        close = self.data.Close

                        # pandas-taはSeriesを期待するためDF化
                        temp_df = pd.DataFrame(
                            {"high": high, "low": low, "close": close}
                        )

                        # pandas-taで計算
                        self._precomputed_atr = ta.atr(
                            temp_df["high"],
                            temp_df["low"],
                            temp_df["close"],
                            length=lookback,
                        ).values

                        logger.debug("ATR事前計算完了")

                    except ImportError:
                        # pandas-taがない場合などはフォールバック
                        logger.warning(
                            "pandas-taが見つからないためATR事前計算をスキップ"
                        )
                        pass
                    except Exception as e:
                        logger.debug(
                            f"ATR事前計算中のエラー（フォールバック使用）: {e}"
                        )
                except Exception as e:
                    logger.debug(f"ATR事前計算失敗: {e}")

            # 5. TP/SL用のATR事前計算（Long/Shortで異なる場合がある）
            self._precomputed_tpsl_atr = {}
            for direction in [1.0, -1.0]:
                tpsl_gene = self._get_effective_tpsl_gene(direction)
                if tpsl_gene and tpsl_gene.method in (
                    TPSLMethod.VOLATILITY_BASED,
                    TPSLMethod.ADAPTIVE,
                    TPSLMethod.STATISTICAL,
                ):
                    try:
                        atr_period = getattr(tpsl_gene, "atr_period", 14)
                        # 同じ期間の計算は一度だけ行う
                        if atr_period not in self._precomputed_tpsl_atr:
                            if hasattr(self.data, "df"):
                                import pandas_ta_classic as ta

                                high = self.data.df["High"]
                                low = self.data.df["Low"]
                                close = self.data.df["Close"]
                                # pandas-taのATR計算
                                atr_values = ta.atr(
                                    high, low, close, length=atr_period
                                ).values
                                self._precomputed_tpsl_atr[atr_period] = atr_values
                    except Exception as e:
                        logger.debug(f"TP/SL ATR事前計算失敗: {e}")

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

            # 1. 事前計算されたATRを使用（高速）
            if (
                hasattr(self, "_precomputed_tpsl_atr")
                and atr_period in self._precomputed_tpsl_atr
            ):
                idx = len(self.data) - 1
                atr_array = self._precomputed_tpsl_atr[atr_period]
                if 0 <= idx < len(atr_array):
                    val = atr_array[idx]
                    import numpy as np

                    if not np.isnan(val):
                        market_data["atr"] = val

            # 2. フォールバック（事前計算がない場合のみ従来の重い処理）
            if "atr" not in market_data:
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
            self.stateful_conditions_evaluator.process_stateful_triggers()

            # 2. 既存ポジションの悲観的決済チェック
            if self.position and self._sl_price is not None:
                if self.position_manager.check_pessimistic_exit():
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
                    stateful_dir = (
                        self.stateful_conditions_evaluator.get_stateful_entry_direction()
                    )
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

    # ===== ML フィルターメソッド =====

    def _ml_allows_entry(self, direction: float) -> bool:
        """
        MLフィルターがエントリーを許可するかチェック

        Args:
            direction: 1.0 (Long) or -1.0 (Short)

        Returns:
            True: エントリー許可, False: エントリーブロック
        """
        return self.ml_filter.ml_allows_entry(direction)

    def _prepare_current_features(self) -> Optional[pd.DataFrame]:
        """
        MLフィルター用の現在の特徴量を準備

        Returns:
            特徴量DataFrame、準備できない場合はNone
        """
        return self.ml_filter.prepare_current_features()

    def _process_stateful_triggers(self):
        """ステートフルトリガーを処理"""
        self.stateful_conditions_evaluator.process_stateful_triggers()

    def _get_stateful_entry_direction(self) -> Optional[float]:
        """
        ステートフルエントリーの方向を取得

        Returns:
            1.0 (Long), -1.0 (Short), または None
        """
        return self.stateful_conditions_evaluator.get_stateful_entry_direction()

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
