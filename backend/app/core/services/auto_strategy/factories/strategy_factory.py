"""
戦略ファクトリー

StrategyGeneから動的にbacktesting.py互換のStrategy継承クラスを生成します。
既存のTALibAdapterとの統合を重視した実装です。
"""

from typing import Type, List, Tuple
import logging
import pandas as pd
import numpy as np
from backtesting import Strategy

from ..models.strategy_gene import StrategyGene, IndicatorGene, Condition
from app.core.services.indicators.adapters.trend_adapter import TrendAdapter
from app.core.services.indicators.adapters.momentum_adapter import MomentumAdapter
from app.core.services.indicators.adapters.volatility_adapter import VolatilityAdapter

logger = logging.getLogger(__name__)


class StrategyFactory:
    """
    戦略ファクトリー

    StrategyGeneから動的にStrategy継承クラスを生成し、
    既存のTALibAdapterシステムと統合します。
    """

    def __init__(self):
        """初期化"""
        self.indicator_cache = {}

        # 指標タイプとアダプターのマッピング
        self.indicator_adapters = {
            # トレンド系
            "SMA": TrendAdapter.sma,
            "EMA": TrendAdapter.ema,
            "TEMA": TrendAdapter.tema,
            "DEMA": TrendAdapter.dema,
            "T3": TrendAdapter.t3,
            "WMA": TrendAdapter.wma,
            "KAMA": TrendAdapter.kama,
            "MIDPOINT": TrendAdapter.midpoint,
            "MIDPRICE": TrendAdapter.midprice,
            "TRIMA": TrendAdapter.trima,
            # モメンタム系
            "RSI": MomentumAdapter.rsi,
            "STOCH": MomentumAdapter.stochastic,
            "CCI": MomentumAdapter.cci,
            "WILLIAMS": MomentumAdapter.williams_r,
            "ADX": MomentumAdapter.adx,
            "AROON": MomentumAdapter.aroon,
            "MFI": MomentumAdapter.mfi,
            "MOMENTUM": MomentumAdapter.momentum,
            "ROC": MomentumAdapter.roc,
            "BOP": MomentumAdapter.bop,
            "PPO": MomentumAdapter.ppo,
            # ボラティリティ系
            "ATR": VolatilityAdapter.atr,
            "NATR": VolatilityAdapter.natr,
            "TRANGE": VolatilityAdapter.trange,
            # 複合指標（特別処理）
            "MACD": self._calculate_macd,
            "BB": self._calculate_bollinger_bands,
        }

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

            def __init__(self, broker=None, data=None, params=None):
                super().__init__(broker, data, params)
                self.gene = gene
                self.indicators = {}
                self.factory = factory  # ファクトリーへの参照

            def init(self):
                """指標の初期化"""
                try:
                    # 各指標を初期化
                    for indicator_gene in gene.indicators:
                        if indicator_gene.enabled:
                            self._init_indicator(indicator_gene)

                    logger.info(f"戦略初期化完了: {len(self.indicators)}個の指標")

                except Exception as e:
                    logger.error(f"戦略初期化エラー: {e}")
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

                    # リスク管理
                    self._apply_risk_management()

                except Exception as e:
                    logger.error(f"売買ロジックエラー: {e}")
                    # エラーが発生してもバックテストを継続

            def _init_indicator(self, indicator_gene: IndicatorGene):
                """単一指標の初期化"""
                try:
                    indicator_type = indicator_gene.type
                    parameters = indicator_gene.parameters

                    if indicator_type in self.factory.indicator_adapters:
                        # backtesting.pyの_ArrayをPandas Seriesに変換
                        close_data = self._convert_to_series(self.data.Close)
                        high_data = self._convert_to_series(self.data.High)
                        low_data = self._convert_to_series(self.data.Low)

                        # 指標を計算（指標タイプに応じて引数を調整）
                        if indicator_type in ["MACD", "BB"]:
                            # 複合指標の場合
                            result = self.factory.indicator_adapters[indicator_type](
                                close_data, **parameters
                            )
                        elif indicator_type in [
                            "ADX",
                            "CCI",
                            "WILLIAMS",
                            "ATR",
                            "NATR",
                            "TRANGE",
                        ]:
                            # High, Low, Closeが必要な指標
                            period = int(parameters.get("period", 20))
                            result = self.factory.indicator_adapters[indicator_type](
                                high_data, low_data, close_data, period
                            )
                        elif indicator_type == "MIDPRICE":
                            # MIDPRICEはHigh, Lowが必要
                            period = int(parameters.get("period", 14))
                            result = self.factory.indicator_adapters[indicator_type](
                                high_data, low_data, period
                            )
                        elif indicator_type == "BOP":
                            # BOPはOpen, High, Low, Closeが必要
                            open_data = self._convert_to_series(self.data.Open)
                            result = self.factory.indicator_adapters[indicator_type](
                                open_data, high_data, low_data, close_data
                            )
                        elif indicator_type == "PPO":
                            # PPOは複数パラメータが必要
                            fastperiod = int(parameters.get("period", 12))
                            slowperiod = int(parameters.get("slow_period", 26))
                            matype = int(parameters.get("matype", 0))
                            result = self.factory.indicator_adapters[indicator_type](
                                close_data, fastperiod, slowperiod, matype
                            )
                        elif indicator_type in ["STOCH"]:
                            # Stochasticは特別な処理
                            period = int(parameters.get("period", 14))
                            result = self.factory.indicator_adapters[indicator_type](
                                high_data, low_data, close_data, period
                            )
                        else:
                            # 単一値指標の場合（Close価格のみ）
                            period = int(parameters.get("period", 20))
                            result = self.factory.indicator_adapters[indicator_type](
                                close_data, period
                            )

                        # 指標をbacktesting.pyのインジケーターとして登録
                        if indicator_type == "BOP":
                            # BOPは期間を使用しない
                            indicator_name = "BOP"
                        elif indicator_type == "PPO":
                            # PPOは複数パラメータを使用
                            fastperiod = int(parameters.get("period", 12))
                            slowperiod = int(parameters.get("slow_period", 26))
                            indicator_name = f"PPO_{fastperiod}_{slowperiod}"
                        else:
                            # 通常の指標
                            indicator_name = (
                                f"{indicator_type}_{parameters.get('period', '')}"
                            )

                        # resultがSeriesの場合は値のみを取得
                        if hasattr(result, "values"):
                            indicator_values = result.values
                        else:
                            indicator_values = result

                        self.indicators[indicator_name] = self.I(
                            lambda: indicator_values, name=indicator_name
                        )

                        logger.debug(f"指標初期化完了: {indicator_name}")

                    else:
                        logger.warning(f"未対応の指標タイプ: {indicator_type}")

                except Exception as e:
                    logger.error(f"指標初期化エラー ({indicator_gene.type}): {e}")

            def _convert_to_series(self, bt_array):
                """backtesting.pyの_ArrayをPandas Seriesに変換"""
                try:
                    import pandas as pd

                    # _Arrayオブジェクトから値とインデックスを取得
                    if hasattr(bt_array, "_data"):
                        # backtesting.pyの内部データ構造にアクセス
                        values = bt_array._data
                        index = (
                            bt_array._data.index
                            if hasattr(bt_array._data, "index")
                            else range(len(values))
                        )
                        return pd.Series(values, index=index)
                    else:
                        # フォールバック: 配列として扱う
                        return pd.Series(bt_array)
                except Exception as e:
                    logger.error(f"データ変換エラー: {e}")
                    # 最後の手段: 単純な配列として扱う
                    import pandas as pd

                    return pd.Series(list(bt_array))

            def _check_entry_conditions(self) -> bool:
                """エントリー条件をチェック"""
                try:
                    for condition in gene.entry_conditions:
                        if not self._evaluate_condition(condition):
                            return False
                    return True
                except Exception as e:
                    logger.error(f"エントリー条件チェックエラー: {e}")
                    return False

            def _check_exit_conditions(self) -> bool:
                """イグジット条件をチェック"""
                try:
                    for condition in gene.exit_conditions:
                        if self._evaluate_condition(condition):
                            return True
                    return False
                except Exception as e:
                    logger.error(f"イグジット条件チェックエラー: {e}")
                    return False

            def _evaluate_condition(self, condition: Condition) -> bool:
                """単一条件を評価"""
                try:
                    left_value = self._get_condition_value(condition.left_operand)
                    right_value = self._get_condition_value(condition.right_operand)

                    if left_value is None or right_value is None:
                        return False

                    # 演算子に基づく比較
                    if condition.operator == ">":
                        return left_value > right_value
                    elif condition.operator == "<":
                        return left_value < right_value
                    elif condition.operator == ">=":
                        return left_value >= right_value
                    elif condition.operator == "<=":
                        return left_value <= right_value
                    elif condition.operator == "==":
                        return abs(left_value - right_value) < 1e-6
                    elif condition.operator == "cross_above":
                        return self._check_crossover(
                            condition.left_operand, condition.right_operand, "above"
                        )
                    elif condition.operator == "cross_below":
                        return self._check_crossover(
                            condition.left_operand, condition.right_operand, "below"
                        )

                    return False

                except Exception as e:
                    logger.error(f"条件評価エラー: {e}")
                    return False

            def _get_condition_value(self, operand):
                """条件のオペランドから値を取得（OI/FR対応版）"""
                try:
                    # 数値の場合
                    if isinstance(operand, (int, float)):
                        return float(operand)

                    # 文字列の場合（指標名、価格、またはOI/FR）
                    if isinstance(operand, str):
                        # 基本価格データ
                        if operand == "price" or operand == "close":
                            return self.data.Close[-1]
                        elif operand == "high":
                            return self.data.High[-1]
                        elif operand == "low":
                            return self.data.Low[-1]
                        elif operand == "open":
                            return self.data.Open[-1]
                        elif operand == "volume":
                            return self.data.Volume[-1]

                        # OI/FRデータ（新規追加）
                        elif operand == "OpenInterest":
                            return self._get_oi_fr_value("OpenInterest")
                        elif operand == "FundingRate":
                            return self._get_oi_fr_value("FundingRate")

                        # 技術指標
                        elif operand in self.indicators:
                            indicator = self.indicators[operand]
                            return indicator[-1] if len(indicator) > 0 else None

                    return None

                except Exception as e:
                    logger.error(f"オペランド値取得エラー: {e}")
                    return None

            def _get_oi_fr_value(self, data_type: str):
                """OI/FRデータから値を取得（堅牢版）"""
                try:
                    # backtesting.pyのdataオブジェクトからOI/FRデータにアクセス
                    if hasattr(self.data, data_type):
                        data_series = getattr(self.data, data_type)

                        # データ系列の型チェックと変換
                        if hasattr(data_series, "__len__") and len(data_series) > 0:
                            # pandas Series, numpy array, listなどに対応
                            try:
                                if hasattr(data_series, "iloc"):
                                    # pandas Series
                                    value = data_series.iloc[-1]
                                elif hasattr(data_series, "__getitem__"):
                                    # numpy array, list
                                    value = data_series[-1]
                                else:
                                    logger.warning(
                                        f"{data_type}データの型が不明: {type(data_series)}"
                                    )
                                    return 0.0

                                # NaN値チェック
                                if pd.isna(value) or (
                                    isinstance(value, float) and np.isnan(value)
                                ):
                                    logger.warning(
                                        f"{data_type}データにNaN値が含まれています"
                                    )
                                    # 有効な値を後ろから探す
                                    for i in range(len(data_series) - 2, -1, -1):
                                        if hasattr(data_series, "iloc"):
                                            prev_value = data_series.iloc[i]
                                        else:
                                            prev_value = data_series[i]

                                        if not pd.isna(prev_value) and not (
                                            isinstance(prev_value, float)
                                            and np.isnan(prev_value)
                                        ):
                                            return float(prev_value)

                                    # 全てNaNの場合
                                    logger.warning(f"{data_type}データが全てNaNです")
                                    return 0.0

                                return float(value)

                            except (IndexError, KeyError) as e:
                                logger.warning(
                                    f"{data_type}データのインデックスエラー: {e}"
                                )
                                return 0.0
                        else:
                            logger.warning(f"{data_type}データが空です")
                            return 0.0
                    else:
                        logger.warning(f"{data_type}データが利用できません")
                        return 0.0

                except Exception as e:
                    logger.error(f"{data_type}データ取得エラー: {e}")
                    return 0.0

            def _check_crossover(
                self, left_operand: str, right_operand: str, direction: str
            ) -> bool:
                """クロスオーバーをチェック"""
                try:
                    left_current = self._get_condition_value(left_operand)
                    right_current = self._get_condition_value(right_operand)

                    # 前の値も取得（簡略化）
                    if len(self.data.Close) < 2:
                        return False

                    # 簡略化: 現在の値のみで判定
                    if direction == "above":
                        return left_current > right_current
                    else:  # below
                        return left_current < right_current

                except Exception as e:
                    logger.error(f"クロスオーバーチェックエラー: {e}")
                    return False

            def _apply_risk_management(self):
                """リスク管理を適用"""
                try:
                    if not self.position or not self.trades:
                        return

                    risk_config = gene.risk_management
                    current_price = self.data.Close[-1]

                    # アクティブな取引から平均エントリー価格を計算
                    total_value = 0
                    total_size = 0
                    for trade in self.trades:
                        total_value += abs(trade.size) * trade.entry_price
                        total_size += abs(trade.size)

                    if total_size == 0:
                        return

                    avg_entry_price = total_value / total_size

                    # ストップロス
                    if "stop_loss" in risk_config:
                        stop_loss_pct = risk_config["stop_loss"]
                        if self.position.is_long:
                            stop_price = avg_entry_price * (1 - stop_loss_pct)
                            if current_price <= stop_price:
                                self.position.close()
                        else:
                            stop_price = avg_entry_price * (1 + stop_loss_pct)
                            if current_price >= stop_price:
                                self.position.close()

                    # テイクプロフィット
                    if "take_profit" in risk_config:
                        take_profit_pct = risk_config["take_profit"]
                        if self.position.is_long:
                            take_price = avg_entry_price * (1 + take_profit_pct)
                            if current_price >= take_price:
                                self.position.close()
                        else:
                            take_price = avg_entry_price * (1 - take_profit_pct)
                            if current_price <= take_price:
                                self.position.close()

                except Exception as e:
                    logger.error(f"リスク管理エラー: {e}")

        # クラス名を設定
        GeneratedStrategy.__name__ = f"GeneratedStrategy_{gene.id}"
        GeneratedStrategy.__qualname__ = GeneratedStrategy.__name__

        return GeneratedStrategy

    def validate_gene(self, gene: StrategyGene) -> Tuple[bool, List[str]]:
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
            if indicator.enabled and indicator.type not in self.indicator_adapters:
                errors.append(f"未対応の指標: {indicator.type}")

        # 条件の参照整合性チェック
        available_indicators = []
        for ind in gene.indicators:
            if ind.enabled:
                if ind.type == "BOP":
                    # BOPは期間を使用しない
                    available_indicators.append("BOP")
                elif ind.type == "PPO":
                    # PPOは複数パラメータを使用
                    fastperiod = int(ind.parameters.get("period", 12))
                    slowperiod = int(ind.parameters.get("slow_period", 26))
                    available_indicators.append(f"PPO_{fastperiod}_{slowperiod}")
                else:
                    # 通常の指標
                    available_indicators.append(
                        f"{ind.type}_{ind.parameters.get('period', '')}"
                    )

        # 有効なデータソース（OI/FR対応版）
        valid_data_sources = [
            "price",
            "close",
            "high",
            "low",
            "open",
            "volume",
            "OpenInterest",
            "FundingRate",  # OI/FRデータソースを追加
        ]

        for condition in gene.entry_conditions + gene.exit_conditions:
            # 左オペランドの検証
            if isinstance(condition.left_operand, str):
                if (
                    condition.left_operand not in valid_data_sources
                    and condition.left_operand not in available_indicators
                ):
                    errors.append(f"未定義の指標参照: {condition.left_operand}")

            # 右オペランドの検証（文字列の場合）
            if isinstance(condition.right_operand, str):
                if (
                    condition.right_operand not in valid_data_sources
                    and condition.right_operand not in available_indicators
                ):
                    errors.append(f"未定義の指標参照: {condition.right_operand}")

        return len(errors) == 0, errors

    def _calculate_macd(self, data, fast_period=12, slow_period=26, signal_period=9):
        """MACD計算（複合指標）"""
        try:
            return MomentumAdapter.macd(data, fast_period, slow_period, signal_period)
        except Exception as e:
            logger.error(f"MACD計算エラー: {e}")
            return None

    def _calculate_bollinger_bands(self, data, period=20, std_dev=2):
        """ボリンジャーバンド計算（複合指標）"""
        try:
            return VolatilityAdapter.bollinger_bands(data, period, std_dev)
        except Exception as e:
            logger.error(f"ボリンジャーバンド計算エラー: {e}")
            return None
