"""
テクニカル指標サービス

pandasの標準機能を使用してテクニカル指標を計算し、
データベースに保存する機能を提供します。
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Union
import logging

from database.connection import SessionLocal
from database.repositories.technical_indicator_repository import (
    TechnicalIndicatorRepository,
)
from database.repositories.ohlcv_repository import OHLCVRepository

logger = logging.getLogger(__name__)


class TechnicalIndicatorService:
    """テクニカル指標計算サービス"""

    def __init__(self):
        """サービスを初期化"""
        self.supported_indicators = {
            "SMA": {
                "periods": [5, 10, 20, 50, 100, 200],
                "function": self._calculate_sma,
            },
            "EMA": {
                "periods": [5, 10, 20, 50, 100, 200],
                "function": self._calculate_ema,
            },
            "RSI": {"periods": [14, 21, 30], "function": self._calculate_rsi},
            "MACD": {"periods": [12], "function": self._calculate_macd},
            "BB": {"periods": [20], "function": self._calculate_bollinger_bands},
            "ATR": {"periods": [14, 21], "function": self._calculate_atr},
            "STOCH": {"periods": [14], "function": self._calculate_stochastic},
            "CCI": {"periods": [20], "function": self._calculate_cci},
            "WILLR": {"periods": [14], "function": self._calculate_williams_r},
            "MOM": {"periods": [10, 14], "function": self._calculate_momentum},
            "ROC": {"periods": [10, 14], "function": self._calculate_roc},
            "PSAR": {"periods": [1], "function": self._calculate_psar},
        }

    def _validate_parameters(
        self, symbol: str, timeframe: str, indicator_type: str, period: int
    ):
        """
        パラメータの検証

        Args:
            symbol: 取引ペア
            timeframe: 時間枠
            indicator_type: 指標タイプ
            period: 期間

        Raises:
            ValueError: パラメータが無効な場合
        """
        if not symbol or not isinstance(symbol, str):
            raise ValueError("シンボルは有効な文字列である必要があります")

        if not timeframe or not isinstance(timeframe, str):
            raise ValueError("時間枠は有効な文字列である必要があります")

        if indicator_type not in self.supported_indicators:
            raise ValueError(
                f"サポートされていない指標タイプです: {indicator_type}. "
                f"サポート対象: {list(self.supported_indicators.keys())}"
            )

        if period not in self.supported_indicators[indicator_type]["periods"]:
            raise ValueError(
                f"{indicator_type}でサポートされていない期間です: {period}. "
                f"サポート対象: {self.supported_indicators[indicator_type]['periods']}"
            )

    async def _get_ohlcv_data(
        self, symbol: str, timeframe: str, limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        OHLCVデータを取得してDataFrameに変換

        Args:
            symbol: 取引ペア
            timeframe: 時間枠
            limit: 取得件数制限

        Returns:
            OHLCVデータのDataFrame
        """
        try:
            db = SessionLocal()
            try:
                ohlcv_repository = OHLCVRepository(db)
                ohlcv_data = ohlcv_repository.get_ohlcv_data(
                    symbol=symbol, timeframe=timeframe, limit=limit
                )

                if not ohlcv_data:
                    raise ValueError(
                        f"OHLCVデータが見つかりません: {symbol} {timeframe}"
                    )

                # DataFrameに変換
                df_data = []
                for record in ohlcv_data:
                    df_data.append(
                        {
                            "timestamp": record.timestamp,
                            "open": record.open,
                            "high": record.high,
                            "low": record.low,
                            "close": record.close,
                            "volume": record.volume,
                        }
                    )

                df = pd.DataFrame(df_data)
                df.set_index("timestamp", inplace=True)
                df.sort_index(inplace=True)

                logger.info(f"OHLCVデータ取得成功: {len(df)}件 ({symbol} {timeframe})")
                return df

            finally:
                db.close()

        except Exception as e:
            logger.error(f"OHLCVデータ取得エラー: {e}")
            raise

    def _calculate_sma(self, df: pd.DataFrame, period: int) -> pd.Series:
        """
        単純移動平均（SMA）を計算

        Args:
            df: OHLCVデータのDataFrame
            period: 期間

        Returns:
            SMA値のSeries
        """
        return df["close"].rolling(window=period, min_periods=period).mean()

    def _calculate_ema(self, df: pd.DataFrame, period: int) -> pd.Series:
        """
        指数移動平均（EMA）を計算

        Args:
            df: OHLCVデータのDataFrame
            period: 期間

        Returns:
            EMA値のSeries
        """
        return df["close"].ewm(span=period, adjust=False).mean()

    def _calculate_rsi(self, df: pd.DataFrame, period: int) -> pd.Series:
        """
        相対力指数（RSI）を計算

        Args:
            df: OHLCVデータのDataFrame
            period: 期間

        Returns:
            RSI値のSeries
        """
        close = df["close"]
        delta = close.diff()

        # 上昇と下降を分離
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # 平均上昇と平均下降を計算
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()

        # RSを計算
        rs = avg_gain / avg_loss

        # RSIを計算
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_macd(self, df: pd.DataFrame, period: int) -> pd.DataFrame:
        """
        MACD（Moving Average Convergence Divergence）を計算

        Args:
            df: OHLCVデータのDataFrame
            period: 期間（通常12、26、9の組み合わせ）

        Returns:
            MACD値を含むDataFrame（macd_line, signal_line, histogram）
        """
        close = df["close"]

        # EMA12とEMA26を計算
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()

        # MACD線 = EMA12 - EMA26
        macd_line = ema12 - ema26

        # シグナル線 = MACD線のEMA9
        signal_line = macd_line.ewm(span=9, adjust=False).mean()

        # ヒストグラム = MACD線 - シグナル線
        histogram = macd_line - signal_line

        # 結果をDataFrameで返す
        result = pd.DataFrame({
            'macd_line': macd_line,
            'signal_line': signal_line,
            'histogram': histogram
        })

        return result

    def _calculate_bollinger_bands(self, df: pd.DataFrame, period: int) -> pd.DataFrame:
        """
        ボリンジャーバンド（Bollinger Bands）を計算

        Args:
            df: OHLCVデータのDataFrame
            period: 期間（通常20）

        Returns:
            ボリンジャーバンド値を含むDataFrame（middle, upper, lower）
        """
        close = df["close"]

        # 中央線（SMA）
        middle = close.rolling(window=period, min_periods=period).mean()

        # 標準偏差
        std = close.rolling(window=period, min_periods=period).std()

        # 上限・下限（標準偏差の2倍）
        upper = middle + (std * 2)
        lower = middle - (std * 2)

        # 結果をDataFrameで返す
        result = pd.DataFrame({
            'middle': middle,
            'upper': upper,
            'lower': lower
        })

        return result

    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """
        ATR（Average True Range）を計算

        Args:
            df: OHLCVデータのDataFrame
            period: 期間（通常14）

        Returns:
            ATR値のSeries
        """
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # 前日終値
        prev_close = close.shift(1)

        # True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # ATR = True Rangeの移動平均
        atr = true_range.rolling(window=period, min_periods=period).mean()

        return atr

    def _calculate_stochastic(self, df: pd.DataFrame, period: int) -> pd.DataFrame:
        """
        ストキャスティクス（Stochastic Oscillator）を計算

        Args:
            df: OHLCVデータのDataFrame
            period: 期間（通常14）

        Returns:
            ストキャスティクス値を含むDataFrame（%K, %D）
        """
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # 指定期間の最高値・最安値
        highest_high = high.rolling(window=period, min_periods=period).max()
        lowest_low = low.rolling(window=period, min_periods=period).min()

        # %K = ((close - lowest_low) / (highest_high - lowest_low)) * 100
        k_percent = ((close - lowest_low) / (highest_high - lowest_low)) * 100

        # %D = %Kの3期間移動平均
        d_percent = k_percent.rolling(window=3, min_periods=3).mean()

        # 結果をDataFrameで返す
        result = pd.DataFrame({
            'k_percent': k_percent,
            'd_percent': d_percent
        })

        return result

    def _calculate_cci(self, df: pd.DataFrame, period: int) -> pd.Series:
        """
        CCI（Commodity Channel Index）を計算

        Args:
            df: OHLCVデータのDataFrame
            period: 期間（通常20）

        Returns:
            CCI値のSeries
        """
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # Typical Price = (high + low + close) / 3
        typical_price = (high + low + close) / 3

        # Typical Priceの移動平均
        sma_tp = typical_price.rolling(window=period, min_periods=period).mean()

        # Mean Deviation = Typical Priceと移動平均の差の絶対値の移動平均
        mean_deviation = (typical_price - sma_tp).abs().rolling(window=period, min_periods=period).mean()

        # CCI = (Typical Price - SMA) / (0.015 * Mean Deviation)
        cci = (typical_price - sma_tp) / (0.015 * mean_deviation)

        return cci

    def _calculate_williams_r(self, df: pd.DataFrame, period: int) -> pd.Series:
        """
        Williams %R を計算

        Args:
            df: OHLCVデータのDataFrame
            period: 期間（通常14）

        Returns:
            Williams %R値のSeries
        """
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # 指定期間の最高値・最安値
        highest_high = high.rolling(window=period, min_periods=period).max()
        lowest_low = low.rolling(window=period, min_periods=period).min()

        # Williams %R = ((highest_high - close) / (highest_high - lowest_low)) * -100
        williams_r = ((highest_high - close) / (highest_high - lowest_low)) * -100

        return williams_r

    def _calculate_momentum(self, df: pd.DataFrame, period: int) -> pd.Series:
        """
        モメンタム（Momentum）を計算

        Args:
            df: OHLCVデータのDataFrame
            period: 期間（通常10または14）

        Returns:
            モメンタム値のSeries
        """
        close = df["close"]

        # Momentum = 現在の終値 - N期間前の終値
        momentum = close - close.shift(period)

        return momentum

    def _calculate_roc(self, df: pd.DataFrame, period: int) -> pd.Series:
        """
        ROC（Rate of Change）を計算

        Args:
            df: OHLCVデータのDataFrame
            period: 期間（通常10または14）

        Returns:
            ROC値のSeries
        """
        close = df["close"]

        # ROC = ((現在の終値 - N期間前の終値) / N期間前の終値) * 100
        prev_close = close.shift(period)
        roc = ((close - prev_close) / prev_close) * 100

        return roc

    def _calculate_psar(self, df: pd.DataFrame, period: int) -> pd.Series:
        """
        PSAR（Parabolic SAR）を計算

        Args:
            df: OHLCVデータのDataFrame
            period: 期間（通常1、PSARは期間に依存しない）

        Returns:
            PSAR値のSeries
        """
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # PSARの初期設定
        af = 0.02  # 加速因子の初期値
        max_af = 0.20  # 加速因子の最大値

        # 結果を格納するSeries
        psar = pd.Series(index=df.index, dtype=float)

        if len(df) < 2:
            return psar

        # 初期値設定
        psar.iloc[0] = low.iloc[0]
        trend = 1  # 1: 上昇トレンド, -1: 下降トレンド
        ep = high.iloc[0]  # Extreme Point
        current_af = af

        for i in range(1, len(df)):
            if trend == 1:  # 上昇トレンド
                psar.iloc[i] = psar.iloc[i-1] + current_af * (ep - psar.iloc[i-1])

                # PSARが前日または当日の安値を上回った場合、トレンド転換
                if psar.iloc[i] > low.iloc[i] or psar.iloc[i] > low.iloc[i-1]:
                    trend = -1
                    psar.iloc[i] = ep
                    ep = low.iloc[i]
                    current_af = af
                else:
                    # 新しい高値更新
                    if high.iloc[i] > ep:
                        ep = high.iloc[i]
                        current_af = min(current_af + af, max_af)
            else:  # 下降トレンド
                psar.iloc[i] = psar.iloc[i-1] + current_af * (ep - psar.iloc[i-1])

                # PSARが前日または当日の高値を下回った場合、トレンド転換
                if psar.iloc[i] < high.iloc[i] or psar.iloc[i] < high.iloc[i-1]:
                    trend = 1
                    psar.iloc[i] = ep
                    ep = high.iloc[i]
                    current_af = af
                else:
                    # 新しい安値更新
                    if low.iloc[i] < ep:
                        ep = low.iloc[i]
                        current_af = min(current_af + af, max_af)

        return psar

    async def calculate_technical_indicator(
        self,
        symbol: str,
        timeframe: str,
        indicator_type: str,
        period: int,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        テクニカル指標を計算

        Args:
            symbol: 取引ペア
            timeframe: 時間枠
            indicator_type: 指標タイプ（SMA, EMA, RSI）
            period: 期間
            limit: OHLCVデータの取得件数制限

        Returns:
            計算されたテクニカル指標データのリスト

        Raises:
            ValueError: パラメータが無効な場合
            Exception: 計算エラーの場合
        """
        # パラメータの検証
        self._validate_parameters(symbol, timeframe, indicator_type, period)

        try:
            logger.info(
                f"テクニカル指標計算開始: {symbol} {timeframe} {indicator_type}({period})"
            )

            # OHLCVデータを取得
            df = await self._get_ohlcv_data(symbol, timeframe, limit)

            # 指標を計算
            calculate_func = self.supported_indicators[indicator_type]["function"]
            indicator_result = calculate_func(df, period)

            # 結果をリストに変換
            results = []

            # 複数値を返す指標（MACD、ボリンジャーバンド）の処理
            if indicator_type == "MACD":
                for timestamp in indicator_result.index:
                    macd_line = indicator_result.loc[timestamp, 'macd_line']
                    signal_line = indicator_result.loc[timestamp, 'signal_line']
                    histogram = indicator_result.loc[timestamp, 'histogram']

                    if pd.notna(macd_line) and pd.notna(signal_line):
                        results.append({
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "indicator_type": indicator_type,
                            "period": period,
                            "value": float(macd_line),
                            "signal_value": float(signal_line),
                            "histogram_value": float(histogram),
                            "upper_band": None,
                            "lower_band": None,
                            "timestamp": timestamp,
                        })

            elif indicator_type == "BB":
                for timestamp in indicator_result.index:
                    middle = indicator_result.loc[timestamp, 'middle']
                    upper = indicator_result.loc[timestamp, 'upper']
                    lower = indicator_result.loc[timestamp, 'lower']

                    if pd.notna(middle) and pd.notna(upper) and pd.notna(lower):
                        results.append({
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "indicator_type": indicator_type,
                            "period": period,
                            "value": float(middle),
                            "signal_value": None,
                            "histogram_value": None,
                            "upper_band": float(upper),
                            "lower_band": float(lower),
                            "timestamp": timestamp,
                        })

            elif indicator_type == "STOCH":
                for timestamp in indicator_result.index:
                    k_percent = indicator_result.loc[timestamp, 'k_percent']
                    d_percent = indicator_result.loc[timestamp, 'd_percent']

                    if pd.notna(k_percent) and pd.notna(d_percent):
                        results.append({
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "indicator_type": indicator_type,
                            "period": period,
                            "value": float(k_percent),
                            "signal_value": float(d_percent),
                            "histogram_value": None,
                            "upper_band": None,
                            "lower_band": None,
                            "timestamp": timestamp,
                        })

            else:
                # 単一値を返す指標（SMA、EMA、RSI、ATR、CCI、WILLR）の処理
                for timestamp, value in indicator_result.items():
                    if pd.notna(value):  # NaN値をスキップ
                        results.append({
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "indicator_type": indicator_type,
                            "period": period,
                            "value": float(value),
                            "signal_value": None,
                            "histogram_value": None,
                            "upper_band": None,
                            "lower_band": None,
                            "timestamp": timestamp,
                        })

            logger.info(
                f"テクニカル指標計算完了: {len(results)}件 "
                f"({symbol} {timeframe} {indicator_type}({period}))"
            )
            return results

        except Exception as e:
            logger.error(f"テクニカル指標計算エラー: {e}")
            raise

    async def _save_technical_indicator_to_database(
        self,
        technical_indicator_data: List[Dict[str, Any]],
        repository: TechnicalIndicatorRepository,
    ) -> int:
        """
        テクニカル指標データをデータベースに保存

        Args:
            technical_indicator_data: テクニカル指標データのリスト
            repository: テクニカル指標リポジトリ

        Returns:
            保存された件数
        """
        try:
            if not technical_indicator_data:
                logger.warning("保存するテクニカル指標データがありません")
                return 0

            # データベースに保存
            saved_count = repository.insert_technical_indicator_data(
                technical_indicator_data
            )

            logger.info(f"テクニカル指標データ保存完了: {saved_count}件")
            return saved_count

        except Exception as e:
            logger.error(f"テクニカル指標データ保存エラー: {e}")
            raise

    async def calculate_and_save_technical_indicator(
        self,
        symbol: str,
        timeframe: str,
        indicator_type: str,
        period: int,
        repository: Optional[TechnicalIndicatorRepository] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        テクニカル指標を計算してデータベースに保存

        Args:
            symbol: 取引ペア
            timeframe: 時間枠
            indicator_type: 指標タイプ
            period: 期間
            repository: テクニカル指標リポジトリ（テスト用）
            limit: OHLCVデータの取得件数制限

        Returns:
            計算・保存結果

        Raises:
            ValueError: パラメータが無効な場合
            Exception: データベースエラーの場合
        """
        try:
            # テクニカル指標を計算
            technical_indicator_data = await self.calculate_technical_indicator(
                symbol, timeframe, indicator_type, period, limit
            )

            # データベースに保存
            if repository is None:
                # 実際のデータベースセッションを使用
                db = SessionLocal()
                try:
                    repository = TechnicalIndicatorRepository(db)
                    saved_count = await self._save_technical_indicator_to_database(
                        technical_indicator_data, repository
                    )
                    db.close()
                except Exception as e:
                    db.close()
                    raise
            else:
                # テスト用のリポジトリを使用
                saved_count = await self._save_technical_indicator_to_database(
                    technical_indicator_data, repository
                )

            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "indicator_type": indicator_type,
                "period": period,
                "calculated_count": len(technical_indicator_data),
                "saved_count": saved_count,
                "success": True,
            }

        except Exception as e:
            logger.error(f"テクニカル指標計算・保存エラー: {e}")
            raise

    async def calculate_and_save_multiple_indicators(
        self,
        symbol: str,
        timeframe: str,
        indicators: List[Dict[str, Any]],
        repository: Optional[TechnicalIndicatorRepository] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        複数のテクニカル指標を一括計算・保存

        Args:
            symbol: 取引ペア
            timeframe: 時間枠
            indicators: 指標設定のリスト [{"type": "SMA", "period": 20}, ...]
            repository: テクニカル指標リポジトリ（テスト用）
            limit: OHLCVデータの取得件数制限

        Returns:
            一括計算・保存結果
        """
        try:
            logger.info(
                f"複数テクニカル指標一括計算開始: {symbol} {timeframe} "
                f"({len(indicators)}種類)"
            )

            results = []
            total_calculated = 0
            total_saved = 0
            successful_indicators = 0
            failed_indicators = []

            for indicator_config in indicators:
                try:
                    indicator_type = indicator_config["type"]
                    period = indicator_config["period"]

                    result = await self.calculate_and_save_technical_indicator(
                        symbol=symbol,
                        timeframe=timeframe,
                        indicator_type=indicator_type,
                        period=period,
                        repository=repository,
                        limit=limit,
                    )

                    results.append(result)
                    total_calculated += result["calculated_count"]
                    total_saved += result["saved_count"]
                    successful_indicators += 1

                    logger.info(
                        f"✅ {indicator_type}({period}): "
                        f"{result['calculated_count']}件計算, {result['saved_count']}件保存"
                    )

                    # レート制限対応
                    await asyncio.sleep(0.1)

                except Exception as e:
                    logger.error(f"❌ {indicator_config} 計算エラー: {e}")
                    failed_indicators.append(
                        {"indicator": indicator_config, "error": str(e)}
                    )

            logger.info(
                f"複数テクニカル指標一括計算完了: "
                f"{successful_indicators}/{len(indicators)}成功, "
                f"計算{total_calculated}件, 保存{total_saved}件"
            )

            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "total_indicators": len(indicators),
                "successful_indicators": successful_indicators,
                "failed_indicators": len(failed_indicators),
                "total_calculated": total_calculated,
                "total_saved": total_saved,
                "results": results,
                "failures": failed_indicators,
                "success": successful_indicators > 0,
            }

        except Exception as e:
            logger.error(f"複数テクニカル指標一括計算エラー: {e}")
            raise

    def get_default_indicators(self) -> List[Dict[str, Any]]:
        """
        デフォルトの指標設定を取得

        Returns:
            デフォルト指標設定のリスト
        """
        default_indicators = []

        # SMA: 20, 50期間
        for period in [20, 50]:
            default_indicators.append({"type": "SMA", "period": period})

        # EMA: 20, 50期間
        for period in [20, 50]:
            default_indicators.append({"type": "EMA", "period": period})

        # RSI: 14期間
        default_indicators.append({"type": "RSI", "period": 14})

        # MACD: 12期間（標準設定）
        default_indicators.append({"type": "MACD", "period": 12})

        # ボリンジャーバンド: 20期間
        default_indicators.append({"type": "BB", "period": 20})

        # ATR: 14期間
        default_indicators.append({"type": "ATR", "period": 14})

        # ストキャスティクス: 14期間
        default_indicators.append({"type": "STOCH", "period": 14})

        # CCI: 20期間
        default_indicators.append({"type": "CCI", "period": 20})

        # Williams %R: 14期間
        default_indicators.append({"type": "WILLR", "period": 14})

        # モメンタム: 10期間
        default_indicators.append({"type": "MOM", "period": 10})

        # ROC: 10期間
        default_indicators.append({"type": "ROC", "period": 10})

        # PSAR: 1期間（固定）
        default_indicators.append({"type": "PSAR", "period": 1})

        return default_indicators

    def get_supported_indicators(self) -> Dict[str, Any]:
        """
        サポートされている指標の情報を取得

        Returns:
            サポート指標の情報
        """
        return {
            indicator_type: {
                "periods": config["periods"],
                "description": self._get_indicator_description(indicator_type),
            }
            for indicator_type, config in self.supported_indicators.items()
        }

    def _get_indicator_description(self, indicator_type: str) -> str:
        """
        指標の説明を取得

        Args:
            indicator_type: 指標タイプ

        Returns:
            指標の説明
        """
        descriptions = {
            "SMA": "単純移動平均 - 指定期間の終値の平均値",
            "EMA": "指数移動平均 - 直近の価格により重みを置いた移動平均",
            "RSI": "相対力指数 - 買われすぎ・売られすぎを示すオシレーター（0-100）",
            "MACD": "MACD - トレンドの方向性と強さを示すオシレーター",
            "BB": "ボリンジャーバンド - ボラティリティとサポート・レジスタンスを示す",
            "ATR": "ATR - 平均真の値幅、ボラティリティを測定する指標",
            "STOCH": "ストキャスティクス - 買われすぎ・売られすぎを示すオシレーター（0-100）",
            "CCI": "CCI - 商品チャネル指数、トレンドの強さを測定",
            "WILLR": "Williams %R - 逆張りシグナルに有効なオシレーター（-100-0）",
            "MOM": "モメンタム - 価格変化の勢いを測定する指標",
            "ROC": "ROC - 変化率、価格の変化をパーセンテージで表示",
            "PSAR": "PSAR - パラボリックSAR、トレンド転換点を示す",
        }
        return descriptions.get(indicator_type, "説明なし")
