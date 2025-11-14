"""
特徴量エンジニアリングサービス

OHLCV、ファンディングレート（FR）、建玉残高（OI）データを受け取り、
市場の歪みや偏りを捉える高度な特徴量を計算します。

各特徴量計算クラスを統合し、単一責任原則に従って実装されています。
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from app.config.unified_config import unified_config

from .advanced_features import AdvancedFeatureEngineer
from .crypto_features import CryptoFeatures
from .data_frequency_manager import DataFrequencyManager
from .interaction_features import InteractionFeatureCalculator
from .market_data_features import MarketDataFeatureCalculator
from .price_features import PriceFeatureCalculator
from .technical_features import TechnicalFeatureCalculator

logger = logging.getLogger(__name__)


# デフォルト特徴量リスト
# 研究目的のため、デフォルトでは全特徴量を使用（None）
# 必要に応じて環境変数 ML_FEATURE_ENGINEERING__FEATURE_ALLOWLIST でカスタムリストを指定可能
DEFAULT_FEATURE_ALLOWLIST: Optional[List[str]] = None

# 以下は参考用の特徴量リスト（過学習防止用に絞り込む場合に使用可能）
"""
RECOMMENDED_FEATURES = [
        # === 基本的なテクニカル指標 ===
        "RSI_14",
        "MACD",
        "MACD_signal",
        "MACD_hist",
        "MA_Short_7",
        "MA_Long_25",
        "BB_Upper",
        "BB_Middle",
        "BB_Lower",
        "BB_Position",
        "BB_Width",
        "ATR_14",
        # === ボリューム関連 ===
        "Volume_MA_Ratio",
        "Volume_Trend",
        # === ボラティリティ関連 ===
        "Volatility_20",
        "Volatility_Ratio",
        # === モメンタム指標 ===
        "Momentum_14",
        "ROC_10",
        # === 価格関連 ===
        "Price_Change_Pct",
        "High_Low_Range",
        "Close_Position_in_Range",
        # === 市場レジーム ===
        "Market_Regime",
        "Trend_Strength",
        # === 建玉残高関連（OI） ===
        "OI_Change_Rate_24h",
        "Volatility_Adjusted_OI",
        "OI_Trend",
        "OI_Normalized",
        # === 複合指標 ===
        "FR_OI_Ratio",
        "Market_Heat_Index",
        "Market_Stress",
        "Market_Balance",
        # === 暗号通貨特化特徴量 ===
        "Price_Volume_Correlation",
        "Funding_Rate_Impact",
        # === 高度な特徴量（一部） ===
        "Price_Momentum_Regime",
        "Volatility_Regime",
]
"""


class FeatureEngineeringService:
    """
    特徴量エンジニアリングサービス

    各特徴量計算クラスを統合し、高度な特徴量を生成します。
    単一責任原則に従い、各特徴量タイプの計算は専用クラスに委譲します。
    """

    def __init__(self):
        """初期化"""
        self.feature_cache = {}
        self.max_cache_size = 10  # 最大キャッシュサイズ
        self.cache_ttl = 3600  # キャッシュ有効期限（秒）

        # 特徴量計算クラスを初期化
        self.price_calculator = PriceFeatureCalculator()
        self.market_data_calculator = MarketDataFeatureCalculator()
        self.technical_calculator = TechnicalFeatureCalculator()
        self.interaction_calculator = InteractionFeatureCalculator()

        # データ頻度統一マネージャー
        self.frequency_manager = DataFrequencyManager()

        # 暗号通貨特化特徴量エンジニアリング（デフォルトで有効）
        self.crypto_features = CryptoFeatures()
        logger.debug("暗号通貨特化特徴量を有効化しました")

        # 高度な特徴量エンジニアリング（デフォルトで有効）
        self.advanced_features = AdvancedFeatureEngineer()
        logger.debug("高度な特徴量エンジニアリングを有効化しました")

    def calculate_advanced_features(
        self,
        ohlcv_data: pd.DataFrame,
        funding_rate_data: Optional[pd.DataFrame] = None,
        open_interest_data: Optional[pd.DataFrame] = None,
        lookback_periods: Optional[Dict[str, int]] = None,
        profile: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        高度な特徴量を計算

        Args:
            ohlcv_data: OHLCV価格データ
            funding_rate_data: ファンディングレートデータ（オプション）
            open_interest_data: 建玉残高データ（オプション）
            lookback_periods: 各特徴量の計算期間設定
            profile: 特徴量プロファイル ('research' または 'production')。
                    Noneの場合は設定から読み込み

        Returns:
            特徴量が追加されたDataFrame

        Examples:
            >>> # 研究用（全特徴量）
            >>> features = service.calculate_advanced_features(
            ...     ohlcv_data, profile="research"
            ... )
            >>>
            >>> # 本番用（厳選特徴量）
            >>> features = service.calculate_advanced_features(
            ...     ohlcv_data, profile="production"
            ... )
        """
        try:
            if ohlcv_data.empty:
                raise ValueError("OHLCVデータが空です")

            # DataFrameのインデックスをDatetimeIndexに変換（脆弱性修正）
            if not isinstance(ohlcv_data.index, pd.DatetimeIndex):
                if "timestamp" in ohlcv_data.columns:
                    ohlcv_data = ohlcv_data.set_index("timestamp")
                    logger.info("timestampカラムをインデックスに設定しました")
                else:
                    # timestampカラムがない場合は、現在の時刻から生成
                    logger.warning(
                        "timestampカラムが見つからないため、仮のDatetimeIndexを生成します"
                    )
                    ohlcv_data.index = pd.date_range(
                        start="2024-01-01", periods=len(ohlcv_data), freq="1H"
                    )

            # インデックスがDatetimeIndexであることを確認
            if not isinstance(ohlcv_data.index, pd.DatetimeIndex):
                raise ValueError(
                    "DataFrameのインデックスはDatetimeIndexである必要があります"
                )

            # メモリ使用量制限
            if len(ohlcv_data) > 50000:
                logger.warning(
                    f"大量のデータ（{len(ohlcv_data)}行）、最新50,000行に制限"
                )
                ohlcv_data = ohlcv_data.tail(50000)

            # キャッシュキーを生成
            cache_key = self._generate_cache_key(
                ohlcv_data,
                funding_rate_data,
                open_interest_data,
                lookback_periods,
            )

            # キャッシュから結果を取得
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                return cached_result

            # データ頻度統一処理（最優先問題の解決）
            logger.info("データ頻度統一処理を開始")
            ohlcv_timeframe = self.frequency_manager.detect_ohlcv_timeframe(ohlcv_data)

            # データ整合性検証
            validation_result = self.frequency_manager.validate_data_alignment(
                ohlcv_data, funding_rate_data, open_interest_data
            )

            if not validation_result["is_valid"]:
                logger.warning("データ整合性に問題があります:")
                for error in validation_result["errors"]:
                    logger.warning(f"  エラー: {error}")

            # データ頻度を統一
            funding_rate_data, open_interest_data = (
                self.frequency_manager.align_data_frequencies(
                    ohlcv_data, funding_rate_data, open_interest_data, ohlcv_timeframe
                )
            )

            # デフォルトの計算期間
            if lookback_periods is None:
                lookback_periods = {
                    "short_ma": 10,
                    "long_ma": 50,
                    "volatility": 20,
                    "momentum": 14,
                    "volume": 20,
                }

            # 結果DataFrameを初期化（メモリ効率化）
            result_df = ohlcv_data.copy()

            # データ型を最適化
            result_df = self._optimize_dtypes(result_df)

            # 基本的な価格特徴量
            result_df = self.price_calculator.calculate_price_features(
                result_df, lookback_periods
            )

            # ボラティリティ特徴量
            result_df = self.price_calculator.calculate_volatility_features(
                result_df, lookback_periods
            )

            # 出来高特徴量
            result_df = self.price_calculator.calculate_volume_features(
                result_df, lookback_periods
            )

            # ファンディングレート特徴量（データがある場合）
            if funding_rate_data is not None and not funding_rate_data.empty:
                result_df = self.market_data_calculator.calculate_funding_rate_features(  # noqa: E501
                    result_df, funding_rate_data, lookback_periods
                )
                # Removed: FR疑似特徴量の中間クリーニング
                # (低寄与度特徴量削除: 2025-01-05)
                # 削除された特徴量: FR_MA_24, FR_MA_168, FR_Change,
                # FR_Change_Rate, Price_FR_Divergence, FR_Extreme_High,
                # FR_Extreme_Low, FR_Normalized, FR_Trend, FR_Volatility
                # 注: これらの特徴量はすべて削除されたため、
                # 中間クリーニングは不要
            else:
                # ファンディングレートデータが不足している場合、疑似データを生成
                logger.warning(
                    "ファンディングレートデータが不足しています。疑似特徴量を生成します。"
                )
                result_df = self._generate_pseudo_funding_rate_features(
                    result_df, lookback_periods
                )

            # 建玉残高特徴量（データがある場合）
            if open_interest_data is not None and not open_interest_data.empty:
                result_df = self.market_data_calculator.calculate_open_interest_features(  # noqa: E501
                    result_df, open_interest_data, lookback_periods
                )
                # 中間クリーニング（削除された特徴量を除外）
                # Removed: "OI_Change_Rate", "OI_Surge",
                # "OI_Price_Correlation" (低寄与度: 2025-01-05)
                existing_oi_columns = [
                    "OI_Change_Rate_24h",
                    "Volatility_Adjusted_OI",
                    "OI_Trend",
                    "OI_Normalized",
                ]
                existing_oi_columns = [
                    col for col in existing_oi_columns if col in result_df.columns
                ]
                if existing_oi_columns:
                    try:
                        medians = result_df[existing_oi_columns].median()
                        result_df[existing_oi_columns] = result_df[
                            existing_oi_columns
                        ].fillna(medians)
                    except Exception as e:
                        logger.warning(f"OI中間クリーニングでエラー: {e}")
            else:
                # 建玉残高データが不足している場合、疑似データを生成
                logger.warning(
                    "建玉残高データが不足しています。疑似特徴量を生成します。"
                )
                result_df = self._generate_pseudo_open_interest_features(
                    result_df, lookback_periods
                )

            # 複合特徴量（FR + OI）
            if (
                funding_rate_data is not None
                and not funding_rate_data.empty
                and open_interest_data is not None
                and not open_interest_data.empty
            ):
                result_df = self.market_data_calculator.calculate_composite_features(
                    result_df, funding_rate_data, open_interest_data, lookback_periods
                )
                # 中間クリーニング
                composite_columns = [
                    "FR_OI_Ratio",
                    "Market_Heat_Index",
                    "Market_Stress",
                    "Market_Balance",
                ]
                existing_composite_columns = [
                    col for col in composite_columns if col in result_df.columns
                ]
                if existing_composite_columns:
                    try:
                        medians = result_df[existing_composite_columns].median()
                        result_df[existing_composite_columns] = result_df[
                            existing_composite_columns
                        ].fillna(medians)
                    except Exception as e:
                        logger.warning(f"Composite中間クリーニングでエラー: {e}")

            # 市場レジーム特徴量
            result_df = self.technical_calculator.calculate_market_regime_features(
                result_df, lookback_periods
            )

            # モメンタム特徴量
            result_df = self.technical_calculator.calculate_momentum_features(
                result_df, lookback_periods
            )

            # パターン認識特徴量
            result_df = self.technical_calculator.calculate_pattern_features(
                result_df, lookback_periods
            )

            # 暗号通貨特化特徴量（デフォルトで追加）
            if self.crypto_features is not None:
                logger.debug("暗号通貨特化特徴量を計算中...")
                result_df = self.crypto_features.create_crypto_features(
                    result_df, funding_rate_data, open_interest_data
                )

            # 高度な特徴量エンジニアリング（デフォルトで追加）
            if self.advanced_features is not None:
                logger.debug("高度な特徴量を計算中...")
                result_df = self.advanced_features.create_advanced_features(
                    result_df, funding_rate_data, open_interest_data
                )

            # 相互作用特徴量（全ての基本特徴量が計算された後に実行）
            result_df = self.interaction_calculator.calculate_interaction_features(
                result_df
            )

            # データバリデーションとクリーンアップ
            # 主要な価格列を除外した特徴量列の一覧が必要な場合のみ、その場で計算すること
            # （未使用変数を避けるため、ここでは保持しません）

            # 高品質なデータ前処理を実行（スケーリング有効化、IQRベース外れ値検出）
            logger.info("統計的手法による特徴量前処理を実行中...")
            try:
                # シンプルなNaN補完処理で置換
                numeric_columns = result_df.select_dtypes(include=[np.number]).columns
                for col in numeric_columns:
                    # Pandas Series比較を安全に行う - boolに変換してから評価
                    has_nan = bool(result_df[col].isna().any())
                    if has_nan:
                        median_val = result_df[col].median()
                        result_df[col] = result_df[col].fillna(median_val)
                logger.info("データ前処理完了")
            except Exception as e:
                logger.warning(f"データ前処理エラー: {e}")

            # NaN値の追加的な処理（配列形状エラーを防ぐ）
            try:
                # 数値列のNaN値を安全に変換
                numeric_columns = result_df.select_dtypes(include=[np.number]).columns
                for col in numeric_columns:
                    # Pandas Series比較を安全に行う - boolに変換してから評価
                    has_null = bool(result_df[col].isnull().any())
                    if has_null:
                        # NaNや無限大をmedianで置換
                        median_val = result_df[col].median()
                        if pd.isna(median_val):
                            median_val = 0.0
                        result_df[col] = result_df[col].fillna(median_val)

                        # 無限大値を有限値に置換
                        result_df[col] = result_df[col].replace(
                            [np.inf, -np.inf], median_val
                        )

                # 配列形状の検証と修正
                if result_df.empty:
                    raise ValueError("前処理後に空のDataFrameになりました")

                # 全ての列が2D配列として扱えることを確認
                for col in result_df.columns:
                    if not isinstance(result_df[col].iloc[0], (int, float)):
                        logger.warning(
                            f"非数値列を検出: {col}, 型: {type(result_df[col].iloc[0])}"
                        )
                        # 非数値列を数値に変換または削除
                        try:
                            result_df[col] = pd.to_numeric(
                                result_df[col], errors="coerce"
                            )
                            result_df[col] = result_df[col].fillna(0.0)
                        except Exception:
                            logger.warning(f"列 {col} を削除（非数値データ）")
                            result_df = result_df.drop(columns=[col])

            except Exception as nan_error:
                logger.warning(f"NaN値処理エラー、基本情報のみ使用: {nan_error}")
                # 基本的なNaN処理のみ実行
                result_df = result_df.fillna(0.0)

            # 重複カラムの削除（複数の計算クラスで生成される可能性がある）
            # Pandas Series比較を安全に行う - boolに変換してから評価
            has_duplicates = bool(result_df.columns.duplicated().any())
            if has_duplicates:
                duplicated_cols = result_df.columns[
                    result_df.columns.duplicated()
                ].tolist()
                logger.warning(
                    f"重複カラムを検出、最初のもののみ保持: {duplicated_cols}"
                )
                # 重複を保持せずに削除（keep='first'で最初のもののみ保持）
                result_df = result_df.loc[
                    :, ~result_df.columns.duplicated(keep="first")
                ]

            logger.info(f"特徴量計算完了: {len(result_df.columns)}個の特徴量を生成")

            # プロファイルベースの特徴量フィルタリング
            result_df = self._apply_feature_profile(result_df, profile)

            # 結果をキャッシュに保存
            self._save_to_cache(cache_key, result_df)

            return result_df

        except Exception as e:
            logger.error(f"特徴量計算エラー: {e}")
            raise

    def get_feature_names(self) -> List[str]:
        """
        生成される特徴量名のリストを取得

        Returns:
            特徴量名のリスト
        """
        feature_names = []

        # 各計算クラスから特徴量名を取得
        feature_names.extend(self.price_calculator.get_feature_names())
        feature_names.extend(self.market_data_calculator.get_feature_names())
        feature_names.extend(self.technical_calculator.get_feature_names())
        feature_names.extend(self.interaction_calculator.get_feature_names())

        return feature_names

    def _generate_cache_key(
        self,
        ohlcv_data: pd.DataFrame,
        funding_rate_data: Optional[pd.DataFrame],
        open_interest_data: Optional[pd.DataFrame],
        lookback_periods: Optional[Dict[str, int]],
    ) -> str:
        """
        キャッシュキーを生成

        Args:
            ohlcv_data: OHLCV価格データ
            funding_rate_data: ファンディングレートデータ（オプション）
            open_interest_data: 建玉残高データ（オプション）
            lookback_periods: 各特徴量の計算期間設定

        Returns:
            生成されたキャッシュキー文字列
        """
        import hashlib

        # データのハッシュを計算
        ohlcv_hash = hashlib.md5(str(ohlcv_data.shape).encode()).hexdigest()[:8]
        fr_hash = hashlib.md5(
            str(
                funding_rate_data.shape if funding_rate_data is not None else "None"
            ).encode()
        ).hexdigest()[:8]
        oi_hash = hashlib.md5(
            str(
                open_interest_data.shape if open_interest_data is not None else "None"
            ).encode()
        ).hexdigest()[:8]
        periods_hash = hashlib.md5(
            str(
                sorted(lookback_periods.items())
                if lookback_periods is not None
                else "None"
            ).encode()
        ).hexdigest()[:8]

        return f"features_{ohlcv_hash}_{fr_hash}_{oi_hash}_{periods_hash}"

    def _get_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """
        キャッシュから結果を取得

        Args:
            cache_key: キャッシュキー

        Returns:
            キャッシュされたDataFrame、またはNone
        """
        try:
            if cache_key in self.feature_cache:
                cached_data, timestamp = self.feature_cache[cache_key]

                # TTLチェック
                if datetime.now().timestamp() - timestamp < self.cache_ttl:
                    return cached_data.copy()
                else:
                    # 期限切れのキャッシュを削除
                    del self.feature_cache[cache_key]

            return None

        except Exception as e:
            logger.warning(f"キャッシュ取得エラー: {e}")
            return None

    def _save_to_cache(self, cache_key: str, data: pd.DataFrame):
        """
        結果をキャッシュに保存

        Args:
            cache_key: キャッシュキー
            data: 保存するDataFrame
        """
        try:
            # キャッシュサイズ制限
            if len(self.feature_cache) >= self.max_cache_size:
                # 最も古いキャッシュを削除
                oldest_key = min(
                    self.feature_cache.keys(), key=lambda k: self.feature_cache[k][1]
                )
                del self.feature_cache[oldest_key]

            # 新しいキャッシュを保存
            self.feature_cache[cache_key] = (data.copy(), datetime.now().timestamp())

        except Exception as e:
            logger.warning(f"キャッシュ保存エラー: {e}")

    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        データ型を最適化してメモリ使用量を削減

        Args:
            df: 最適化するDataFrame

        Returns:
            最適化されたDataFrame
        """
        try:
            optimized_df = df.copy()

            for col in optimized_df.columns:
                if col == "timestamp":
                    continue

                if optimized_df[col].dtype == "float64":
                    # float64をfloat32に変換（精度は十分）
                    optimized_df[col] = optimized_df[col].astype("float32")
                elif optimized_df[col].dtype == "int64":
                    # int64をint32に変換（範囲が十分な場合）
                    # Pandas Series比較を安全に行う - スカラー値として評価
                    col_min = float(optimized_df[col].min())
                    col_max = float(optimized_df[col].max())
                    if col_min >= -2147483648 and col_max <= 2147483647:
                        optimized_df[col] = optimized_df[col].astype("int32")

            return optimized_df

        except Exception as e:
            logger.warning(f"データ型最適化エラー: {e}")
            return df

    def _apply_feature_profile(
        self, df: pd.DataFrame, profile: Optional[str] = None
    ) -> pd.DataFrame:
        """
        特徴量フィルタリングを適用（研究目的用にシンプル化）

        Args:
            df: フィルタリング前のDataFrame
            profile: 未使用（後方互換性のため残す）

        Returns:
            フィルタリング後のDataFrame
        """
        try:
            # 設定からallowlistを取得
            allowlist = unified_config.ml.feature_engineering.feature_allowlist

            # allowlistが指定されていない場合は、デフォルトリストまたは全特徴量を使用
            if allowlist is None:
                # デフォルトリストを使用
                allowlist = DEFAULT_FEATURE_ALLOWLIST
                if allowlist is None:
                    # デフォルトもNoneなら全特徴量を使用
                    logger.info(f"全特徴量を使用: {len(df.columns)}個")
                    return df
                else:
                    logger.info(f"デフォルト特徴量リストを適用: {len(allowlist)}個")
            else:
                logger.info(f"カスタム特徴量リストを適用: {len(allowlist)}個")

            # 価格列など、必ず保持すべき基本カラム
            essential_columns = ["open", "high", "low", "close", "volume"]

            # 保持する列を決定
            columns_to_keep = []

            # 基本カラムを追加（存在する場合のみ）
            for col in essential_columns:
                if col in df.columns:
                    columns_to_keep.append(col)

            # allowlistの特徴量を追加（存在する場合のみ）
            missing_features = []
            for feature in allowlist:
                if feature in df.columns:
                    if feature not in columns_to_keep:
                        columns_to_keep.append(feature)
                else:
                    missing_features.append(feature)

            # 存在しない特徴量に対する警告
            if missing_features:
                logger.warning(
                    f"allowlistに指定された{len(missing_features)}個の特徴量が "
                    f"見つかりません: {missing_features[:10]}"
                    + (
                        f"... (他{len(missing_features) - 10}個)"
                        if len(missing_features) > 10
                        else ""
                    )
                )

            # フィルタリング実行
            original_count = len(df.columns)
            filtered_df = df[columns_to_keep]
            dropped_count = original_count - len(filtered_df.columns)

            logger.info(
                f"特徴量フィルタリング完了: "
                f"{original_count}個 → {len(filtered_df.columns)}個の特徴量 "
                f"({dropped_count}個をドロップ)"
            )
            essential_in_result = [
                c for c in essential_columns if c in filtered_df.columns
            ]
            selected_count = len(filtered_df.columns) - len(essential_in_result)
            logger.info(
                f"保持された特徴量: "
                f"基本カラム={len(essential_in_result)}個, "
                f"選択特徴量={selected_count}個"
            )

            return filtered_df

        except Exception as e:
            logger.error(f"特徴量プロファイル適用エラー: {e}")
            # エラー時は元のDataFrameを返す
            logger.warning(
                "エラーのため、フィルタリングをスキップして全特徴量を保持します"
            )
            return df

    def clear_cache(self):
        """
        キャッシュをクリア
        """
        self.feature_cache.clear()
        logger.info("特徴量キャッシュをクリアしました")

    def _generate_pseudo_funding_rate_features(
        self, df: pd.DataFrame, lookback_periods: Dict[str, int]
    ) -> pd.DataFrame:
        """
        ファンディングレート疑似特徴量を生成

        Args:
            df: 価格データ
            lookback_periods: 計算期間設定

        Returns:
            疑似特徴量が追加されたDataFrame
        """
        try:
            # Removed: FR疑似特徴量生成（低寄与度特徴量削除: 2025-01-05）
            # 削除された特徴量: fr_extreme_high, fr_extreme_low, fr_ma_24,
            # fr_ma_168, fr_change, fr_change_rate, price_fr_divergence,
            # fr_normalized, fr_trend, fr_volatility
            result_df = df.copy()
            logger.info(
                "ファンディングレート疑似特徴量の生成をスキップしました（低寄与度のため削除）"
            )
            return result_df

        except Exception as e:
            logger.error(f"ファンディングレート疑似特徴量生成エラー: {e}")
            return df

    def _generate_pseudo_open_interest_features(
        self, df: pd.DataFrame, lookback_periods: Dict[str, int]
    ) -> pd.DataFrame:
        """
        建玉残高疑似特徴量を生成

        Args:
            df: 価格データ
            lookback_periods: 計算期間設定

        Returns:
            疑似特徴量が追加されたDataFrame
        """
        try:
            result_df = df.copy()

            # ボリュームベースの疑似建玉残高
            pseudo_oi = result_df["volume"].rolling(24).mean() * 10
            # 明示的にpandas Seriesであることを保証
            pseudo_oi = pd.Series(pseudo_oi, index=result_df.index)

            # OI特徴量を生成（削除された特徴量を除く）
            # Removed: OI_Change_Rate, OI_Surge, OI_Price_Correlation
            # (低寄与度: 2025-01-05)
            result_df["OI_Change_Rate_24h"] = pseudo_oi.pct_change(24)

            # ボラティリティ調整建玉残高
            volatility = result_df["close"].pct_change().rolling(24).std()
            result_df["Volatility_Adjusted_OI"] = pseudo_oi / (volatility + 1e-8)

            # OI移動平均（中間計算用）
            oi_ma_24 = pseudo_oi.rolling(24).mean()
            oi_ma_168 = pseudo_oi.rolling(168).mean()

            # OIトレンド
            result_df["OI_Trend"] = (oi_ma_24 / oi_ma_168).fillna(1.0) - 1

            # OI正規化
            result_df["OI_Normalized"] = (
                pseudo_oi - pseudo_oi.rolling(168).mean()
            ) / pseudo_oi.rolling(168).std()

            # NaN値を0で補完
            remaining_oi_columns = [
                "OI_Change_Rate_24h",
                "Volatility_Adjusted_OI",
                "OI_Trend",
                "OI_Normalized",
            ]
            for col in remaining_oi_columns:
                if col in result_df.columns:
                    result_df[col] = result_df[col].fillna(0)

            logger.info("建玉残高疑似特徴量を生成しました")
            return result_df

        except Exception as e:
            logger.error(f"建玉残高疑似特徴量生成エラー: {e}")
            return df

    def cleanup_resources(self):
        """リソースのクリーンアップ"""
        try:
            logger.info("FeatureEngineeringServiceのリソースクリーンアップを開始")

            # キャッシュをクリア
            self.clear_cache()

            logger.info("FeatureEngineeringServiceのリソースクリーンアップ完了")

        except Exception as e:
            logger.error(f"FeatureEngineeringServiceクリーンアップエラー: {e}")
