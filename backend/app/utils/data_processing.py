"""
データ処理ユーティリティ

data_cleaning_utils.py と data_preprocessing.py を統合したモジュール。
データ補間、クリーニング、前処理、最適化のロジックを統一的に提供します。
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from .unified_error_handler import UnifiedDataError

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    統合データ処理クラス

    データクリーニング、前処理、補間、最適化を統一的に処理します。
    """

    def __init__(self):
        """初期化"""
        self.imputers = {}  # カラムごとのimputer
        self.scalers = {}  # カラムごとのscaler
        self.imputation_stats = {}  # 補完統計情報

    def interpolate_oi_fr_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        OI/FRデータの補間処理

        Args:
            df: 対象のDataFrame

        Returns:
            補間処理されたDataFrame
        """
        result_df = df.copy()

        # 共通補間ヘルパーで OI / FR を処理（ffill → 統計補完）
        target_cols = [
            c for c in ["open_interest", "funding_rate"] if c in result_df.columns
        ]
        if target_cols:
            result_df = self.interpolate_columns(
                result_df,
                columns=target_cols,
                strategy="median",
                forward_fill=True,
                dtype="float64",
                default_fill_values=None,
                fit_if_needed=True,
            )

        return result_df

    def interpolate_fear_greed_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fear & Greedデータの補間処理

        Args:
            df: 対象のDataFrame

        Returns:
            補間処理されたDataFrame
        """
        result_df = df.copy()

        # Fear & Greed 値: 数値列のため統計補完
        if "fear_greed_value" in result_df.columns:
            result_df = self.interpolate_columns(
                result_df,
                columns=["fear_greed_value"],
                strategy="median",
                forward_fill=True,
                dtype="float64",
                default_fill_values={"fear_greed_value": 50.0},
                fit_if_needed=True,
            )

        # Fear & Greed クラス列: 文字列列のため簡易前方補完＋既定値
        if "fear_greed_classification" in result_df.columns:
            fg_class_series = result_df["fear_greed_classification"].replace(
                {pd.NA: None}
            )
            fg_class_series = fg_class_series.astype("string")
            result_df["fear_greed_classification"] = fg_class_series.ffill().fillna(
                "Neutral"
            )

        return result_df

    def interpolate_all_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        全データの補間処理

        Args:
            df: 対象のDataFrame

        Returns:
            補間処理されたDataFrame
        """
        logger.info("データ補間処理を開始")

        result_df = df.copy()

        # OI/FRデータの補間
        result_df = self.interpolate_oi_fr_data(result_df)

        # Fear & Greedデータの補間
        result_df = self.interpolate_fear_greed_data(result_df)

        logger.info("データ補間処理が完了")
        return result_df

    def optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        データ型の最適化

        Args:
            df: 対象のDataFrame

        Returns:
            最適化されたDataFrame
        """
        result_df = df.copy()

        for col in result_df.columns:
            if result_df[col].dtype == "object":
                continue

            # 数値型の最適化
            if pd.api.types.is_numeric_dtype(result_df[col]):
                # 整数型の場合
                if pd.api.types.is_integer_dtype(result_df[col]):
                    col_min = result_df[col].min()
                    col_max = result_df[col].max()

                    if col_min >= 0:
                        if col_max < 255:
                            result_df[col] = result_df[col].astype("uint8")
                        elif col_max < 65535:
                            result_df[col] = result_df[col].astype("uint16")
                        elif col_max < 4294967295:
                            result_df[col] = result_df[col].astype("uint32")
                    else:
                        if col_min > -128 and col_max < 127:
                            result_df[col] = result_df[col].astype("int8")
                        elif col_min > -32768 and col_max < 32767:
                            result_df[col] = result_df[col].astype("int16")
                        elif col_min > -2147483648 and col_max < 2147483647:
                            result_df[col] = result_df[col].astype("int32")

                # 浮動小数点型の場合
                elif pd.api.types.is_float_dtype(result_df[col]):
                    result_df[col] = pd.to_numeric(result_df[col], downcast="float")

        return result_df

    def validate_ohlcv_data(self, df: pd.DataFrame) -> None:
        """
        OHLCVデータの基本検証

        Args:
            df: 検証対象のDataFrame

        Raises:
            ValueError: データが無効な場合
        """
        if df.empty:
            raise ValueError("DataFrameが空です")

        # 必要な列の存在確認
        required_columns = ["Open", "High", "Low", "Close"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"必要な列が不足しています: {missing_columns}")

        # 価格の論理的整合性チェック
        for idx in df.index:
            row = df.loc[idx]
            if pd.isna(row[["Open", "High", "Low", "Close"]]).any():
                continue

            high = row["High"]
            low = row["Low"]
            open_price = row["Open"]
            close = row["Close"]

            # High >= Low の確認
            if high < low:
                logger.warning(f"インデックス {idx}: High ({high}) < Low ({low})")

            # High >= Open, Close の確認
            if high < open_price or high < close:
                logger.warning(f"インデックス {idx}: High価格が Open/Close より低い")

            # Low <= Open, Close の確認
            if low > open_price or low > close:
                logger.warning(f"インデックス {idx}: Low価格が Open/Close より高い")

    def validate_extended_data(
        self, df: pd.DataFrame, required_columns: List[str]
    ) -> None:
        """
        拡張データの整合性をチェック

        Args:
            df: 検証対象のDataFrame
            required_columns: 必須カラムのリスト

        Raises:
            ValueError: DataFrameが無効な場合
        """
        # 基本的なOHLCVデータの検証
        self.validate_ohlcv_data(df)

        # 追加カラムの存在確認
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"必須カラムが不足しています: {missing_columns}")

        # 重複インデックスのチェック
        if df.index.duplicated().any():
            logger.warning("重複したタイムスタンプが検出されました。")

        # ソート確認
        if not df.index.is_monotonic_increasing:
            logger.warning("インデックスが時系列順にソートされていません。")

    def clean_and_validate_data(
        self,
        df: pd.DataFrame,
        required_columns: List[str],
        interpolate: bool = True,
        optimize: bool = True,
    ) -> pd.DataFrame:
        """
        データのクリーニングと検証を一括実行

        Args:
            df: 対象のDataFrame
            required_columns: 必須カラムのリスト
            interpolate: 補間処理を実行するか
            optimize: データ型最適化を実行するか

        Returns:
            クリーニング済みのDataFrame
        """
        result_df = df.copy()

        # データ補間
        if interpolate:
            result_df = self.interpolate_all_data(result_df)

        # データ型最適化
        if optimize:
            result_df = self.optimize_dtypes(result_df)

        # データ検証
        self.validate_extended_data(result_df, required_columns)

        # 時系列順にソート
        result_df = result_df.sort_index()

        return result_df

    def transform_missing_values(
        self,
        df: pd.DataFrame,
        strategy: str = "median",
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        欠損値を統計的手法で補完

        Args:
            df: 対象DataFrame
            strategy: 補完戦略 ('mean', 'median', 'most_frequent', 'constant')
            columns: 対象カラム（Noneの場合は数値カラム全て）

        Returns:
            補完されたDataFrame
        """
        if df is None or df.empty:
            return df

        result_df = df.copy()
        target_columns = (
            columns or result_df.select_dtypes(include=[np.number]).columns.tolist()
        )

        if not target_columns:
            logger.warning("補完対象の数値カラムが見つかりません")
            return result_df

        try:
            # カラムごとに個別のImputerを使用
            for col in target_columns:
                if col not in result_df.columns:
                    continue

                col_data = result_df[col]
                missing_count = col_data.isna().sum()
                valid_count = col_data.notna().sum()

                if missing_count == 0:
                    continue  # 欠損値がない場合はスキップ

                if valid_count == 0:
                    # 全て欠損値の場合はデフォルト値で埋める
                    logger.warning(
                        f"カラム {col}: 全ての値が欠損値のため、デフォルト値0で補完します"
                    )
                    result_df[col] = 0.0
                    self.imputation_stats[col] = {
                        "strategy": "default_zero",
                        "missing_count": missing_count,
                        "fill_value": 0.0,
                    }
                    continue

                # カラム用のImputerを取得または作成
                imputer_key = f"{col}_{strategy}"
                if imputer_key not in self.imputers:
                    self.imputers[imputer_key] = SimpleImputer(strategy=strategy)

                # 2次元配列に変換してfit_transform
                col_values = col_data.values.reshape(-1, 1)
                imputed_values = self.imputers[imputer_key].fit_transform(col_values)
                result_df[col] = imputed_values.flatten()

                # 統計情報を記録
                self.imputation_stats[col] = {
                    "strategy": strategy,
                    "missing_count": missing_count,
                    "fill_value": (
                        self.imputers[imputer_key].statistics_[0]
                        if hasattr(self.imputers[imputer_key], "statistics_")
                        else None
                    ),
                }

            logger.info(f"欠損値補完完了: {len(target_columns)}カラム, 戦略={strategy}")
            return result_df

        except Exception as e:
            logger.error(f"欠損値補完エラー: {e}")
            raise UnifiedDataError(f"欠損値補完に失敗しました: {e}")

    def _remove_outliers(
        self,
        df: pd.DataFrame,
        columns: List[str],
        threshold: float = 3.0,
        method: str = "iqr",
    ) -> pd.DataFrame:
        """
        外れ値を除去（最適化版）

        Args:
            df: 対象DataFrame
            columns: 対象カラム
            threshold: 閾値
            method: 検出方法 ('iqr', 'zscore')

        Returns:
            外れ値除去後のDataFrame
        """
        result_df = df.copy()

        # 存在するカラムのみをフィルタ
        valid_columns = [col for col in columns if col in result_df.columns]
        if not valid_columns:
            return result_df

        # 数値カラムのみを対象
        numeric_columns = result_df[valid_columns].select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_columns:
            return result_df

        try:
            if method == "iqr":
                # ベクトル化されたIQR計算
                Q1 = result_df[numeric_columns].quantile(0.25)
                Q3 = result_df[numeric_columns].quantile(0.75)
                IQR = Q3 - Q1
                lower_bounds = Q1 - threshold * IQR
                upper_bounds = Q3 + threshold * IQR

                # 一括で外れ値マスクを計算
                outlier_mask = (
                    (result_df[numeric_columns] < lower_bounds) |
                    (result_df[numeric_columns] > upper_bounds)
                )

            elif method == "zscore":
                # ベクトル化されたZ-score計算
                means = result_df[numeric_columns].mean()
                stds = result_df[numeric_columns].std()
                z_scores = np.abs((result_df[numeric_columns] - means) / stds)
                outlier_mask = z_scores > threshold

            else:
                logger.warning(f"未知の外れ値検出方法: {method}")
                return result_df

            # 一括で外れ値をNaNに設定
            total_outliers = 0
            for col in numeric_columns:
                if col in outlier_mask.columns:
                    col_outliers = outlier_mask[col].sum()
                    if col_outliers > 0:
                        result_df.loc[outlier_mask[col], col] = np.nan
                        total_outliers += col_outliers

            if total_outliers > 0:
                logger.info(f"外れ値除去完了: {total_outliers}個の外れ値を除去 ({method})")

        except Exception as e:
            logger.warning(f"外れ値除去でエラーが発生: {e}")
            return df

        return result_df

    def _scale_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        method: str = "standard",
    ) -> pd.DataFrame:
        """
        特徴量スケーリング

        Args:
            df: 対象DataFrame
            columns: 対象カラム
            method: スケーリング方法 ('standard', 'robust', 'minmax')

        Returns:
            スケーリング後のDataFrame
        """
        result_df = df.copy()

        if method == "standard":
            scaler = StandardScaler()
        elif method == "robust":
            scaler = RobustScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        else:
            logger.warning(f"未知のスケーリング方法: {method}")
            return result_df

        try:
            for col in columns:
                if col not in result_df.columns:
                    continue

                col_data = result_df[col].values.reshape(-1, 1)
                scaled_data = scaler.fit_transform(col_data)
                result_df[col] = scaled_data.flatten()

                # スケーラーを保存
                self.scalers[col] = scaler

            logger.info(f"特徴量スケーリング完了: {len(columns)}カラム, 方法={method}")
            return result_df

        except Exception as e:
            logger.error(f"特徴量スケーリングエラー: {e}")
            raise UnifiedDataError(f"特徴量スケーリングに失敗しました: {e}")

    def preprocess_features(
        self,
        df: pd.DataFrame,
        imputation_strategy: str = "median",
        scale_features: bool = False,
        remove_outliers: bool = True,
        outlier_threshold: float = 3.0,
        scaling_method: str = "standard",
        outlier_method: str = "iqr",
    ) -> pd.DataFrame:
        """
        特徴量の包括的前処理

        Args:
            df: 対象DataFrame
            imputation_strategy: 欠損値補完戦略
            scale_features: 特徴量スケーリングを行うか
            remove_outliers: 外れ値除去を行うか
            outlier_threshold: 外れ値の閾値
            scaling_method: スケーリング方法
            outlier_method: 外れ値検出方法

        Returns:
            前処理されたDataFrame
        """
        if df is None or df.empty:
            return df

        logger.info("特徴量の包括的前処理を開始")
        result_df = df.copy()

        # 1. 無限値をNaNに変換
        result_df = result_df.replace([np.inf, -np.inf], np.nan)

        # 2. 数値カラムを特定
        numeric_columns = result_df.select_dtypes(include=[np.number]).columns.tolist()

        # 3. 外れ値除去（オプション）
        if remove_outliers:
            result_df = self._remove_outliers(
                result_df, numeric_columns, outlier_threshold, method=outlier_method
            )

        # 4. 欠損値補完
        result_df = self.transform_missing_values(
            result_df, strategy=imputation_strategy, columns=numeric_columns
        )

        # 5. 特徴量スケーリング（オプション）
        if scale_features:
            result_df = self._scale_features(
                result_df, numeric_columns, method=scaling_method
            )

        logger.info("特徴量の包括的前処理が完了")
        return result_df

    def interpolate_columns(
        self,
        df: pd.DataFrame,
        columns: List[str],
        strategy: str = "median",
        forward_fill: bool = True,
        dtype: Optional[str] = None,
        default_fill_values: Optional[Dict[str, float]] = None,
        fit_if_needed: bool = True,
    ) -> pd.DataFrame:
        """
        指定されたカラムの補間処理

        Args:
            df: 対象DataFrame
            columns: 補間対象カラム
            strategy: 補間戦略
            forward_fill: 前方補完を行うか
            dtype: 変換後のデータ型
            default_fill_values: デフォルト補完値
            fit_if_needed: 必要に応じてImputerをfitするか

        Returns:
            補間されたDataFrame
        """
        result_df = df.copy()
        default_fill_values = default_fill_values or {}

        # まず型正規化と前方補完を行う
        for col in columns:
            if col not in result_df.columns:
                continue
            try:
                series = result_df[col]

                # pd.NA を None/NaN に正規化
                series = series.replace({pd.NA: None})

                # 数値型を想定する場合は to_numeric で強制変換
                if dtype is not None:
                    series = pd.to_numeric(series, errors="coerce")
                    try:
                        series = series.astype(dtype)
                    except Exception:
                        pass

                # 前方補完
                if forward_fill:
                    series = series.ffill()

                result_df[col] = series
            except Exception as e:
                logger.warning(f"列 {col} の前処理（型/ffill）でエラー: {e}")

        # 統計的補完を実行
        if fit_if_needed:
            result_df = self.transform_missing_values(
                result_df, strategy=strategy, columns=columns
            )

        # デフォルト値で最終補完
        for col in columns:
            if col in result_df.columns and col in default_fill_values:
                result_df[col] = result_df[col].fillna(default_fill_values[col])

        return result_df

    def prepare_training_data(
        self, features_df: pd.DataFrame, label_generator, **training_params
    ) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        """
        学習用データを準備

        Args:
            features_df: 特徴量DataFrame
            label_generator: ラベル生成器（LabelGeneratorWrapper）
            **training_params: 学習パラメータ

        Returns:
            features_clean: クリーンな特徴量DataFrame
            labels_clean: クリーンなラベルSeries
            threshold_info: 閾値情報の辞書
        """
        try:
            logger.info("学習用データの準備を開始")

            # 1. 特徴量の前処理
            logger.info("特徴量の前処理を実行中...")
            features_processed = self.preprocess_features(
                features_df,
                imputation_strategy=training_params.get(
                    "imputation_strategy", "median"
                ),
                scale_features=training_params.get("scale_features", True),
                remove_outliers=training_params.get("remove_outliers", True),
                outlier_threshold=training_params.get("outlier_threshold", 3.0),
                scaling_method=training_params.get("scaling_method", "robust"),
                outlier_method=training_params.get("outlier_method", "iqr"),
            )

            # 2. ラベル生成のための価格データを取得
            if "Close" not in features_processed.columns:
                raise ValueError("Close価格データが特徴量に含まれていません")

            price_data = features_processed["Close"]

            # 3. ラベル生成
            logger.info("ラベル生成を実行中...")
            labels, threshold_info = label_generator.generate_dynamic_labels(
                price_data, **training_params
            )

            # 4. 特徴量とラベルのインデックスを整合
            logger.info("データの整合性を確保中...")
            common_index = features_processed.index.intersection(labels.index)

            if len(common_index) == 0:
                raise ValueError("特徴量とラベルに共通のインデックスがありません")

            features_clean = features_processed.loc[common_index]
            labels_clean = labels.loc[common_index]

            # 5. NaNを含む行を除去
            valid_mask = features_clean.notna().all(axis=1) & labels_clean.notna()
            features_clean = features_clean[valid_mask]
            labels_clean = labels_clean[valid_mask]

            # 6. 最終的なデータ検証
            if len(features_clean) == 0 or len(labels_clean) == 0:
                raise ValueError("有効な学習データが存在しません")

            if len(features_clean) != len(labels_clean):
                raise ValueError("特徴量とラベルの長さが一致しません")

            logger.info(
                f"学習用データの準備が完了: {len(features_clean)}行, {len(features_clean.columns)}特徴量"
            )
            logger.info(f"ラベル分布: {labels_clean.value_counts().to_dict()}")

            return features_clean, labels_clean, threshold_info

        except Exception as e:
            logger.error(f"学習用データの準備でエラーが発生: {e}")
            raise

    def clear_cache(self):
        """キャッシュをクリア"""
        self.imputers.clear()
        self.scalers.clear()
        self.imputation_stats.clear()
        logger.info("DataProcessorのキャッシュをクリアしました")


# グローバルインスタンス（後方互換性のため）
data_processor = DataProcessor()

# 後方互換性のためのエイリアス
DataCleaner = DataProcessor
data_preprocessor = data_processor
