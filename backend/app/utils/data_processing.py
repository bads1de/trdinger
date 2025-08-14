"""
データ処理ユーティリティ

data_cleaning_utils.py と data_preprocessing.py を統合したモジュール。
データ補間、クリーニング、前処理、最適化のロジックを統一的に提供します。
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    LabelEncoder,
    OneHotEncoder,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.base import BaseEstimator, TransformerMixin

from .unified_error_handler import UnifiedDataError

logger = logging.getLogger(__name__)


class OutlierRemovalTransformer(BaseEstimator, TransformerMixin):
    """
    scikit-learnの標準外れ値検出アルゴリズムを使用したTransformer

    IsolationForestやLocalOutlierFactorを活用し、Pipeline内で使用可能。
    """

    def __init__(self, method="isolation_forest", contamination=0.1, **kwargs):
        """
        Args:
            method: 外れ値検出方法 ('isolation_forest', 'local_outlier_factor')
            contamination: 外れ値の割合
            **kwargs: 各アルゴリズム固有のパラメータ
        """
        self.method = method
        self.contamination = contamination
        self.kwargs = kwargs
        self.detector = None
        self.outlier_mask_ = None

    def fit(self, X, y=None):
        """外れ値検出器をフィット"""
        if self.method == "isolation_forest":
            self.detector = IsolationForest(
                contamination=self.contamination, random_state=42, **self.kwargs
            )
            self.detector.fit(X)
            # 外れ値マスクを計算（-1が外れ値、1が正常値）
            predictions = self.detector.predict(X)
            self.outlier_mask_ = predictions == -1

        elif self.method == "local_outlier_factor":
            self.detector = LocalOutlierFactor(
                contamination=self.contamination, **self.kwargs
            )
            # LOFは fit_predict を使用
            predictions = self.detector.fit_predict(X)
            self.outlier_mask_ = predictions == -1

        else:
            raise ValueError(f"未対応の外れ値検出方法: {self.method}")

        return self

    def transform(self, X):
        """外れ値をNaNに置き換え"""
        if self.detector is None:
            raise ValueError("fit()を先に実行してください")

        X_transformed = X.copy()

        if self.method == "isolation_forest":
            # 新しいデータに対する予測
            predictions = self.detector.predict(X_transformed)
            outlier_mask = predictions == -1
        elif self.method == "local_outlier_factor":
            # LOFは学習データでのみ外れ値マスクを使用
            if hasattr(self, "outlier_mask_") and len(X_transformed) == len(
                self.outlier_mask_
            ):
                outlier_mask = self.outlier_mask_
            else:
                # 新しいデータの場合は変換しない
                return X_transformed

        # 外れ値をNaNに置き換え
        if isinstance(X_transformed, pd.DataFrame):
            X_transformed.loc[outlier_mask] = np.nan
        else:
            X_transformed[outlier_mask] = np.nan

        return X_transformed


class CategoricalEncoderTransformer(BaseEstimator, TransformerMixin):
    """
    カテゴリカル変数エンコーディング用のTransformer

    scikit-learnの標準エンコーダーを使用し、Pipeline内で利用可能。
    """

    def __init__(self, encoding_type="label", handle_unknown="ignore"):
        """
        Args:
            encoding_type: エンコーディング方法 ('label', 'onehot')
            handle_unknown: 未知のカテゴリの処理方法
        """
        self.encoding_type = encoding_type
        self.handle_unknown = handle_unknown
        self.encoders_ = {}
        self.feature_names_out_ = None

    def fit(self, X, y=None):
        """エンコーダーをフィット"""
        if isinstance(X, pd.DataFrame):
            categorical_columns = X.select_dtypes(
                include=["object", "category"]
            ).columns
        else:
            # numpy配列の場合は全列をカテゴリカルとして扱う
            categorical_columns = range(X.shape[1])

        for col in categorical_columns:
            if self.encoding_type == "label":
                encoder = LabelEncoder()
                if isinstance(X, pd.DataFrame):
                    # 欠損値を文字列で埋める
                    data = X[col].fillna("Unknown").astype(str)
                else:
                    data = X[:, col]
                encoder.fit(data)
                self.encoders_[col] = encoder

            elif self.encoding_type == "onehot":
                encoder = OneHotEncoder(
                    handle_unknown=self.handle_unknown, sparse_output=False
                )
                if isinstance(X, pd.DataFrame):
                    data = X[[col]].fillna("Unknown").astype(str)
                else:
                    data = X[:, [col]]
                encoder.fit(data)
                self.encoders_[col] = encoder

        return self

    def transform(self, X):
        """カテゴリカル変数をエンコード"""
        if not self.encoders_:
            return X

        X_transformed = X.copy()

        for col, encoder in self.encoders_.items():
            try:
                if isinstance(X_transformed, pd.DataFrame):
                    data = X_transformed[col].fillna("Unknown").astype(str)
                    if self.encoding_type == "label":
                        # 未知のカテゴリを処理
                        known_classes = set(encoder.classes_)
                        data = data.apply(
                            lambda x: x if x in known_classes else "Unknown"
                        )
                        X_transformed[col] = encoder.transform(data)
                    elif self.encoding_type == "onehot":
                        encoded = encoder.transform(data.values.reshape(-1, 1))
                        # OneHotの結果を元のDataFrameに統合
                        feature_names = [
                            f"{col}_{cls}" for cls in encoder.categories_[0]
                        ]
                        encoded_df = pd.DataFrame(
                            encoded, columns=feature_names, index=X_transformed.index
                        )
                        X_transformed = X_transformed.drop(columns=[col])
                        X_transformed = pd.concat([X_transformed, encoded_df], axis=1)
                else:
                    # numpy配列の場合
                    if self.encoding_type == "label":
                        X_transformed[:, col] = encoder.transform(X_transformed[:, col])

            except Exception as e:
                logger.warning(f"カラム {col} のエンコーディングでエラー: {e}")
                if isinstance(X_transformed, pd.DataFrame):
                    X_transformed[col] = 0  # デフォルト値
                else:
                    X_transformed[:, col] = 0

        return X_transformed


# 推奨される新しいユーティリティ関数
def create_outlier_removal_pipeline(
    method="isolation_forest", contamination=0.1, **kwargs
) -> Pipeline:
    """
    外れ値除去用のPipelineを作成（推奨アプローチ）

    Args:
        method: 外れ値検出方法 ('isolation_forest', 'local_outlier_factor')
        contamination: 外れ値の割合
        **kwargs: 各アルゴリズム固有のパラメータ

    Returns:
        外れ値除去Pipeline
    """
    return Pipeline(
        [
            (
                "outlier_removal",
                OutlierRemovalTransformer(
                    method=method, contamination=contamination, **kwargs
                ),
            ),
            ("imputer", SimpleImputer(strategy="median")),  # 外れ値除去後の欠損値補完
        ]
    )


def create_categorical_encoding_pipeline(
    encoding_type="label", handle_unknown="ignore"
) -> Pipeline:
    """
    カテゴリカルエンコーディング用のPipelineを作成（推奨アプローチ）

    Args:
        encoding_type: エンコーディング方法 ('label', 'onehot')
        handle_unknown: 未知のカテゴリの処理方法

    Returns:
        カテゴリカルエンコーディングPipeline
    """
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            (
                "encoder",
                CategoricalEncoderTransformer(
                    encoding_type=encoding_type, handle_unknown=handle_unknown
                ),
            ),
        ]
    )


def create_comprehensive_preprocessing_pipeline(
    outlier_method="isolation_forest",
    outlier_contamination=0.1,
    categorical_encoding="label",
    scaling_method="robust",
) -> Pipeline:
    """
    包括的な前処理Pipelineを作成（推奨アプローチ）

    scikit-learnの標準機能を活用した効率的で保守性の高い実装。

    Args:
        outlier_method: 外れ値検出方法
        outlier_contamination: 外れ値の割合
        categorical_encoding: カテゴリカルエンコーディング方法
        scaling_method: スケーリング方法

    Returns:
        包括的な前処理Pipeline
    """
    # 数値カラム用の前処理
    numeric_pipeline = Pipeline(
        [
            (
                "outlier_removal",
                OutlierRemovalTransformer(
                    method=outlier_method, contamination=outlier_contamination
                ),
            ),
            ("imputer", SimpleImputer(strategy="median")),
            (
                "scaler",
                RobustScaler() if scaling_method == "robust" else StandardScaler(),
            ),
        ]
    )

    # カテゴリカルカラム用の前処理
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            (
                "encoder",
                CategoricalEncoderTransformer(encoding_type=categorical_encoding),
            ),
        ]
    )

    # ColumnTransformerで統合
    preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                numeric_pipeline,
                make_column_selector(dtype_include=np.number),
            ),
            (
                "categorical",
                categorical_pipeline,
                make_column_selector(dtype_include=object),
            ),
        ],
        remainder="passthrough",
    )

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "final_cleanup",
                FunctionTransformer(
                    func=lambda X: np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0),
                    validate=False,
                ),
            ),
        ]
    )


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
        self.preprocessing_pipeline = None  # Pipeline-based前処理パイプライン
        self.fitted_pipelines = {}  # 用途別のfittedパイプライン

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

    def create_preprocessing_pipeline(
        self,
        numeric_strategy: str = "median",
        categorical_strategy: str = "most_frequent",
        scaling_method: str = "robust",
        remove_outliers: bool = True,
        outlier_threshold: float = 3.0,
        outlier_method: str = "iqr",
        outlier_transform: Optional[str] = None,  # 'robust' | 'quantile' | 'power'
    ) -> Pipeline:
        """
        scikit-learnのPipelineとColumnTransformerを使った宣言的な前処理パイプライン作成

        レポート3.6で指摘された問題を解決：
        - 独立した前処理関数を統合
        - 処理順序の明確化
        - カラム管理の簡素化
        - 宣言的で見通しの良い実装

        Args:
            numeric_strategy: 数値カラムの欠損値補完戦略
            categorical_strategy: カテゴリカルカラムの欠損値補完戦略
            scaling_method: スケーリング方法
            remove_outliers: 外れ値除去を行うか
            outlier_threshold: 外れ値の閾値
            outlier_method: 外れ値検出方法

        Returns:
            前処理パイプライン
        """
        # 数値カラム用の前処理パイプライン
        numeric_pipeline_steps = []

        # 1. 無限値をNaNに変換
        numeric_pipeline_steps.append(
            (
                "inf_to_nan",
                FunctionTransformer(
                    func=lambda X: np.where(np.isinf(X), np.nan, X), validate=False
                ),
            )
        )

        # 2. 外れ値除去（オプション）- scikit-learn標準アルゴリズムを使用
        if remove_outliers:
            if outlier_method == "isolation_forest":
                outlier_transformer = OutlierRemovalTransformer(
                    method="isolation_forest",
                    contamination=(
                        outlier_threshold / 100.0
                        if outlier_threshold > 1
                        else outlier_threshold
                    ),
                    n_estimators=100,
                )
            elif outlier_method == "local_outlier_factor":
                outlier_transformer = OutlierRemovalTransformer(
                    method="local_outlier_factor",
                    contamination=(
                        outlier_threshold / 100.0
                        if outlier_threshold > 1
                        else outlier_threshold
                    ),
                    n_neighbors=20,
                )
            elif outlier_method in ("iqr", "zscore"):
                outlier_transformer = OutlierRemovalTransformer(
                    method="isolation_forest",
                    contamination=(
                        outlier_threshold / 100.0
                        if outlier_threshold > 1
                        else outlier_threshold
                    ),
                    n_estimators=100,
                )
            else:
                outlier_transformer = FunctionTransformer(
                    func=lambda X: X, validate=False
                )

            numeric_pipeline_steps.append(("outlier_removal", outlier_transformer))

        # 3. 欠損値補完
        numeric_pipeline_steps.append(
            ("imputer", SimpleImputer(strategy=numeric_strategy))
        )

        # 4. スケーリング（外れ値に強い変換への差し替えも可能）
        if outlier_transform == "robust":
            # 中央値/IQRで頑健にスケーリング
            scaler = RobustScaler()
        elif outlier_transform == "quantile":
            from sklearn.preprocessing import QuantileTransformer

            # ランク変換して外れ値の影響を緩和（正規分布にマップ）
            scaler = QuantileTransformer(output_distribution="normal", random_state=42)
        elif outlier_transform == "power":
            from sklearn.preprocessing import PowerTransformer

            # 歪度の補正（Yeo-Johnsonは負値対応）
            scaler = PowerTransformer(method="yeo-johnson", standardize=True)
        else:
            # 既存のscaling_methodに従う
            if scaling_method == "standard":
                scaler = StandardScaler()
            elif scaling_method == "robust":
                scaler = RobustScaler()
            elif scaling_method == "minmax":
                scaler = MinMaxScaler()
            else:
                scaler = StandardScaler()  # デフォルト

        numeric_pipeline_steps.append(("scaler", scaler))

        # 数値パイプラインを作成
        numeric_pipeline = Pipeline(numeric_pipeline_steps)

        # カテゴリカルカラム用の前処理パイプライン - scikit-learn標準エンコーダーを使用
        categorical_pipeline = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy=categorical_strategy, fill_value="Unknown"),
                ),
                (
                    "encoder",
                    CategoricalEncoderTransformer(encoding_type="label"),
                ),
            ]
        )

        # ColumnTransformerで数値とカテゴリカルを統合
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    numeric_pipeline,
                    make_column_selector(dtype_include=np.number),
                ),
                (
                    "categorical",
                    categorical_pipeline,
                    make_column_selector(dtype_include=object),
                ),
            ],
            remainder="passthrough",  # その他のカラムはそのまま通す
            verbose_feature_names_out=False,
        )

        # 最終的なパイプライン
        pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "final_cleanup",
                    FunctionTransformer(func=self._final_cleanup, validate=False),
                ),
            ]
        )

        return pipeline

    def _final_cleanup(self, X):
        """最終的なクリーンアップ"""
        # 残っているNaNを0で埋める
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

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
        numeric_columns = (
            result_df[valid_columns].select_dtypes(include=[np.number]).columns.tolist()
        )
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
                outlier_mask = (result_df[numeric_columns] < lower_bounds) | (
                    result_df[numeric_columns] > upper_bounds
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
                logger.info(
                    f"外れ値除去完了: {total_outliers}個の外れ値を除去 ({method})"
                )

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

        # 2. カテゴリカル変数のエンコーディング
        result_df = self._encode_categorical_variables(result_df)

        # 3. 数値カラムを特定（カテゴリカル変数エンコーディング後）
        numeric_columns = result_df.select_dtypes(include=[np.number]).columns.tolist()

        # 4. 外れ値除去（オプション）
        if remove_outliers:
            result_df = self._remove_outliers(
                result_df, numeric_columns, outlier_threshold, method=outlier_method
            )

        # 5. 欠損値補完
        result_df = self.transform_missing_values(
            result_df, strategy=imputation_strategy, columns=numeric_columns
        )

        # 6. 特徴量スケーリング（オプション）
        if scale_features:
            result_df = self._scale_features(
                result_df, numeric_columns, method=scaling_method
            )

        logger.info("特徴量の包括的前処理が完了")
        return result_df

    def _encode_categorical_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        カテゴリカル変数を数値にエンコーディング（非推奨）

        注意: この方法は非推奨です。代わりにCategoricalEncoderTransformerを使用してください。

        Args:
            df: 対象のDataFrame

        Returns:
            エンコーディング済みのDataFrame
        """
        # 新しいTransformerベースのエンコーディングを使用する実装に置き換え
        try:
            result_df = df.copy()

            categorical_columns = result_df.select_dtypes(
                include=["object", "string", "category"]
            ).columns.tolist()

            if not categorical_columns:
                return result_df

            logger.info(
                f"カテゴリカル変数をエンコーディング (Transformer使用): {categorical_columns}"
            )

            # パイプラインを作成して適用
            pipeline = create_categorical_encoding_pipeline(encoding_type="label")
            encoded = pipeline.fit_transform(result_df[categorical_columns])

            if isinstance(encoded, pd.DataFrame):
                encoded_df = encoded
            else:
                # numpy配列の場合はエンコーダから特徴名を組み立てる
                try:
                    encoder = pipeline.named_steps["encoder"]
                    if getattr(encoder, "encoding_type", None) == "onehot":
                        feature_names = []
                        for col, enc in encoder.encoders_.items():
                            cats = getattr(enc, "categories_", [[]])[0]
                            for cls in cats:
                                feature_names.append(f"{col}_{cls}")
                    else:
                        feature_names = [str(col) for col in categorical_columns]
                except Exception:
                    feature_names = [f"cat_{i}" for i in range(encoded.shape[1])]

                encoded_df = pd.DataFrame(
                    encoded, columns=feature_names, index=result_df.index
                )

            # 元のカテゴリカルカラムを削除して結合
            result_df = result_df.drop(columns=categorical_columns)
            result_df = pd.concat([result_df, encoded_df], axis=1)

            logger.info("カテゴリカル変数のエンコーディングが完了 (Transformer使用)")
            return result_df

        except Exception as e:
            logger.error(f"カテゴリカル変数エンコーディングエラー: {e}")
            return df

    def _encode_fear_greed_classification(
        self, df: pd.DataFrame, col: str
    ) -> pd.DataFrame:
        """
        Fear & Greed Classification の特別なエンコーディング

        Args:
            df: 対象のDataFrame
            col: カラム名

        Returns:
            エンコーディング済みのDataFrame
        """
        result_df = df.copy()

        # Fear & Greed Classification の標準的なマッピング
        fg_mapping = {
            "Extreme Fear": 0,
            "Fear": 1,
            "Neutral": 2,
            "Greed": 3,
            "Extreme Greed": 4,
        }

        # 欠損値を 'Neutral' で埋める
        result_df[col] = result_df[col].fillna("Neutral")

        # マッピングを適用
        result_df[col + "_encoded"] = result_df[col].map(fg_mapping)

        # マッピングできなかった値は 'Neutral' (2) にする
        result_df[col + "_encoded"] = result_df[col + "_encoded"].fillna(2)

        # 元のカラムを削除
        result_df = result_df.drop(columns=[col])

        # エンコード済みカラムの名前を元の名前に変更
        result_df = result_df.rename(columns={col + "_encoded": col})

        logger.info(f"Fear & Greed Classification エンコーディング完了: {fg_mapping}")
        return result_df

    def _encode_general_categorical(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """
        一般的なカテゴリカル変数のエンコーディング

        Args:
            df: 対象のDataFrame
            col: カラム名

        Returns:
            エンコーディング済みのDataFrame
        """
        from sklearn.preprocessing import LabelEncoder

        result_df = df.copy()

        # 欠損値を文字列で埋める
        result_df[col] = result_df[col].fillna("Unknown")

        # LabelEncoderを使用
        le = LabelEncoder()
        result_df[col] = le.fit_transform(result_df[col].astype(str))

        logger.info(f"一般カテゴリカル変数エンコーディング完了: {col}")
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
            logger.info(
                f"入力データサイズ: {len(features_df)}行, {len(features_df.columns)}列"
            )

            # 1. 入力データの基本検証
            if features_df is None or features_df.empty:
                raise ValueError("入力特徴量データが空です")

            # 2. 特徴量の前処理
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

            logger.info(
                f"前処理後データサイズ: {len(features_processed)}行, {len(features_processed.columns)}列"
            )

            # 3. ラベル生成のための価格データを取得
            if "Close" not in features_processed.columns:
                logger.error(f"利用可能なカラム: {list(features_processed.columns)}")
                raise ValueError("Close価格データが特徴量に含まれていません")

            price_data = features_processed["Close"]
            logger.info(f"価格データサイズ: {len(price_data)}行")

            # 4. ラベル生成
            logger.info("ラベル生成を実行中...")
            try:
                labels, threshold_info = label_generator.generate_dynamic_labels(
                    price_data, **training_params
                )
                logger.info(f"ラベル生成完了: {len(labels)}行")
            except Exception as label_error:
                logger.error(f"ラベル生成エラー: {label_error}")
                raise ValueError(f"ラベル生成に失敗しました: {label_error}")

            # 5. 特徴量とラベルのインデックスを整合
            logger.info("データの整合性を確保中...")

            common_index = features_processed.index.intersection(labels.index)

            if len(common_index) == 0:
                logger.error("特徴量とラベルに共通のインデックスがありません")
                logger.error(
                    f"特徴量インデックス: {features_processed.index[:10].tolist()}..."
                )
                logger.error(f"ラベルインデックス: {labels.index[:10].tolist()}...")
                raise ValueError("特徴量とラベルに共通のインデックスがありません")

            features_clean = features_processed.loc[common_index]
            labels_clean = labels.loc[common_index]

            # 6. NaNを含む行を除去
            logger.info("NaN値の除去を実行中...")
            valid_mask = features_clean.notna().all(axis=1) & labels_clean.notna()
            features_clean = features_clean[valid_mask]
            labels_clean = labels_clean[valid_mask]

            # 7. 最終的なデータ検証
            if len(features_clean) == 0 or len(labels_clean) == 0:
                logger.error("有効な学習データが存在しません")
                logger.error(f"最終的な特徴量サイズ: {len(features_clean)}")
                logger.error(f"最終的なラベルサイズ: {len(labels_clean)}")
                raise ValueError("有効な学習データが存在しません")

            if len(features_clean) != len(labels_clean):
                raise ValueError("特徴量とラベルの長さが一致しません")

            return features_clean, labels_clean, threshold_info

        except Exception as e:
            logger.error(f"学習用データの準備でエラーが発生: {e}")
            raise

    def preprocess_with_pipeline(
        self,
        df: pd.DataFrame,
        pipeline_name: str = "default",
        fit_pipeline: bool = True,
        **pipeline_params,
    ) -> pd.DataFrame:
        """
        Pipelineベースの前処理実行

        レポート3.6の改善：宣言的で見通しの良い前処理実装

        Args:
            df: 対象DataFrame
            pipeline_name: パイプライン名（キャッシュ用）
            fit_pipeline: パイプラインをfitするか（Falseの場合は既存のfittedパイプラインを使用）
            **pipeline_params: パイプライン作成パラメータ

        Returns:
            前処理されたDataFrame
        """
        try:
            logger.info(f"Pipelineベース前処理開始: {pipeline_name}")

            if fit_pipeline or pipeline_name not in self.fitted_pipelines:
                # 新しいパイプラインを作成
                pipeline = self.create_preprocessing_pipeline(**pipeline_params)

                # パイプラインをfitして保存
                logger.info("パイプラインをfitting中...")
                fitted_pipeline = pipeline.fit(df)
                self.fitted_pipelines[pipeline_name] = fitted_pipeline

                logger.info(f"パイプライン '{pipeline_name}' をfitして保存しました")
            else:
                # 既存のfittedパイプラインを使用
                fitted_pipeline = self.fitted_pipelines[pipeline_name]
                logger.info(f"既存のパイプライン '{pipeline_name}' を使用")

            # 変換実行
            logger.info("データ変換実行中...")
            transformed_data = fitted_pipeline.transform(df)

            # 結果をDataFrameに変換
            if hasattr(transformed_data, "toarray"):
                # sparse matrixの場合
                transformed_data = transformed_data.toarray()

            # カラム名を生成
            try:
                feature_names = fitted_pipeline.get_feature_names_out()
            except Exception:
                # feature名が取得できない場合は自動生成
                feature_names = [
                    f"feature_{i}" for i in range(transformed_data.shape[1])
                ]

            result_df = pd.DataFrame(
                transformed_data, index=df.index, columns=feature_names
            )

            logger.info(
                f"Pipeline前処理完了: {len(result_df)}行, {len(result_df.columns)}列"
            )
            return result_df

        except Exception as e:
            logger.error(f"Pipeline前処理エラー: {e}")
            # フォールバック：従来の方法
            logger.warning("従来の前処理方法にフォールバック")
            return self.preprocess_features(df, **pipeline_params)

    def create_ml_preprocessing_pipeline(
        self,
        target_column: str = "Close",
        feature_selection: bool = False,
        n_features: Optional[int] = None,
    ) -> Pipeline:
        """
        機械学習用の特化したパイプライン作成

        Args:
            target_column: ターゲットカラム名
            feature_selection: 特徴選択を行うか
            n_features: 選択する特徴数

        Returns:
            ML用前処理パイプライン
        """
        from sklearn.feature_selection import SelectKBest, f_regression

        # 基本的な前処理パイプライン
        base_pipeline = self.create_preprocessing_pipeline(
            scaling_method="robust",  # MLには頑健なスケーリングを使用
            remove_outliers=True,
            outlier_method="iqr",
        )

        steps = [("base_preprocessing", base_pipeline)]

        # 特徴選択（オプション）
        if feature_selection and n_features:
            steps.append(
                (
                    "feature_selection",
                    SelectKBest(score_func=f_regression, k=n_features),
                )
            )

        return Pipeline(steps)

    def get_pipeline_info(self, pipeline_name: str) -> Dict[str, Any]:
        """
        パイプライン情報を取得

        Args:
            pipeline_name: パイプライン名

        Returns:
            パイプライン情報の辞書
        """
        if pipeline_name not in self.fitted_pipelines:
            return {"exists": False}

        pipeline = self.fitted_pipelines[pipeline_name]

        info = {
            "exists": True,
            "steps": [step[0] for step in pipeline.steps],
            "n_features_in": getattr(pipeline, "n_features_in_", None),
            "feature_names_in": getattr(pipeline, "feature_names_in_", None),
        }

        try:
            info["feature_names_out"] = pipeline.get_feature_names_out().tolist()
        except Exception:
            info["feature_names_out"] = None

        return info

    def clear_cache(self):
        """キャッシュをクリア"""
        self.imputers.clear()
        self.scalers.clear()
        self.imputation_stats.clear()
        self.fitted_pipelines.clear()  # Pipelineキャッシュもクリア
        logger.info("DataProcessorのキャッシュをクリアしました")


# グローバルインスタンス（後方互換性のため）
data_processor = DataProcessor()

# 後方互換性のためのエイリアス
DataCleaner = DataProcessor
data_preprocessor = data_processor
