"""
データ処理ユーティリティ

scikit-learnの標準機能を最大限活用した効率的なデータ処理モジュール。
Pipeline、ColumnTransformer、標準Transformerを使用した現代的な実装。

推奨される使用方法:
- 新しいコード: preprocess_with_pipeline(), create_optimized_pipeline()
- 外れ値除去: OutlierRemovalTransformer, create_outlier_removal_pipeline()
- カテゴリカル変数: CategoricalEncoderTransformer, create_categorical_encoding_pipeline()
- 包括の前処理: create_comprehensive_preprocessing_pipeline()
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

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


logger = logging.getLogger(__name__)


class OutlierRemovalTransformer(BaseEstimator, TransformerMixin):
    """
    scikit-learnの標準外れ値検出アルゴリズムを使用したTransformer

    IsolationForestやLocalOutlierFactorを活用し、Pipeline内で使用可能。
    """

    def __init__(
        self,
        method: str = "isolation_forest",
        contamination: float = 0.1,
        **kwargs: Any,
    ) -> None:
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
                contamination=str(self.contamination), random_state=42, **self.kwargs
            )
            self.detector.fit(X)
            # 外れ値マスクを計算（-1が外れ値、1が正常値）
            predictions = self.detector.predict(X)
            self.outlier_mask_ = predictions == -1

        elif self.method == "local_outlier_factor":
            self.detector = LocalOutlierFactor(
                contamination=str(self.contamination), **self.kwargs
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
            if (
                hasattr(self, "outlier_mask_")
                and self.outlier_mask_ is not None
                and len(X_transformed) == len(self.outlier_mask_)
            ):
                # 学習時と同じデータサイズの場合、学習時の外れ値マスクを使用
                outlier_mask = self.outlier_mask_
            else:
                # 新しいデータの場合は変換しない（学習データのみで外れ値検出）
                return X_transformed

            # 外れ値をNaNに置き換え
            if isinstance(X_transformed, pd.DataFrame):
                # DataFrameの場合はlocで行を指定してNaNに置き換え
                X_transformed.loc[outlier_mask] = np.nan
            else:
                # numpy配列の場合は直接インデックス指定でNaNに置き換え
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
                        # 确保data.values具有reshape方法
                        data_values = (
                            data.values if hasattr(data, "values") else np.array(data)
                        )
                        # 总是转换为numpy数组以确保reshape方法可用
                        data_array = np.array(data_values)
                        encoded = encoder.transform(data_array.reshape(-1, 1))
                        # OneHotの結果を元のDataFrameに統合
                        feature_names = [
                            f"{col}_{cls}" for cls in encoder.categories_[0]
                        ]
                        encoded_df = pd.DataFrame(
                            encoded,
                            columns=pd.Index(feature_names),
                            index=X_transformed.index,
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

    scikit-learnの標準機能を活用した効率的なデータ処理を提供します。
    """

    def __init__(self):
        """初期化"""
        self.imputation_stats = {}  # 補完統計情報
        self.fitted_pipelines = {}  # 用途別のfittedパイプライン

    def create_preprocessing_pipeline(
        self,
        numeric_strategy: str = "median",
        categorical_strategy: str = "most_frequent",
        scaling_method: Optional[str] = "robust",
        remove_outliers: bool = True,
        outlier_threshold: float = 3.0,
        outlier_method: str = "iqr",
        outlier_transform: Optional[str] = None,  # 'robust' | 'quantile' | 'power'
    ) -> Pipeline:
        """
        scikit-learnのPipelineとColumnTransformerを使った宣言的な前処理パイプライン作成

        Args:
            numeric_strategy: 数値カラムの欠損値補完戦略
            categorical_strategy: カテゴリカルカラムの欠損値補完戦略
            scaling_method: スケーリング方法
            remove_outliers: 外れ値除去を行うか
            outlier_threshold: 外れ値の閾値
            outlier_method: 外れ値検出方法
            outlier_transform: 外れ値変換方法 ('robust', 'quantile', 'power')

        Returns:
            前処理パイプライン
        """
        # 数値カラム用の前処理パイプライン
        numeric_pipeline_steps = []

        # 1. 無限値をNaNに変換（数値計算で発生するinfをNaNに統一）
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
            # 外れ値検出方法に応じて適切なTransformerを選択
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
                # IQRまたはZ-scoreはIsolationForestで代替
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
                # 未対応の方法の場合は何もしないTransformer
                outlier_transformer = FunctionTransformer(
                    func=lambda X: X, validate=False
                )

            numeric_pipeline_steps.append(("outlier_removal", outlier_transformer))

        # 3. 欠損値補完（外れ値除去後のNaNを補完）
        numeric_pipeline_steps.append(
            ("imputer", SimpleImputer(strategy=numeric_strategy))
        )

        # 4. スケーリング（オプション）
        if scaling_method is not None:
            # 外れ値変換方法が指定されている場合
            if outlier_transform == "robust":
                # 中央値/IQRで頑健にスケーリング（外れ値の影響を受けにくい）
                scaler = RobustScaler()
            elif outlier_transform == "quantile":
                from sklearn.preprocessing import QuantileTransformer

                # ランク変換して外れ値の影響を緩和（正規分布にマップ）
                scaler = QuantileTransformer(
                    output_distribution="normal", random_state=42
                )
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
                    # カテゴリカルデータの欠損値は"Unknown"で埋める
                    SimpleImputer(strategy=categorical_strategy, fill_value="Unknown"),
                ),
                (
                    "encoder",
                    # ラベルエンコーディングを使用（OneHotは次元が大きくなるため）
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
                    # 数値型のカラムを選択
                    make_column_selector(dtype_include=np.number),
                ),
                (
                    "categorical",
                    categorical_pipeline,
                    # object型のカラムを選択
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
                    # 最終的なクリーンアップ（残ったNaNを0に変換）
                    FunctionTransformer(func=self._final_cleanup, validate=False),
                ),
            ]
        )

        return pipeline

    def _final_cleanup(self, X):
        """最終的なクリーンアップ"""
        # 残っているNaNを0で埋める
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

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

        result_df = df.copy()

        # OI/FRデータの補間
        result_df = self.interpolate_oi_fr_data(result_df)

        # Fear & Greedデータの補間
        result_df = self.interpolate_fear_greed_data(result_df)

        # logger.info("データ補間処理が完了")
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
            if bool(pd.isna(row[["Open", "High", "Low", "Close"]]).any()):
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

    def _encode_categorical_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        カテゴリカル変数を数値にエンコーディング

        CategoricalEncoderTransformerを使用した現代的な実装。
        scikit-learnのPipelineパターンに従い、保守性と再利用性を向上。

        Args:
            df: 対象のDataFrame

        Returns:
            エンコーディング済みのDataFrame
        """
        try:
            result_df = df.copy()

            categorical_columns = result_df.select_dtypes(
                include=["object", "string", "category"]
            ).columns.tolist()

            if not categorical_columns:
                return result_df

            logger.info(
                f"カテゴリカル変数をエンコーディング (CategoricalEncoderTransformer使用): {categorical_columns}"
            )

            # CategoricalEncoderTransformerを使用したパイプラインを作成
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
                    encoded, columns=pd.Index(feature_names), index=result_df.index
                )

            # 元のカテゴリカルカラムを削除して結合
            result_df = result_df.drop(columns=categorical_columns)
            result_df = pd.concat([result_df, encoded_df], axis=1)

            logger.info(
                "カテゴリカル変数のエンコーディングが完了 (CategoricalEncoderTransformer使用)"
            )
            return result_df

        except Exception as e:
            logger.error(f"カテゴリカル変数エンコーディングエラー: {e}")
            return df

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

                # pd.NA を None/NaN に正規化（pandasの欠損値表現を統一）
                series = series.replace({pd.NA: None})

                # 数値型を想定する場合は to_numeric で強制変換（文字列などの異常値をNaNに）
                if dtype is not None:
                    try:
                        # pd.to_numericで安全に数値変換を試行（errors="coerce"で変換不能はNaN）
                        series = pd.to_numeric(series, errors="coerce")

                        # 数値型の場合は指定されたdtypeに変換
                        if pd.api.types.is_numeric_dtype(series) and hasattr(
                            series, "astype"
                        ):
                            # 确保series是pandas Series类型
                            if isinstance(series, (pd.Series, np.ndarray)):
                                series = series.astype(dtype)
                    except Exception as e:
                        logger.warning(f"データ型変換でエラーが発生 ({dtype}): {e}")
                        # エラーの場合はpd.to_numericで数値型に変換
                        series = pd.to_numeric(series, errors="coerce")

                # 前方補完（時系列データの欠損を前の値で埋める）
                if forward_fill and hasattr(series, "ffill"):
                    # 确保series是pandas Series类型
                    if isinstance(series, (pd.Series, pd.DataFrame)):
                        series = series.ffill()
                    elif isinstance(series, np.ndarray):
                        # 对于numpy数组使用pad方法
                        series = pd.Series(series).ffill().values

                result_df[col] = series
            except Exception as e:
                logger.warning(f"列 {col} の前処理（型/ffill）でエラー: {e}")

        # 統計的補完を実行（SimpleImputerでstrategyに応じた補完を行う）
        if fit_if_needed:
            # scikit-learn ColumnTransformerを使用した効率的な実装
            from sklearn.compose import ColumnTransformer

            target_columns = [col for col in columns if col in result_df.columns]

            if target_columns:
                # 有効なカラムのみを対象（NaNが一つもないカラムは除外）
                valid_columns = [
                    col
                    for col in target_columns
                    if col in result_df.columns and result_df[col].notna().sum() > 0
                ]

                if valid_columns:
                    # ColumnTransformerで一括処理（指定されたstrategyで補完）
                    ct = ColumnTransformer(
                        [("imputer", SimpleImputer(strategy=strategy), valid_columns)],
                        remainder="passthrough",
                        verbose_feature_names_out=False,
                    )

                    imputed_data = ct.fit_transform(result_df)

                    # 特徴名を取得（ColumnTransformerから出力される特徴名）
                    try:
                        feature_names = ct.get_feature_names_out()
                    except Exception:
                        # get_feature_names_outが失敗した場合は元のカラム名を使用
                        feature_names = result_df.columns

                    # 結果をDataFrameに変換（インデックスは元のDataFrameを保持）
                    result_df = pd.DataFrame(
                        imputed_data, columns=feature_names, index=df.index
                    )

        # デフォルト値で最終補完（指定されたデフォルト値で残りのNaNを埋める）
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
            label_generator: ラベル生成器（LabelGenerator）
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
            features_processed = self.preprocess_with_pipeline(
                features_df,
                pipeline_name="training_preprocess",
                fit_pipeline=True,
                numeric_strategy=training_params.get("imputation_strategy", "median"),
                scaling_method=(
                    training_params.get("scaling_method", "robust")
                    if training_params.get("scale_features", True)
                    else None
                ),
                remove_outliers=training_params.get("remove_outliers", True),
                outlier_threshold=training_params.get("outlier_threshold", 3.0),
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
                # LabelGeneratorを直接使用（動的ラベル生成）
                from .label_generation import ThresholdMethod

                # 閾値計算方法を決定
                threshold_method_str = training_params.get(
                    "threshold_method", "dynamic_volatility"
                )

                # 文字列からEnumに変換
                method_mapping = {
                    "fixed": ThresholdMethod.FIXED,
                    "quantile": ThresholdMethod.QUANTILE,
                    "std_deviation": ThresholdMethod.STD_DEVIATION,
                    "adaptive": ThresholdMethod.ADAPTIVE,
                    "dynamic_volatility": ThresholdMethod.DYNAMIC_VOLATILITY,
                }

                threshold_method = method_mapping.get(
                    threshold_method_str, ThresholdMethod.STD_DEVIATION
                )

                # 目標分布を設定
                target_distribution = training_params.get(
                    "target_distribution", {"up": 0.33, "down": 0.33, "range": 0.34}
                )

                # 方法固有のパラメータを準備
                method_params = {}

                if threshold_method == ThresholdMethod.FIXED:
                    method_params["threshold"] = training_params.get(
                        "threshold_up", 0.02
                    )
                elif threshold_method == ThresholdMethod.STD_DEVIATION:
                    method_params["std_multiplier"] = training_params.get(
                        "std_multiplier", 0.25
                    )
                elif threshold_method == ThresholdMethod.DYNAMIC_VOLATILITY:
                    method_params["volatility_window"] = training_params.get(
                        "volatility_window", 24
                    )
                    method_params["threshold_multiplier"] = training_params.get(
                        "threshold_multiplier", 0.5
                    )
                    method_params["min_threshold"] = training_params.get(
                        "min_threshold", 0.005
                    )
                    method_params["max_threshold"] = training_params.get(
                        "max_threshold", 0.05
                    )
                elif threshold_method in [
                    ThresholdMethod.QUANTILE,
                    ThresholdMethod.ADAPTIVE,
                ]:
                    method_params["target_distribution"] = target_distribution

                # ラベルを生成
                labels, threshold_info = label_generator.generate_labels(
                    price_data,
                    method=threshold_method,
                    target_distribution=target_distribution,
                    **method_params,
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
        Pipelineベースの前処理実行（推奨API）

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
                # feature_namesがNoneまたは空の場合は自動生成
                if feature_names is None or len(feature_names) == 0:
                    feature_names = [
                        f"feature_{i}" for i in range(transformed_data.shape[1])
                    ]
            except Exception:
                # feature名が取得できない場合は自動生成
                feature_names = [
                    f"feature_{i}" for i in range(transformed_data.shape[1])
                ]

            result_df = pd.DataFrame(
                transformed_data, index=df.index, columns=pd.Index(feature_names)
            )

            logger.info(
                f"Pipeline前処理完了: {len(result_df)}行, {len(result_df.columns)}列"
            )
            return result_df

        except Exception as e:
            logger.error(f"Pipeline前処理エラー: {e}")
            raise

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
        from sklearn.base import BaseEstimator

        # 基本的な前処理パイプライン
        base_pipeline = self.create_preprocessing_pipeline(
            scaling_method="robust",  # MLには頑健なスケーリングを使用
            remove_outliers=True,
            outlier_method="iqr",
        )

        # Pipeline stepsリストを作成
        steps: List[Tuple[str, Union[Pipeline, BaseEstimator]]] = [
            ("base_preprocessing", base_pipeline)
        ]

        # 特徴選択（オプション）
        if feature_selection and n_features is not None and n_features > 0:
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
        self.imputation_stats.clear()
        self.fitted_pipelines.clear()  # Pipelineキャッシュもクリア
        logger.info("DataProcessorのキャッシュをクリアしました")

    # 推奨される新しいAPI
    def create_optimized_pipeline(
        self,
        for_ml: bool = True,
        include_feature_selection: bool = False,
        n_features: Optional[int] = None,
        **kwargs,
    ) -> Pipeline:
        """
        最適化されたパイプラインを作成（推奨API）

        Args:
            for_ml: 機械学習用の最適化を行うか
            include_feature_selection: 特徴選択を含めるか
            n_features: 選択する特徴数
            **kwargs: その他のパイプライン設定

        Returns:
            最適化されたPipeline
        """
        if for_ml:
            return self.create_ml_preprocessing_pipeline(
                feature_selection=include_feature_selection,
                n_features=n_features,
                **kwargs,
            )
        else:
            return create_comprehensive_preprocessing_pipeline(**kwargs)

    def process_data_efficiently(
        self,
        df: pd.DataFrame,
        pipeline_name: str = "efficient_processing",
        **pipeline_params,
    ) -> pd.DataFrame:
        """
        効率的なデータ処理（推奨API）

        scikit-learnの標準機能を最大限活用した高速処理。

        Args:
            df: 対象DataFrame
            pipeline_name: パイプライン名
            **pipeline_params: パイプライン設定

        Returns:
            処理されたDataFrame
        """
        logger.info("効率的なデータ処理を開始 (scikit-learn Pipeline使用)")

        try:
            # 最適化されたパイプラインを使用
            result = self.preprocess_with_pipeline(
                df, pipeline_name=pipeline_name, fit_pipeline=True, **pipeline_params
            )

            logger.info(
                f"効率的なデータ処理完了: {len(result)}行, {len(result.columns)}列"
            )
            return result

        except Exception as e:
            logger.error(f"効率的なデータ処理エラー: {e}")
            raise


# グローバルインスタンス
data_processor = DataProcessor()
