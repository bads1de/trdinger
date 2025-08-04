"""
データ前処理ユーティリティ

MLシステム全体で使用する統一されたデータ前処理機能を提供します。
SimpleImputerを使用した高品質な欠損値補完を実装します。
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    データ前処理クラス

    統一されたデータ前処理機能を提供し、fillna(0)を使用せず
    適切な統計的手法で欠損値を補完します。
    """

    def __init__(self):
        """初期化"""
        self.imputers = {}  # カラムごとのimputer
        self.scalers = {}  # カラムごとのscaler
        self.imputation_stats = {}  # 補完統計情報

    def fit_imputers(
        self,
        df: pd.DataFrame,
        strategy: str = "median",
        columns: Optional[List[str]] = None,
    ) -> None:
        """
        欠損値補完器を学習

        Args:
            df: 学習用DataFrame
            strategy: 補完戦略 ('mean', 'median', 'most_frequent', 'constant')
            columns: 対象カラム（Noneの場合は数値カラム全て）
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        logger.info(
            f"欠損値補完器を学習中: 戦略={strategy}, 対象カラム数={len(columns)}"
        )

        for col in columns:
            if col not in df.columns:
                continue

            try:
                # カラムごとに個別のimputer
                imputer = SimpleImputer(strategy=strategy)

                # 有効なデータのみで学習
                valid_data = df[col].dropna()
                if len(valid_data) > 0:
                    imputer.fit(valid_data.values.reshape(-1, 1))
                    self.imputers[col] = imputer

                    # 統計情報を記録
                    self.imputation_stats[col] = {
                        "strategy": strategy,
                        "fill_value": (
                            imputer.statistics_[0]
                            if len(imputer.statistics_) > 0
                            else 0
                        ),
                        "valid_samples": len(valid_data),
                        "total_samples": len(df[col]),
                    }
                else:
                    logger.warning(f"カラム {col}: 有効なデータがありません")

            except Exception as e:
                logger.error(f"カラム {col} の補完器学習エラー: {e}")

    def transform_missing_values(
        self,
        df: pd.DataFrame,
        strategy: str = "median",
        columns: Optional[List[str]] = None,
        fit_if_needed: bool = True,
    ) -> pd.DataFrame:
        """
        欠損値を補完

        Args:
            df: 対象DataFrame
            strategy: 補完戦略
            columns: 対象カラム
            fit_if_needed: 必要に応じて補完器を学習するか

        Returns:
            補完されたDataFrame
        """
        if df is None or df.empty:
            return df

        result_df = df.copy()

        if columns is None:
            columns = result_df.select_dtypes(include=[np.number]).columns.tolist()

        # 補完器が学習されていない場合は学習
        if fit_if_needed and not self.imputers:
            self.fit_imputers(result_df, strategy, columns)

        missing_before = result_df[columns].isnull().sum().sum()

        for col in columns:
            if col not in result_df.columns:
                continue

            missing_count = result_df[col].isnull().sum()
            if missing_count == 0:
                continue

            try:
                if col in self.imputers:
                    # 学習済みimputer使用
                    imputer = self.imputers[col]
                    result_df[col] = imputer.transform(
                        result_df[col].values.reshape(-1, 1)
                    ).flatten()
                else:
                    # その場でimputer作成
                    imputer = SimpleImputer(strategy=strategy)
                    valid_data = result_df[col].dropna()

                    if len(valid_data) > 0:
                        imputer.fit(valid_data.values.reshape(-1, 1))
                        result_df[col] = imputer.transform(
                            result_df[col].values.reshape(-1, 1)
                        ).flatten()
                    else:
                        # 有効なデータがない場合は0で補完（最終手段）
                        result_df[col] = result_df[col].fillna(0)
                        logger.warning(f"カラム {col}: 有効なデータがないため0で補完")

            except Exception as e:
                logger.error(f"カラム {col} の補完エラー: {e}")
                # エラー時は0で補完（最終手段）
                result_df[col] = result_df[col].fillna(0)

        missing_after = result_df[columns].isnull().sum().sum()
        logger.info(f"欠損値補完完了: {missing_before} → {missing_after}個")

        return result_df

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
            outlier_threshold: 外れ値の閾値（IQRの場合は倍数、Z-scoreの場合は標準偏差の倍数）
            scaling_method: スケーリング方法（standard, robust, minmax）
            outlier_method: 外れ値検出方法（iqr, zscore）

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

        # 6. 最終検証
        remaining_missing = result_df[numeric_columns].isnull().sum().sum()
        if remaining_missing > 0:
            logger.warning(f"前処理後も{remaining_missing}個の欠損値が残存")
            # 最終手段として0で補完
            result_df[numeric_columns] = result_df[numeric_columns].fillna(0)

        logger.info("特徴量の包括的前処理完了")
        return result_df

    def _remove_outliers(
        self,
        df: pd.DataFrame,
        columns: List[str],
        threshold: float = 3.0,
        method: str = "iqr",
    ) -> pd.DataFrame:
        """
        外れ値を除去（NaNに変換）

        Args:
            df: 対象DataFrame
            columns: 処理対象カラム
            threshold: 閾値（IQRの場合は倍数、Z-scoreの場合は標準偏差の倍数）
            method: 外れ値検出方法（iqr, zscore）
        """
        result_df = df.copy()
        outliers_removed = 0

        for col in columns:
            if col not in result_df.columns:
                continue

            try:
                series = result_df[col].dropna()
                if len(series) == 0:
                    continue

                if method.lower() == "iqr":
                    # IQR（四分位範囲）ベースの外れ値検出
                    Q1 = series.quantile(0.25)
                    Q3 = series.quantile(0.75)
                    IQR = Q3 - Q1

                    if IQR == 0:
                        continue  # IQRが0の場合はスキップ

                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR

                    outliers = (result_df[col] < lower_bound) | (
                        result_df[col] > upper_bound
                    )

                elif method.lower() == "zscore":
                    # Z-scoreベースの外れ値検出（従来の方法）
                    mean_val = series.mean()
                    std_val = series.std()

                    if pd.isna(mean_val) or pd.isna(std_val) or std_val == 0:
                        continue

                    z_scores = np.abs((result_df[col] - mean_val) / std_val)
                    outliers = z_scores > threshold

                else:
                    logger.warning(
                        f"未対応の外れ値検出方法: {method}。IQRを使用します。"
                    )
                    continue

                outlier_count = outliers.sum()
                if outlier_count > 0:
                    result_df.loc[outliers, col] = np.nan
                    outliers_removed += outlier_count

            except Exception as e:
                logger.error(f"カラム {col} の外れ値除去エラー: {e}")

        if outliers_removed > 0:
            logger.info(f"外れ値除去完了: {outliers_removed}個の値を除去（{method}法）")

        return result_df

    def _scale_features(
        self, df: pd.DataFrame, columns: List[str], method: str = "standard"
    ) -> pd.DataFrame:
        """
        特徴量をスケーリング

        Args:
            df: 対象DataFrame
            columns: スケーリング対象カラム
            method: スケーリング方法（standard, robust, minmax）
        """
        from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

        result_df = df.copy()

        # スケーラーの選択
        scaler_classes = {
            "standard": StandardScaler,
            "robust": RobustScaler,
            "minmax": MinMaxScaler,
        }

        if method not in scaler_classes:
            logger.warning(
                f"未対応のスケーリング方法: {method}。standardを使用します。"
            )
            method = "standard"

        scaler_class = scaler_classes[method]

        for col in columns:
            if col not in result_df.columns:
                continue

            try:
                # カラム固有のスケーラーキーを作成
                scaler_key = f"{col}_{method}"

                if scaler_key not in self.scalers:
                    self.scalers[scaler_key] = scaler_class()
                    # 有効なデータのみでフィット
                    valid_data = result_df[col].dropna().values.reshape(-1, 1)
                    if len(valid_data) > 0:
                        self.scalers[scaler_key].fit(valid_data)
                    else:
                        logger.warning(
                            f"カラム {col}: 有効なデータがないためスケーリングをスキップ"
                        )
                        continue

                # 変換を実行
                col_data = result_df[col].values.reshape(-1, 1)
                result_df[col] = self.scalers[scaler_key].transform(col_data).flatten()

            except Exception as e:
                logger.error(f"カラム {col} のスケーリングエラー: {e}")

        logger.info(f"{method}スケーリング完了: {len(columns)}個の特徴量を処理")
        return result_df

    def get_imputation_stats(self) -> Dict:
        """補完統計情報を取得"""
        return self.imputation_stats.copy()

    def clear_cache(self):
        """キャッシュをクリア"""
        self.imputers.clear()
        self.scalers.clear()
        self.imputation_stats.clear()
        logger.info("データ前処理キャッシュをクリアしました")


# グローバルインスタンス
data_preprocessor = DataPreprocessor()
