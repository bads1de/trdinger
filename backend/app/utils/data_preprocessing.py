"""
データ前処理ユーティリティ

MLシステム全体で使用する統一されたデータ前処理機能を提供します。
SimpleImputerを使用した高品質な欠損値補完を実装します。
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

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

    def prepare_training_data(
        self, features_df: pd.DataFrame, label_generator, **training_params
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        学習用データを準備（BaseMLTrainerから分離された汎用関数）

        Args:
            features_df: 特徴量DataFrame
            label_generator: ラベル生成器のインスタンス
            **training_params: 学習パラメータ

        Returns:
            前処理済み特徴量DataFrame, ラベルSeries
        """
        logger.info("学習用データ準備を開始")

        # デフォルト実装：最後の列をラベルとして使用
        if features_df.empty:
            from ..exceptions.unified_exceptions import UnifiedDataError

            raise UnifiedDataError("特徴量データが空です")

        # 数値列のみを選択
        numeric_columns = features_df.select_dtypes(include=[np.number]).columns
        features_df_numeric = features_df[numeric_columns]

        # 統計的手法で欠損値を補完（スケーリング有効化、IQRベース外れ値検出）
        logger.info("統計的手法による特徴量前処理を実行中...")
        features_df_clean = self.preprocess_features(
            features_df_numeric,
            imputation_strategy="median",
            scale_features=True,  # 特徴量スケーリングを有効化
            remove_outliers=True,
            outlier_threshold=3.0,
            scaling_method="robust",  # ロバストスケーリングを使用
            outlier_method="iqr",  # IQRベースの外れ値検出を使用
        )

        # 特徴量とラベルを分離（改善されたラベル生成ロジック）
        if "Close" in features_df_clean.columns:
            # 動的ラベル生成を使用
            labels, threshold_info = label_generator.generate_dynamic_labels(
                features_df_clean["Close"], **training_params
            )

            # 閾値情報をログ出力
            logger.info(f"ラベル生成方法: {threshold_info['description']}")
            logger.info(
                f"使用閾値: {threshold_info['threshold_down']:.6f} ～ {threshold_info['threshold_up']:.6f}"
            )

            # 最後の行は予測できないので除外
            features_df_clean = features_df_clean.iloc[:-1]
        else:
            from ..exceptions.unified_exceptions import UnifiedDataError

            raise UnifiedDataError("価格データ（Close）が見つかりません")

        # 無効なデータを除外
        valid_mask = ~(features_df_clean.isnull().any(axis=1) | labels.isnull())
        features_clean = features_df_clean[valid_mask]
        labels_clean = labels[valid_mask]

        if len(features_clean) == 0:
            from ..exceptions.unified_exceptions import UnifiedDataError

            raise UnifiedDataError("有効な学習データがありません")

        logger.info(
            f"学習データ準備完了: {len(features_clean)}サンプル, {len(features_clean.columns)}特徴量"
        )
        logger.info(
            f"ラベル分布: 下落={sum(labels_clean==0)}, レンジ={sum(labels_clean==1)}, 上昇={sum(labels_clean==2)}"
        )

        return features_clean, labels_clean, threshold_info

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

    def interpolate_columns(
        self,
        df: pd.DataFrame,
        columns: List[str],
        strategy: str = "median",
        forward_fill: bool = True,
        dtype: Optional[str] = "float64",
        default_fill_values: Optional[Dict[str, float]] = None,
        fit_if_needed: bool = True,
    ) -> pd.DataFrame:
        """
        指定カラムに対して共通の補間パターン（型正規化 → ffill → 統計補完 → 既定値フォールバック）を適用

        Args:
            df: 対象DataFrame
            columns: 対象カラム名のリスト（存在しないカラムはスキップ）
            strategy: 統計補完戦略（'mean' | 'median' | 'most_frequent' | 'constant'）
            forward_fill: True の場合は前方補完を先に実施
            dtype: 数値補完する場合の目標dtype（例: 'float64'）。None の場合は型変換を行わない
            default_fill_values: 列ごとの最終フォールバック値マップ（欠損が残った場合に fillna で適用）
            fit_if_needed: transform_missing_values 呼び出し時に必要に応じて学習を行うか

        Returns:
            補間適用後のDataFrame
        """
        if df is None or df.empty or not columns:
            return df

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

                # 数値型を想定する場合は to_numeric で強制変換（非数値は NaN へ）
                if dtype is not None:
                    series = pd.to_numeric(series, errors="coerce")
                    try:
                        series = series.astype(dtype)
                    except Exception:
                        # 型変換に失敗しても続行
                        pass

                # 前方補完
                if forward_fill:
                    series = series.ffill()

                result_df[col] = series
            except Exception as e:
                logger.warning(f"列 {col} の前処理（型/ffill）でエラー: {e}")

        # 統計補完を適用
        try:
            result_df = self.transform_missing_values(
                result_df,
                strategy=strategy,
                columns=[c for c in columns if c in result_df.columns],
                fit_if_needed=fit_if_needed,
            )
        except Exception as e:
            logger.error(f"統計補完の適用に失敗: {e}")

        # 最終フォールバック（列ごと既定値）
        for col, def_val in default_fill_values.items():
            if col in result_df.columns and result_df[col].isnull().any():
                result_df[col] = result_df[col].fillna(def_val)

        return result_df

    def clear_cache(self):
        """キャッシュをクリア"""
        self.imputers.clear()
        self.scalers.clear()
        self.imputation_stats.clear()
        logger.info("データ前処理キャッシュをクリアしました")


# グローバルインスタンス
data_preprocessor = DataPreprocessor()
