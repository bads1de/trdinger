"""
データプロセッサーの統合インターフェース

リファクタリング後の新しいモジュール構造を統合した高レベルAPIを提供。
transformers, pipelines, validatorsモジュールを統一的に操作可能。
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .transformers.dtype_optimizer import DtypeOptimizer

from .pipelines.preprocessing_pipeline import get_pipeline_info
from .pipelines.ml_pipeline import create_ml_pipeline
from .pipelines.comprehensive_pipeline import create_comprehensive_pipeline

from .validators.data_validator import (
    validate_ohlcv_data,
    validate_extended_data,
    validate_data_integrity,
)

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    統合データ処理クラス

    transformers, pipelines, validatorsモジュールを統合した高レベルAPIを提供。
    既存のAPIとの後方互換性を維持。
    """

    def __init__(self):
        """初期化"""
        self.imputation_stats = {}  # 補完統計情報
        self.fitted_pipelines = {}  # 用途別のfittedパイプライン
        self._pipeline_cache = {}  # パイプラインキャッシュ

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

        # カラム名を小文字に統一（大文字小文字のケースを統一）
        result_df.columns = result_df.columns.str.lower()

        # 拡張データの範囲クリップ（funding_rateなど）
        result_df = self._clip_extended_data_ranges(result_df)

        # 必要なカラムを定義
        ohlcv_columns = ["open", "high", "low", "close", "volume"]
        
        # データ補間 (NaN/null値を先に処理)
        if interpolate:
            result_df = self._interpolate_data(result_df)

        # データ検証 (補間後に実行)
        try:
            # 空のデータは検証をスキップ
            if result_df.empty:
                logger.warning("データが空のため検証をスキップ")
                return result_df
                
            # 必要なカラムに基づいて検証を実行
            if not result_df.empty:
                ok_columns = {"open", "high", "low", "close", "volume"}
                if any(col in required_columns for col in ohlcv_columns):
                    validate_ohlcv_data(result_df)
                validate_extended_data(result_df)
                validate_data_integrity(result_df)
        except Exception as e:
            logger.error(f"データ検証でエラー: {e}")
            raise ValueError(f"データ検証に失敗しました: {e}")

        # データ型最適化
        if optimize:
            optimizer = DtypeOptimizer()
            result_df = optimizer.fit_transform(result_df)

        # 時系列順にソート
        if hasattr(result_df.index, "is_monotonic_increasing"):
            # Pandas Series/Index比較を安全に行う - boolに変換してから評価
            is_sorted = bool(result_df.index.is_monotonic_increasing)
            if not is_sorted:
                result_df = result_df.sort_index()

        return result_df

    def prepare_training_data(
        self, features_df: pd.DataFrame, label_generator, **training_params
    ) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        """
        学習用データを準備

        Args:
            features_df: 特徴量DataFrame
            label_generator: ラベル生成器
            **training_params: 学習パラメータ

        Returns:
            features_clean: クリーンな特徴量DataFrame
            labels_clean: クリーンなラベルSeries
            threshold_info: 閾値情報の辞書
        """
        logger.info("学習用データの準備を開始")
        logger.info(
            f"入力データサイズ: {len(features_df)}行, {len(features_df.columns)}列"
        )

        # 1. 入力データの基本検証
        if features_df is None or features_df.empty:
            raise ValueError("入力特徴量データが空です")

        # 2. データクリーニング
        logger.info("特徴量データのクリーニングを実行")
        features_processed = self.clean_and_validate_data(
            features_df,
            required_columns=[],  # 学習データなので必須カラムなし
            interpolate=True,
            optimize=True,
        )

        logger.info(
            f"クリーニング後データサイズ: {len(features_processed)}行, {len(features_processed.columns)}列"
        )

        # 3. 前処理パイプライン適用
        logger.info("前処理パイプラインを実行")
        features_processed = self.preprocess_with_pipeline(
            features_processed,
            pipeline_name="training_preprocess",
            fit_pipeline=True,
            **training_params,
        )

        # 4. ラベル生成のための価格データを取得
        if "close" not in features_processed.columns:
            logger.error(f"利用可能なカラム: {list(features_processed.columns)}")
            raise ValueError("close価格データが特徴量に含まれていません")

        price_data = features_processed["close"]
        logger.info(f"価格データサイズ: {len(price_data)}行")

        # 5. ラベル生成
        logger.info("ラベル生成を実行")
        try:
            labels, threshold_info = label_generator.generate_labels(
                price_data, **training_params
            )
            logger.info(f"ラベル生成完了: {len(labels)}行")
        except Exception as label_error:
            logger.error(f"ラベル生成エラー: {label_error}")
            raise ValueError(f"ラベル生成に失敗しました: {label_error}")

        # 6. 特徴量とラベルのインデックスを整合
        logger.info("データの整合性を確保中")
        common_index = features_processed.index.intersection(labels.index)

        if len(common_index) == 0:
            logger.error("特徴量とラベルに共通のインデックスがありません")
            raise ValueError("特徴量とラベルに共通のインデックスがありません")

        features_clean = features_processed.loc[common_index]
        labels_clean = labels.loc[common_index]

        # 7. NaNを含む行を除去
        logger.info("NaN値の除去を実行中")
        # Pandasのブール比較問題を回避する安全な方法
        features_notna = features_clean.notna().all(axis=1)
        labels_notna = labels_clean.notna()
        valid_mask = features_notna & labels_notna
        features_clean = features_clean[valid_mask]
        labels_clean = labels_clean[valid_mask]

        # 8. 最終的なデータ検証
        if len(features_clean) == 0 or len(labels_clean) == 0:
            logger.error("有効な学習データが存在しません")
            raise ValueError("有効な学習データが存在しません")

        if len(features_clean) != len(labels_clean):
            raise ValueError("特徴量とラベルの長さが一致しません")

        logger.info(f"学習用データ準備完了: {len(features_clean)}行")
        return features_clean, labels_clean, threshold_info

    def preprocess_with_pipeline(
        self,
        df: pd.DataFrame,
        pipeline_name: str = "default",
        fit_pipeline: bool = True,
        **pipeline_params,
    ) -> pd.DataFrame:
        """
        Pipelineベースの前処理実行

        Args:
            df: 対象DataFrame
            pipeline_name: パイプライン名（キャッシュ用）
            fit_pipeline: パイプラインをfitするか
            **pipeline_params: パイプライン作成パラメータ

        Returns:
            前処理されたDataFrame
        """
        logger.info(f"Pipelineベース前処理開始: {pipeline_name}")
        
        # 元のカラム名を保存
        original_columns = df.columns.tolist()
        n_original_cols = len(original_columns)

        if fit_pipeline or pipeline_name not in self.fitted_pipelines:
            # 新しいパイプラインを作成
            pipeline = self.create_optimized_pipeline(**pipeline_params)

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

        # 列数を確認
        n_transformed_cols = transformed_data.shape[1]
        
        if n_transformed_cols == n_original_cols:
            # 列数が一致する場合、元のカラム名を使用
            feature_names = original_columns
            logger.info(f"元のカラム名を使用: {n_original_cols}列")
        else:
            # 列数が一致しない場合、警告を出して元のカラム名の範囲内で使用
            logger.warning(
                f"列数不一致: 変換後={n_transformed_cols}, 元={n_original_cols}"
            )
            if n_transformed_cols < n_original_cols:
                # 列が減った場合（特徴選択など）
                feature_names = original_columns[:n_transformed_cols]
            else:
                # 列が増えた場合、元のカラム名 + 汎用名
                feature_names = original_columns + [
                    f"generated_feature_{i}"
                    for i in range(n_transformed_cols - n_original_cols)
                ]

        result_df = pd.DataFrame(
            transformed_data, index=df.index, columns=pd.Index(feature_names)
        )
        
        # 重複カラムを削除（最初のものを保持）
        if result_df.columns.duplicated().any():
            duplicated_cols = result_df.columns[result_df.columns.duplicated()].unique().tolist()
            logger.warning(f"重複カラムを削除: {duplicated_cols}")
            result_df = result_df.loc[:, ~result_df.columns.duplicated()]
            logger.info(f"重複削除後: {len(result_df.columns)}列")

        logger.info(
            f"Pipeline前処理完了: {len(result_df)}行, {len(result_df.columns)}列"
        )
        return result_df

    def process_data_efficiently(
        self,
        df: pd.DataFrame,
        pipeline_name: str = "efficient_processing",
        **pipeline_params,
    ) -> pd.DataFrame:
        """
        効率的なデータ処理実行

        Args:
            df: 対象DataFrame
            pipeline_name: パイプライン名
            **pipeline_params: パイプライン設定

        Returns:
            処理されたDataFrame
        """
        logger.info("効率的なデータ処理を開始")

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

    def create_optimized_pipeline(
        self,
        for_ml: bool = True,
        include_feature_selection: bool = False,
        n_features: Optional[int] = None,
        **kwargs,
    ) -> Pipeline:
        """
        最適化されたパイプラインを作成

        Args:
            for_ml: 機械学習用の最適化を行うか
            include_feature_selection: 特徴選択を含めるか
            n_features: 選択する特徴数
            **kwargs: その他のパイプライン設定

        Returns:
            最適化されたPipeline
        """
        if for_ml:
            # ML用パイプライン
            pipeline = create_ml_pipeline(
                include_feature_selection=include_feature_selection,
                n_features=n_features,
                **kwargs,
            )
        else:
            # 包括的パイプライン
            pipeline = create_comprehensive_pipeline(**kwargs)

        return pipeline

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

        # pipelinesモジュールのget_pipeline_infoを使用
        base_info = get_pipeline_info(pipeline)

        # existsキーを追加
        info = {"exists": True}
        info.update(base_info)
        return info

    def clear_cache(self):
        """キャッシュをクリア"""
        self.imputation_stats.clear()
        self.fitted_pipelines.clear()
        self._pipeline_cache.clear()
        logger.info("DataProcessorのキャッシュをクリアしました")

    def _interpolate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """データ補間処理"""
        result_df = df.copy()

        # 数値カラムの補間
        numeric_columns = result_df.select_dtypes(include=[np.number]).columns

        for col in numeric_columns:
            # inf値をNaNに変換（補間前に）
            result_df[col] = result_df[col].replace([np.inf, -np.inf], np.nan)

            # Pandas Series比較を安全に行う - .item()でスカラー値を取得
            # .any()がSeriesを返す場合に備えて、明示的にitem()を使用
            try:
                has_null_check = result_df[col].isnull().any()
                # has_null_checkがSeriesの場合、.item()でスカラー値を取得
                if hasattr(has_null_check, 'item'):
                    has_null = has_null_check.item()
                else:
                    has_null = bool(has_null_check)
            except (ValueError, AttributeError):
                # フォールバック: 要素ごとにチェック
                has_null = result_df[col].isnull().values.any()
            
            if has_null:
                # 前方補完 → 線形補完 → 後方補完
                result_df[col] = (
                    result_df[col].ffill().interpolate(method="linear").bfill()
                )

        # 特別なOHLCカラムの補間後検証と修正
        if all(col in result_df.columns for col in ['open', 'high', 'low', 'close']):
            # OHLCカラムが全て存在する場合のみ検証
            for idx in result_df.index:
                row = result_df.loc[idx]
                
                # NaN値が含まれている行はスキップ
                ohlc_values = row[['open', 'high', 'low', 'close']]
                # Pandas Series比較を安全に行う - boolに変換してから評価
                has_null_ohlc = bool(ohlc_values.isnull().any())
                if has_null_ohlc:
                    continue
                    
                # OHLC関係が崩れている場合は修正
                # Pandas Series比較を安全に行う
                low_val = float(row['low'])
                open_val = float(row['open'])
                high_val = float(row['high'])
                close_val = float(row['close'])
                
                ohlc_valid = (
                    (low_val <= open_val) and
                    (open_val <= high_val) and
                    (low_val <= close_val) and
                    (close_val <= high_val)
                )
                
                if not ohlc_valid:
                    # OHLC関係を強制的に修正
                    valid_min = min(open_val, close_val)
                    valid_max = max(open_val, close_val)
                    
                    # lowが適切な最小値になるように修正
                    if low_val > valid_min:
                        result_df.loc[idx, 'low'] = valid_min
                        
                    # highが適切な最大値になるように修正
                    if high_val < valid_max:
                        result_df.loc[idx, 'high'] = valid_max

        # カテゴリカルカラムの補間
        categorical_columns = result_df.select_dtypes(
            include=["object", "category"]
        ).columns

        for col in categorical_columns:
            # Pandas Series比較を安全に行う - boolに変換してから評価
            has_null = bool(result_df[col].isnull().any())
            if has_null:
                # 最頻値で補完
                mode_value = result_df[col].mode()
                if not mode_value.empty:
                    result_df[col] = result_df[col].fillna(mode_value.iloc[0])

        return result_df

    def _clip_extended_data_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """拡張データの範囲クリップ処理"""
        result_df = df.copy()

        # funding_rateの範囲クリップ (-1から1)
        if "funding_rate" in result_df.columns:
            # NaNとinfを処理してからクリップ
            funding_rate_clean = result_df["funding_rate"].replace(
                [np.inf, -np.inf], np.nan
            )
            # Pandas Series比較を安全に行う
            below_min = (funding_rate_clean < -1).sum()
            above_max = (funding_rate_clean > 1).sum()
            before_count = below_min + above_max

            # 常にクリップを実行（範囲外値がなくてもNaN/infの処理のため）
            result_df["funding_rate"] = np.clip(funding_rate_clean.fillna(0), -1, 1)
            
            # 修正後の範囲外値をカウント
            below_min_after = (result_df["funding_rate"] < -1).sum()
            above_max_after = (result_df["funding_rate"] > 1).sum()
            after_count = below_min_after + above_max_after
            
            if before_count > 0:
                logger.info(f"範囲外値を修正: {before_count}件")

        # open_interestは負値にならないようにクリップ
        if "open_interest" in result_df.columns:
            oi_clean = result_df["open_interest"].replace([np.inf, -np.inf], np.nan)
            # Pandas Series比較を安全に行う
            before_count = (oi_clean < 0).sum()
            if before_count > 0:
                result_df["open_interest"] = np.maximum(oi_clean.fillna(0), 0)

        return result_df


# グローバルインスタンス（後方互換性維持）
data_processor = DataProcessor()
