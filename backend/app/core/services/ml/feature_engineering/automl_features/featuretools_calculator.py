"""
Featuretools特徴量計算クラス

Deep Feature Synthesis (DFS)による既存特徴量の高次相互作用を自動発見します。
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import warnings
import time

from .....utils.unified_error_handler import safe_ml_operation
from .automl_config import FeaturetoolsConfig

import featuretools as ft

# featuretoolsが利用可能かどうかを示すフラグ
try:
    import featuretools as ft

    FEATURETOOLS_AVAILABLE = True
except ImportError:
    FEATURETOOLS_AVAILABLE = False
from featuretools.primitives import (
    Mean,
    Std,
    Min,
    Max,
    Count,
    Sum,
    Skew,
    Kurtosis,
    Trend,
    NumUnique,
    Mode,
    PercentTrue,
    AddNumeric,
    SubtractNumeric,
    MultiplyNumeric,
    DivideNumeric,
    GreaterThan,
    LessThan,
    And,
    Or,
    Not,
    Absolute,
    NaturalLogarithm,
    SquareRoot,
)


logger = logging.getLogger(__name__)


class FeaturetoolsCalculator:
    """
    Featuretools特徴量計算クラス

    Deep Feature Synthesis (DFS)による特徴量の自動生成を行います。
    """

    def __init__(self, config: Optional[FeaturetoolsConfig] = None):
        """
        初期化

        Args:
            config: Featuretools設定
        """
        self.config = config or FeaturetoolsConfig()
        self.entityset = None
        self.feature_defs = None
        self.last_synthesis_info = {}

    @safe_ml_operation(
        default_return=None, context="Featuretools特徴量計算でエラーが発生しました"
    )
    def calculate_featuretools_features(
        self,
        df: pd.DataFrame,
        target_entity: str = "prices",
        time_index: Optional[str] = None,
        max_depth: Optional[int] = None,
        max_features: Optional[int] = None,
        custom_primitives: Optional[Dict[str, List]] = None,
    ) -> pd.DataFrame:
        """
        Featuretools特徴量を計算

        Args:
            df: 入力データ
            target_entity: ターゲットエンティティ名
            time_index: 時間インデックス列名
            max_depth: 最大深度
            max_features: 最大特徴量数
            custom_primitives: カスタムプリミティブ

        Returns:
            Featuretools特徴量が追加されたDataFrame
        """

        if df is None or df.empty:
            logger.warning("空のデータが提供されました")
            return df

        try:
            start_time = time.time()

            # 設定の決定
            depth = max_depth if max_depth is not None else self.config.max_depth
            features_limit = (
                max_features if max_features is not None else self.config.max_features
            )

            logger.info(
                f"Featuretools DFS開始: 深度={depth}, 最大特徴量数={features_limit}"
            )

            # エンティティセットを作成
            self.entityset = self._create_entityset(df, target_entity, time_index)

            if self.entityset is None:
                logger.warning("エンティティセット作成に失敗しました")
                return df

            # プリミティブを設定
            agg_primitives, trans_primitives = self._get_primitives(custom_primitives)

            logger.info(
                f"使用するプリミティブ: agg={len(agg_primitives)}, trans={len(trans_primitives)}"
            )

            # Deep Feature Synthesisを実行
            logger.info(
                f"DFS実行開始: target_entity={target_entity}, depth={depth}, max_features={features_limit}"
            )
            logger.info(
                f"エンティティセット: {len(self.entityset.dataframes)}個のデータフレーム"
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                try:
                    feature_matrix, feature_defs = ft.dfs(
                        entityset=self.entityset,
                        target_dataframe_name=target_entity,  # target_entity → target_dataframe_name
                        agg_primitives=agg_primitives,
                        trans_primitives=trans_primitives,
                        max_depth=depth,
                        max_features=features_limit,
                        verbose=False,
                        n_jobs=1,  # 安定性のため
                    )
                    logger.info(
                        f"DFS実行成功: {len(feature_defs)}個の特徴量定義, 特徴量マトリックス形状: {feature_matrix.shape}"
                    )
                except Exception as dfs_error:
                    logger.error(f"DFS実行エラー: {dfs_error}")
                    logger.error(f"エンティティセット情報: {self.get_entityset_info()}")
                    return df

            self.feature_defs = feature_defs

            # 特徴量名を整理
            feature_matrix = self._clean_feature_names(feature_matrix)

            # 元のDataFrameに結合
            result_df = self._merge_features_with_original(df, feature_matrix)

            # 合成情報を保存
            synthesis_time = time.time() - start_time
            self.last_synthesis_info = {
                "total_features": len(feature_defs),
                "final_features": len(feature_matrix.columns),
                "synthesis_time": synthesis_time,
                "max_depth": depth,
                "target_entity": target_entity,
                "primitives_used": {
                    "aggregation": len(agg_primitives),
                    "transformation": len(trans_primitives),
                },
            }

            logger.info(
                f"Featuretools DFS完了: {len(feature_matrix.columns)}個の特徴量を生成 "
                f"({synthesis_time:.2f}秒)"
            )

            return result_df

        except Exception as e:
            logger.error(f"Featuretools特徴量計算エラー: {e}")
            return df

    def _create_entityset(
        self, df: pd.DataFrame, target_entity: str, time_index: Optional[str]
    ) -> Optional[ft.EntitySet]:
        """エンティティセットを作成"""
        try:
            # データの前処理
            processed_df = self._preprocess_data_for_entityset(df)

            if processed_df.empty:
                logger.warning("前処理後のデータが空です")
                return None

            # エンティティセットを初期化
            es = ft.EntitySet(id="financial_data")

            # インデックス列を確保
            if "index" not in processed_df.columns:
                processed_df = processed_df.reset_index()
                index_col = "index"
            else:
                index_col = "index"

            # 時間インデックスの設定
            time_col = (
                time_index
                if time_index and time_index in processed_df.columns
                else None
            )

            # メインエンティティを追加（新しいAPI）
            es = es.add_dataframe(
                dataframe_name=target_entity,
                dataframe=processed_df,
                index=index_col,
                time_index=time_col,
                make_index=True if index_col not in processed_df.columns else False,
            )

            # 価格データ用の追加エンティティを作成
            if self._has_ohlcv_data(processed_df):
                price_entity_df = self._create_price_entity(processed_df)
                if not price_entity_df.empty:
                    es = es.add_dataframe(
                        dataframe_name="price_summary",
                        dataframe=price_entity_df,
                        index="price_id",
                        make_index=True,
                    )

                    # リレーションシップを追加（新しいAPI）
                    es = es.add_relationship(
                        parent_dataframe_name="price_summary",
                        parent_column_name="price_id",
                        child_dataframe_name=target_entity,
                        child_column_name="price_group",
                    )

            logger.info(
                f"エンティティセット作成完了: {len(es.dataframes)}個のデータフレーム"
            )
            return es

        except Exception as e:
            logger.error(f"エンティティセット作成エラー: {e}")
            return None

    def _preprocess_data_for_entityset(self, df: pd.DataFrame) -> pd.DataFrame:
        """エンティティセット用のデータ前処理"""
        try:
            processed_df = df.copy()

            # 無限値とNaNを処理
            processed_df = processed_df.replace([np.inf, -np.inf], np.nan)

            # 数値列の型を統一
            numeric_columns = processed_df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                processed_df[col] = pd.to_numeric(processed_df[col], errors="coerce")

            # 価格グループを作成（時系列パターン分析用）
            if self._has_ohlcv_data(processed_df):
                processed_df["price_group"] = self._create_price_groups(processed_df)

            # カテゴリ変数を作成
            processed_df = self._add_categorical_features(processed_df)

            return processed_df

        except Exception as e:
            logger.error(f"データ前処理エラー: {e}")
            return df

    def _has_ohlcv_data(self, df: pd.DataFrame) -> bool:
        """OHLCVデータが含まれているかチェック"""
        ohlcv_cols = ["Open", "High", "Low", "Close", "Volume"]
        return any(col in df.columns for col in ohlcv_cols)

    def _create_price_groups(self, df: pd.DataFrame) -> pd.Series:
        """価格グループを作成"""
        try:
            if "Close" in df.columns:
                # 価格レンジでグループ化
                price_ranges = pd.qcut(
                    df["Close"],
                    q=5,
                    labels=False,  # 数値ラベルを使用
                    duplicates="drop",
                )
                # 整数型に変換してNaNを0で埋める
                return price_ranges.fillna(0).astype(int)
            else:
                return pd.Series([0] * len(df), index=df.index, dtype=int)
        except Exception:
            return pd.Series([0] * len(df), index=df.index, dtype=int)

    def _create_price_entity(self, df: pd.DataFrame) -> pd.DataFrame:
        """価格サマリーエンティティを作成"""
        try:
            if "price_group" not in df.columns:
                return pd.DataFrame()

            price_summary = (
                df.groupby("price_group")
                .agg(
                    {
                        "Close": (
                            ["mean", "std", "min", "max"]
                            if "Close" in df.columns
                            else []
                        ),
                        "Volume": ["mean", "sum"] if "Volume" in df.columns else [],
                    }
                )
                .reset_index()
            )

            # 列名を平坦化
            price_summary.columns = [
                "_".join(col).strip() if col[1] else col[0]
                for col in price_summary.columns.values
            ]

            price_summary = price_summary.rename(
                columns={"price_group_": "price_group"}
            )

            return price_summary

        except Exception as e:
            logger.error(f"価格エンティティ作成エラー: {e}")
            return pd.DataFrame()

    def _add_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """カテゴリ特徴量を追加"""
        try:
            result_df = df.copy()

            # 時間ベースのカテゴリ
            if hasattr(df.index, "hour"):
                result_df["hour_category"] = pd.cut(
                    df.index.hour,
                    bins=[0, 6, 12, 18, 24],
                    labels=["night", "morning", "afternoon", "evening"],
                    include_lowest=True,
                ).astype(str)

            # ボラティリティカテゴリ
            if "Close" in df.columns:
                returns = df["Close"].pct_change().abs()
                result_df["volatility_category"] = pd.cut(
                    returns, bins=3, labels=["low_vol", "medium_vol", "high_vol"]
                ).astype(str)

            return result_df

        except Exception as e:
            logger.error(f"カテゴリ特徴量追加エラー: {e}")
            return df

    def _get_primitives(
        self, custom_primitives: Optional[Dict[str, List]] = None
    ) -> Tuple[List, List]:
        """プリミティブを取得"""
        try:
            # デフォルトの集約プリミティブ（金融データ用）
            default_agg_primitives = [
                Mean,
                Std,
                Min,
                Max,
                Count,
                Sum,
                Skew,
                Kurtosis,
                NumUnique,
                Mode,
                PercentTrue,
                Trend,
            ]

            # デフォルトの変換プリミティブ（金融データ用）
            default_trans_primitives = [
                AddNumeric,
                SubtractNumeric,
                MultiplyNumeric,
                DivideNumeric,
                GreaterThan,
                LessThan,
                And,
                Or,
                Not,
                Absolute,
                NaturalLogarithm,
                SquareRoot,
            ]

            # カスタムプリミティブがある場合は追加
            if custom_primitives:
                if "agg_primitives" in custom_primitives:
                    agg_primitives = custom_primitives["agg_primitives"]
                else:
                    agg_primitives = default_agg_primitives

                if "trans_primitives" in custom_primitives:
                    trans_primitives = custom_primitives["trans_primitives"]
                else:
                    trans_primitives = default_trans_primitives
            else:
                agg_primitives = default_agg_primitives
                trans_primitives = default_trans_primitives

            # 設定からプリミティブを制限
            if hasattr(self.config, "agg_primitives") and self.config.agg_primitives:
                agg_primitives = self.config.agg_primitives

            if (
                hasattr(self.config, "trans_primitives")
                and self.config.trans_primitives
            ):
                trans_primitives = self.config.trans_primitives

            logger.debug(
                f"プリミティブ設定: 集約={len(agg_primitives)}, 変換={len(trans_primitives)}"
            )
            return agg_primitives, trans_primitives

        except Exception as e:
            logger.error(f"プリミティブ取得エラー: {e}")
            return [], []

    def _clean_feature_names(self, feature_matrix: pd.DataFrame) -> pd.DataFrame:
        """特徴量名をクリーンアップ"""
        try:
            cleaned_df = feature_matrix.copy()

            # 特徴量名にプレフィックスを追加
            new_columns = {}
            for col in cleaned_df.columns:
                # 特殊文字を置換
                clean_name = (
                    str(col).replace("(", "_").replace(")", "_").replace(" ", "_")
                )
                clean_name = clean_name.replace(",", "_").replace(".", "_")
                clean_name = f"FT_{clean_name}"
                new_columns[col] = clean_name

            cleaned_df = cleaned_df.rename(columns=new_columns)

            return cleaned_df

        except Exception as e:
            logger.error(f"特徴量名クリーンアップエラー: {e}")
            return feature_matrix

    def _merge_features_with_original(
        self, original_df: pd.DataFrame, features: pd.DataFrame
    ) -> pd.DataFrame:
        """特徴量を元のDataFrameに結合"""
        try:
            result_df = original_df.copy()

            # インデックスを合わせて結合
            if len(features) == len(original_df):
                for col in features.columns:
                    result_df[col] = features[col].values
            else:
                logger.warning(
                    f"特徴量の長さ({len(features)})と元データの長さ({len(original_df)})が一致しません"
                )
                # 短い方に合わせる
                min_length = min(len(features), len(original_df))
                for col in features.columns:
                    result_df.loc[: min_length - 1, col] = (
                        features[col].iloc[:min_length].values
                    )

            return result_df

        except Exception as e:
            logger.error(f"特徴量結合エラー: {e}")
            return original_df

    def get_feature_names(self) -> List[str]:
        """生成される特徴量名のリストを取得"""
        if self.feature_defs:
            return [
                f"FT_{str(feat).replace('(', '_').replace(')', '_').replace(' ', '_')}"
                for feat in self.feature_defs
            ]
        else:
            return []

    def get_synthesis_info(self) -> Dict[str, Any]:
        """最後の合成情報を取得"""
        return self.last_synthesis_info.copy()

    def get_entityset_info(self) -> Dict[str, Any]:
        """エンティティセット情報を取得"""
        if self.entityset is None:
            return {"message": "エンティティセットが作成されていません"}

        try:
            dataframes_info = {}
            for df_name, df in self.entityset.dataframes.items():
                dataframes_info[df_name] = {
                    "shape": df.shape,
                    "columns": list(df.columns),
                    "index": df.ww.index,
                    "time_index": df.ww.time_index,
                }

            relationships_info = []
            for relationship in self.entityset.relationships:
                relationships_info.append(
                    {
                        "parent": relationship.parent_dataframe_name,
                        "child": relationship.child_dataframe_name,
                        "parent_column": relationship.parent_column_name,
                        "child_column": relationship.child_column_name,
                    }
                )

            return {
                "dataframes": dataframes_info,
                "relationships": relationships_info,
                "total_dataframes": len(self.entityset.dataframes),
                "total_relationships": len(self.entityset.relationships),
            }

        except Exception as e:
            logger.error(f"エンティティセット情報取得エラー: {e}")
            return {"error": str(e)}

    def clear_entityset(self):
        """エンティティセットをクリア"""
        self.entityset = None
        self.feature_defs = None
        logger.debug("エンティティセットをクリアしました")

    def create_custom_primitives(self) -> Dict[str, Any]:
        """金融データ用カスタムプリミティブを作成"""
        try:
            # 金融データ専用のカスタムプリミティブを定義
            custom_primitives = {
                "agg_primitives": [
                    Mean,
                    Std,
                    Min,
                    Max,
                    Count,
                    Sum,
                    Skew,
                    Kurtosis,
                    NumUnique,
                    Trend,
                ],
                "trans_primitives": [
                    AddNumeric,
                    SubtractNumeric,
                    MultiplyNumeric,
                    DivideNumeric,
                    GreaterThan,
                    LessThan,
                    Absolute,
                    NaturalLogarithm,
                ],
            }

            return custom_primitives

        except Exception as e:
            logger.error(f"カスタムプリミティブ作成エラー: {e}")
            return {"agg_primitives": [], "trans_primitives": []}

    def optimize_feature_matrix(self, feature_matrix: pd.DataFrame) -> pd.DataFrame:
        """特徴量マトリックスを最適化"""
        try:
            optimized_df = feature_matrix.copy()

            # 無限値とNaNを処理
            optimized_df = optimized_df.replace([np.inf, -np.inf], np.nan)

            # 定数列を除去
            constant_columns = []
            for col in optimized_df.columns:
                if optimized_df[col].nunique() <= 1:
                    constant_columns.append(col)

            if constant_columns:
                optimized_df = optimized_df.drop(columns=constant_columns)
                logger.info(f"定数列を除去: {len(constant_columns)}個")

            # 高相関列を除去
            correlation_threshold = 0.99
            corr_matrix = optimized_df.corr().abs()
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )

            high_corr_columns = [
                column
                for column in upper_triangle.columns
                if any(upper_triangle[column] > correlation_threshold)
            ]

            if high_corr_columns:
                optimized_df = optimized_df.drop(columns=high_corr_columns)
                logger.info(f"高相関列を除去: {len(high_corr_columns)}個")

            return optimized_df

        except Exception as e:
            logger.error(f"特徴量マトリックス最適化エラー: {e}")
            return feature_matrix
