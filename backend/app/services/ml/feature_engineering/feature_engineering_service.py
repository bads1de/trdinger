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

from app.services.ml.common.utils import generate_cache_key, optimize_dtypes

from .advanced_rolling_stats import AdvancedRollingStatsCalculator
from .crypto_features import CryptoFeatures
from .data_frequency_manager import DataFrequencyManager
from .interaction_features import InteractionFeatureCalculator
from .market_data_features import MarketDataFeatureCalculator
from .microstructure_features import MicrostructureFeatureCalculator
from .multi_timeframe_features import MultiTimeframeFeatureCalculator
from .oi_fr_interaction_features import OIFRInteractionFeatureCalculator
from .price_features import PriceFeatureCalculator
from .technical_features import TechnicalFeatureCalculator
from .volume_profile_features import VolumeProfileFeatureCalculator

logger = logging.getLogger(__name__)


class FeatureEngineeringService:
    """
    高度な特徴量生成を統括するオーケストレーター

    単一責任原則に基づき、テクニカル指標、マイクロストラクチャ（Roll、Kyle等）、
    OI/FR 相関、マルチタイムフレーム分析といった各専門分野の計算ロジックを
    個別の Calculator クラスに委譲し、最終的な特徴量行列を組み立てます。
    計算結果のキャッシュ、データ型最適化、欠損値補完などの共通処理も一括管理します。
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
        self.microstructure_calculator = MicrostructureFeatureCalculator()

        # 新規追加: 学術的に検証された強力な特徴量計算クラス
        self.volume_profile_calculator = VolumeProfileFeatureCalculator()
        self.oi_fr_interaction_calculator = OIFRInteractionFeatureCalculator()
        self.advanced_stats_calculator = AdvancedRollingStatsCalculator()
        self.multi_timeframe_calculator = MultiTimeframeFeatureCalculator()

        # データ頻度統一マネージャー
        self.frequency_manager = DataFrequencyManager()

        # 暗号通貨特化特徴量エンジニアリング（デフォルトで有効）
        self.crypto_features = CryptoFeatures()
        logger.debug("暗号通貨特化特徴量を有効化しました")

    def calculate_advanced_features(
        self,
        ohlcv_data: pd.DataFrame,
        funding_rate_data: Optional[pd.DataFrame] = None,
        open_interest_data: Optional[pd.DataFrame] = None,
        lookback_periods: Optional[Dict[str, int]] = None,
        profile: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        OHLCV、FR、OI データから統合された特徴量セットを効率的に算出

        各専門分野の計算機（テクニカル、マイクロストラクチャ、OI/FR、
        マルチタイムフレーム等）を順次実行し、最終的な特徴量
        マトリックスを生成します。インデックスの整合性チェックや、
        頻度の統一処理、欠損値補完などの前処理も自動的に行います。

        Args:
            ohlcv_data: 基準となる OHLCV 価格データ
            funding_rate_data: ファンディングレート（FR）データ（任意）
            open_interest_data: 建玉残高（OI）データ（任意）
            lookback_periods: 各指標の計算期間（デフォルト設定あり）
            profile: 特徴量プロファイル（計算範囲の制御用）

        Returns:
            インデックスが ohlcv_data と一致する、計算済みの全特徴量を含む DataFrame
        """
        try:
            if ohlcv_data.empty:
                raise ValueError("OHLCVデータが空です")

            # DataFrameのインデックスをDatetimeIndexに変換
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
                        start="2024-01-01", periods=len(ohlcv_data), freq="1h"
                    )

            # インデックスがDatetimeIndexであることを確認
            if not isinstance(ohlcv_data.index, pd.DatetimeIndex):
                raise ValueError(
                    "DataFrameのインデックスはDatetimeIndexである必要があります"
                )

            # メモリ使用量制限を撤廃（ユーザー要望により全データ使用）
            # if len(ohlcv_data) > 200000:
            #     logger.warning(
            #         f"大量のデータ（{len(ohlcv_data)}行）、最新200,000行に制限"
            #     )
            #     ohlcv_data = ohlcv_data.tail(200000)

            # キャッシュキーを生成
            cache_key = generate_cache_key(
                ohlcv_data,
                funding_rate_data,
                open_interest_data,
                extra_params=lookback_periods,
            )

            # キャッシュから結果を取得
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                return cached_result

            # データ頻度統一処理
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

            # 結果DataFrameを初期化
            result_df = ohlcv_data.copy()

            # データ型を最適化
            result_df = optimize_dtypes(result_df)

            # 1. 基本的な特徴量計算（result_dfを直接更新するタイプ）
            core_calculators = [
                (self.price_calculator, {"lookback_periods": lookback_periods}),
                (self.technical_calculator, {"lookback_periods": lookback_periods}),
                (
                    self.market_data_calculator,
                    {
                        "lookback_periods": lookback_periods,
                        "funding_rate_data": funding_rate_data,
                        "open_interest_data": open_interest_data,
                    },
                ),
            ]

            for calc, config in core_calculators:
                result_df = calc.calculate_features(result_df, config)

            # 2. 特化型特徴量計算（独自のメソッド名を持つタイプ）
            if self.crypto_features:
                logger.debug("暗号通貨特化特徴量を計算中...")
                result_df = self.crypto_features.create_crypto_features(
                    result_df, funding_rate_data, open_interest_data
                )

            result_df = self.interaction_calculator.calculate_interaction_features(
                result_df
            )

            # 3. 追加の特徴量計算（追加のDataFrameを返し、最後にconcatするタイプ）
            logger.info("学術論文・Kaggle実証済み特徴量を計算中...")
            additional_features_list = []

            # (電卓, 引数リスト) の形式で定義
            additional_calculators = [
                (self.volume_profile_calculator, [result_df]),
                (
                    self.oi_fr_interaction_calculator,
                    [result_df, open_interest_data, funding_rate_data],
                ),
                (self.advanced_stats_calculator, [result_df]),
                (self.multi_timeframe_calculator, [result_df]),
                (self.microstructure_calculator, [result_df]),
            ]

            for calc, args in additional_calculators:
                try:
                    feat_df = calc.calculate_features(*args)
                    additional_features_list.append(feat_df)
                except Exception as e:
                    logger.error(f"{calc.__class__.__name__} の計算中にエラー: {e}")

            # 一度だけconcatを実行
            if additional_features_list:
                result_df = pd.concat([result_df] + additional_features_list, axis=1)

            # 重複カラムを削除（新しい値を優先）
            result_df = result_df.loc[:, ~result_df.columns.duplicated(keep="last")]

            # 特徴量選択（動的選択）への橋渡し：ここではフィルタリングせず、すべてを返す
            # ALLOWLISTによる手動制限は廃止し、後続のFeatureSelectorに委ねる
            logger.info(f"生成された全特徴量を使用します: {len(result_df.columns)}個")

            # データ前処理
            logger.info("統計的手法による特徴量前処理を実行中...")
            try:
                # 数値列の一括処理
                num_cols = result_df.select_dtypes(include=[np.number]).columns
                result_df[num_cols] = (
                    result_df[num_cols]
                    .replace([np.inf, -np.inf], np.nan)
                    .ffill()
                    .fillna(0.0)
                )
                logger.info("データ前処理完了")
            except Exception as e:
                logger.error(f"データ前処理中にエラーが発生: {e}")
                # エラーが発生しても処理を続行（部分的な欠損値許容）

            # キャッシュに保存
            self._save_to_cache(cache_key, result_df)

            return result_df

        except Exception as e:
            logger.error(f"高度な特徴量計算中にエラーが発生: {e}")
            # エラー時は元のDataFrameを返す（最低限の動作保証）
            return ohlcv_data

    def _get_from_cache(self, key: str) -> Optional[pd.DataFrame]:
        """キャッシュからデータを取得"""
        if key in self.feature_cache:
            entry = self.feature_cache[key]
            if (datetime.now() - entry["timestamp"]).total_seconds() < self.cache_ttl:
                logger.debug("特徴量キャッシュヒット")
                return entry["data"]
            else:
                del self.feature_cache[key]
        return None

    def _save_to_cache(self, key: str, data: pd.DataFrame):
        """キャッシュにデータを保存"""
        if len(self.feature_cache) >= self.max_cache_size:
            # 最も古いエントリを削除
            oldest_key = min(
                self.feature_cache.keys(),
                key=lambda k: self.feature_cache[k]["timestamp"],
            )
            del self.feature_cache[oldest_key]

        self.feature_cache[key] = {"data": data, "timestamp": datetime.now()}
