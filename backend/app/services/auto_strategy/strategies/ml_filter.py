"""
MLフィルターモジュール

UniversalStrategyのMLフィルター関連ロジックを担当します。
特徴量事前計算、エントリー可否判定、特徴量準備などの機能を提供します。
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class MLFilter:
    """
    MLフィルタークラス

    UniversalStrategyのMLフィルター関連ロジックを分離したクラス。
    特徴量事前計算、エントリー可否判定、特徴量準備などの機能を提供します。
    """

    def __init__(self, strategy):
        """
        初期化

        Args:
            strategy: UniversalStrategyインスタンス
        """
        self.strategy = strategy

    def precompute_ml_features(self) -> None:
        """ML予測に必要な全期間の特徴量を一括計算してキャッシュする"""
        try:
            from ..core.hybrid_feature_adapter import HybridFeatureAdapter

            # アダプターの初期化
            self.strategy.feature_adapter = HybridFeatureAdapter()

            # 全期間のデータを準備
            full_ohlcv = self.strategy.data.df.copy()
            full_ohlcv.columns = [c.lower() for c in full_ohlcv.columns]

            # 一括変換実行
            self.strategy._precomputed_features = (
                self.strategy.feature_adapter.gene_to_features(
                    gene=self.strategy.gene,
                    ohlcv_data=full_ohlcv,
                    apply_preprocessing=False,
                )
            )
        except Exception as e:
            logger.error(f"ML特徴量事前計算エラー: {e}")
            self.strategy._precomputed_features = None

    def ml_allows_entry(self, direction: float) -> bool:
        """
        MLがエントリーを許可するかチェック

        MLフィルター（ダマシ予測モデル）が設定されている場合、
        予測結果に基づいてエントリーの可否を判断します。

        ダマシ予測モデルは「このエントリーシグナルが有効かどうか」を
        0-1の確率で出力します。is_valid が閾値以上であればエントリーを許可。

        Args:
            direction: 取引方向 (1.0=Long, -1.0=Short) ※現在は方向に関係なく判定

        Returns:
            True: エントリー許可, False: エントリー拒否
        """
        # ML予測器が設定されていない場合はエントリーを許可
        if self.strategy.ml_predictor is None:
            return True

        # ML予測器が学習済みでない場合はエントリーを許可
        try:
            if hasattr(self.strategy.ml_predictor, "is_trained"):
                if not self.strategy.ml_predictor.is_trained():
                    logger.debug("ML予測器未学習: エントリー許可")
                    return True
        except Exception as e:
            logger.warning(f"ML学習状態チェックエラー: {e}")
            return True

        try:
            # 1. 事前計算済みの特徴量から現在の行を取得
            features = None

            if (
                hasattr(self.strategy, "_precomputed_features")
                and self.strategy._precomputed_features is not None
            ):
                # 高速化: タイムスタンプ検索(loc)ではなく整数インデックス(iloc)を使用
                idx = len(self.strategy.data) - 1
                if 0 <= idx < len(self.strategy._precomputed_features):
                    features = self.strategy._precomputed_features.iloc[[idx]]

            # 2. キャッシュがない場合はフォールバック（低速）
            if features is None:
                features = self.prepare_current_features()

            # 3. ML予測を実行
            prediction = self.strategy.ml_predictor.predict(features)

            # ダマシ予測モデルの判定
            # is_valid: エントリーが有効である確率 (0.0-1.0)
            # 閾値以上であればエントリーを許可
            is_valid = prediction.get("is_valid", 0.5)
            allowed = is_valid >= self.strategy.ml_filter_threshold

            return allowed

        except Exception as e:
            # 予測エラー時はエントリーを許可（フェイルセーフ）
            logger.warning(f"ML予測エラー（フェイルセーフ適用）: {e}")
            return True

    def prepare_current_features(self) -> pd.DataFrame:
        """
        現在のバーからML用特徴量を準備

        HybridFeatureAdapterに委譲して一貫性を確保します。
        """
        try:
            from ..core.hybrid_feature_adapter import HybridFeatureAdapter

            # アダプターの初期化（まだ存在しない場合）
            if not hasattr(self.strategy, "feature_adapter"):
                self.strategy.feature_adapter = HybridFeatureAdapter()

            # 現在のバーのOHLCVデータを取得
            # backtesting.pyのdataオブジェクトをDataFrameに変換（直近のみ）
            lookback = 30  # 特徴量計算に必要な最低限のルックバック
            data_len = len(self.strategy.data)
            actual_lookback = min(lookback, data_len)

            # 効率のため必要な分だけスライス
            subset = self.strategy.data.df.iloc[-actual_lookback:].copy()

            # 既存のカラム名を小文字に統一（アダプタの期待に合わせる）
            subset.columns = [c.lower() for c in subset.columns]

            # アダプタを使用して特徴量変換
            features_df = self.strategy.feature_adapter.gene_to_features(
                gene=self.strategy.gene,
                ohlcv_data=subset,
                apply_preprocessing=False,  # 推論時は基本的なクリーニングのみ
            )

            # 直近の1行のみを返す
            return features_df.iloc[[-1]]

        except Exception as e:
            logger.error(f"特徴量準備エラー (Adapter使用): {e}")
            # フォールバック（最小限の構造を持つDataFrame）
            return pd.DataFrame(
                [{"close": self.strategy.data.Close[-1], "indicator_count": 1}]
            )
