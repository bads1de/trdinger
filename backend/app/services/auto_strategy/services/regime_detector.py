"""
レジーム検知サービス

HMM (Hidden Markov Model) を使用して市場レジームを検知します。
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from hmmlearn import hmm
from pydantic import BaseModel, ValidationError

from app.utils.error_handler import safe_operation

logger = logging.getLogger(__name__)


class OHLCVData(BaseModel):
    """OHLCVデータモデル"""

    open: float
    high: float
    low: float
    close: float
    volume: float
    funding_rate: Optional[float] = None


class RegimeDetector:
    """
    レジーム検知器

    HMMを使用して市場レジームを検知します。
    3状態: 0=トレンド, 1=レンジ, 2=高ボラ
    """

    def __init__(self, config):
        """
        初期化

        Args:
            config: 設定オブジェクト (n_components, covariance_type, n_iter)
        """
        self.config = config
        self.model = hmm.GaussianHMM(
            n_components=config.n_components,
            covariance_type=config.covariance_type,
            n_iter=config.n_iter,
        )

    @safe_operation(context="レジーム検知", is_api_call=False)
    def detect_regimes(self, data: pd.DataFrame) -> np.ndarray:
        """
        レジーム検知

        Args:
            data: OHLCVデータ (pandas DataFrame)

        Returns:
            レジームラベル配列 (numpy array): 0=トレンド, 1=レンジ, 2=高ボラ
        """
        # データバリデーション
        self._validate_data(data)

        # 特徴量準備
        features = self._prepare_features(data)

        # HMMフィッティング
        try:
            self.model.fit(features)
        except Exception as e:
            logger.error(f"HMMフィッティングエラー: {e}")
            raise RuntimeError(f"HMMモデルのフィッティングに失敗しました: {e}")

        # 状態予測
        regimes = self.model.predict(features)

        logger.info(f"レジーム検知完了: {len(regimes)} サンプル")
        return regimes

    def _validate_data(self, data: pd.DataFrame):
        """
        データバリデーション

        Args:
            data: 検証するDataFrame

        Raises:
            ValueError: データが無効な場合
        """
        if data.empty:
            raise ValueError("データが空です")

        required_columns = ["open", "high", "low", "close", "volume"]
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            msg = f"必要なカラムが不足しています: {missing_columns}"
            raise ValueError(msg)

        # NaNまたは無限大のチェック
        if data.isnull().any().any() or np.isinf(data.values).any():
            raise ValueError("データにNaNまたは無限大が含まれています")

        # Pydanticバリデーション (サンプルチェック)
        sample_data = data.iloc[0].to_dict()
        try:
            OHLCVData(**sample_data)
        except ValidationError as e:
            raise ValueError(f"データ形式が無効です: {e}")

    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        特徴量準備

        Args:
            data: OHLCVデータ

        Returns:
            特徴量配列 (numpy array)
        """
        # リターン計算
        returns = data["close"].pct_change().fillna(0).values

        # ATR計算 (ボラティリティ)
        try:
            import talib

            atr = talib.ATR(
                data["high"].values,
                data["low"].values,
                data["close"].values,
                timeperiod=14,
            )
        except ImportError:
            logger.warning("TA-Libが利用できないため、簡易ボラティリティ計算を使用")
            # 簡易ATR (TRの平均)
            tr = np.maximum(
                data["high"] - data["low"],
                np.maximum(
                    np.abs(data["high"] - data["close"].shift(1)),
                    np.abs(data["low"] - data["close"].shift(1)),
                ),
            ).fillna(0)
            atr = tr.rolling(14).mean().fillna(0).values

        atr_pct_change = pd.Series(atr).pct_change().fillna(0).values

        # ボリューム変化
        volume_pct_change = data["volume"].pct_change().fillna(0).values

        # ファンディングレート変化 (オプション)
        if "funding_rate" in data.columns:
            funding_change = data["funding_rate"].pct_change().fillna(0).values
            features = np.column_stack([
                returns,
                atr_pct_change,
                volume_pct_change,
                funding_change,
            ])
        else:
            features = np.column_stack([
                returns,
                atr_pct_change,
                volume_pct_change,
            ])

        return features
