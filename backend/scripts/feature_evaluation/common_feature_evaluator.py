"""共通の特徴量評価ユーティリティ

DBからのOHLCV/FR/OI取得、FeatureEngineeringServiceによる特徴量生成、
forwardリターンターゲット生成、TimeSeriesSplitによるCV評価など、
feature_evaluationスクリプト間で共通する処理を集約する。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

# プロジェクトルートをパスに追加（スクリプト直実行対応）
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services.ml.feature_engineering.feature_engineering_service import (  # type: ignore  # noqa: E501
    FeatureEngineeringService,
)
from database.connection import SessionLocal  # type: ignore
from database.repositories.funding_rate_repository import (  # type: ignore
    FundingRateRepository,
)
from database.repositories.ohlcv_repository import OHLCVRepository  # type: ignore
from database.repositories.open_interest_repository import (  # type: ignore
    OpenInterestRepository,
)

logger = logging.getLogger(__name__)


@dataclass
class EvaluationData:
    ohlcv: pd.DataFrame
    fr: Optional[pd.DataFrame]
    oi: Optional[pd.DataFrame]


class CommonFeatureEvaluator:
    """特徴量評価用の共通ユーティリティクラス"""

    def __init__(self) -> None:
        self.db = SessionLocal()
        self.ohlcv_repo = OHLCVRepository(self.db)
        self.fr_repo = FundingRateRepository(self.db)
        self.oi_repo = OpenInterestRepository(self.db)
        # プロファイルは必要に応じて切り替え可能（デフォルトはresearch）
        self.feature_service = FeatureEngineeringService(profile="research")

    def close(self) -> None:
        self.db.close()

    # ---- データ取得 ----

    def fetch_data(
        self,
        symbol: str = "BTC/USDT:USDT",
        timeframe: str = "1h",
        limit: int = 2000,
    ) -> EvaluationData:
        logger.info(f"[Common] データ取得開始: {symbol}, timeframe={timeframe}, limit={limit}")

        ohlcv_df = self.ohlcv_repo.get_ohlcv_dataframe(
            symbol=symbol,
            timeframe=timeframe,
            limit=limit,
        )
        if ohlcv_df.empty:
            logger.warning("OHLCVデータが空です")
            return EvaluationData(pd.DataFrame(), None, None)

        start_time = ohlcv_df.index.min()
        end_time = ohlcv_df.index.max()

        try:
            fr_records = self.fr_repo.get_funding_rate_data(
                symbol=symbol,
                start_time=start_time,
                end_time=end_time,
            )
            fr_df: Optional[pd.DataFrame]
            if fr_records:
                fr_df = self.fr_repo.to_dataframe(
                    records=fr_records,
                    column_mapping={
                        "funding_timestamp": "funding_timestamp",
                        "funding_rate": "funding_rate",
                    },
                    index_column="funding_timestamp",
                )
            else:
                fr_df = None
        except Exception as e:  # pragma: no cover - ロバストネス目的
            logger.warning(f"FR取得エラー: {e}")
            fr_df = None

        try:
            oi_records = self.oi_repo.get_open_interest_data(
                symbol=symbol,
                start_time=start_time,
                end_time=end_time,
            )
            oi_df: Optional[pd.DataFrame]
            if oi_records:
                oi_df = pd.DataFrame(
                    [
                        {
                            "data_timestamp": r.data_timestamp,
                            "open_interest_value": r.open_interest_value,
                        }
                        for r in oi_records
                    ]
                )
                oi_df.set_index("data_timestamp", inplace=True)
            else:
                oi_df = None
        except Exception as e:  # pragma: no cover
            logger.warning(f"OI取得エラー: {e}")
            oi_df = None

        return EvaluationData(ohlcv=ohlcv_df, fr=fr_df, oi=oi_df)

    # ---- 特徴量生成 ----

    def build_basic_features(
        self,
        data: EvaluationData,
        skip_crypto_and_advanced: bool = False,
    ) -> pd.DataFrame:
        """FeatureEngineeringServiceを使って特徴量を生成する共通処理。"""
        if data.ohlcv.empty:
            return pd.DataFrame()

        original_crypto = self.feature_service.crypto_features
        original_advanced = self.feature_service.advanced_features

        try:
            if skip_crypto_and_advanced:
                self.feature_service.crypto_features = None
                self.feature_service.advanced_features = None

            features_df = self.feature_service.calculate_advanced_features(
                ohlcv_data=data.ohlcv,
                funding_rate_data=data.fr,
                open_interest_data=data.oi,
            )
        finally:
            self.feature_service.crypto_features = original_crypto
            self.feature_service.advanced_features = original_advanced

        return features_df

    def drop_ohlcv_columns(
        self,
        features_df: pd.DataFrame,
        keep_close: bool = False,
    ) -> pd.DataFrame:
        if features_df.empty:
            return features_df

        drop_cols: List[str] = ["open", "high", "low", "volume"]
        if not keep_close:
            drop_cols.append("close")

        return features_df[[c for c in features_df.columns if c not in drop_cols]]

    # ---- ターゲット生成 ----

    @staticmethod
    def create_forward_return_target(
        close: pd.Series,
        periods: int = 1,
    ) -> pd.Series:
        if close.empty:
            raise ValueError("closeシリーズが空です")
        return close.pct_change(periods).shift(-periods)

    # ---- CV 評価 ----

    @staticmethod
    def time_series_cv(
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
    ) -> Dict[str, float]:
        if X.empty:
            raise ValueError("特徴量が空です")
        if len(X) != len(y):
            raise ValueError("特徴量とターゲットの長さが一致しません")

        tscv = TimeSeriesSplit(n_splits=n_splits)

        rmses: List[float] = []
        maes: List[float] = []
        r2s: List[float] = []
        start_time = time.perf_counter()

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # シンプルな線形モデルでのテスト用; 実際のモデル学習は呼び出し側で行う
            from sklearn.linear_model import LinearRegression

            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            rmse = mean_squared_error(y_test, y_pred, squared=False)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            rmses.append(rmse)
            maes.append(mae)
            r2s.append(r2)

        elapsed = time.perf_counter() - start_time

        return {
            "cv_rmse": float(np.mean(rmses)),
            "cv_rmse_std": float(np.std(rmses)),
            "cv_mae": float(np.mean(maes)),
            "cv_mae_std": float(np.std(maes)),
            "cv_r2": float(np.mean(r2s)),
            "cv_r2_std": float(np.std(r2s)),
            "train_time_sec": float(elapsed),
        }
