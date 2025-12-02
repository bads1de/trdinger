"""共通の特徴量評価ユーティリティ

DBからのOHLCV/FR/OI取得、FeatureEngineeringServiceによる特徴量生成、
forwardリターンターゲット生成、TimeSeriesSplitによるCV評価など、
feature_evaluationスクリプト間で共通する処理を集約する。
"""

from __future__ import annotations

import logging

# プロジェクトルートをパスに追加（スクリプト直実行対応）
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.config.unified_config import unified_config  # type: ignore
from app.services.ml.feature_engineering.feature_engineering_service import (  # type: ignore  # noqa: E501
    FeatureEngineeringService,
)
from app.services.ml.label_generation.presets import (  # type: ignore
    apply_preset_by_name,
)
from database.connection import SessionLocal  # type: ignore
from database.repositories.funding_rate_repository import (  # type: ignore
    FundingRateRepository,
)
from database.repositories.ohlcv_repository import OHLCVRepository  # type: ignore
from database.repositories.open_interest_repository import (  # type: ignore
    OpenInterestRepository,
)
from app.services.ml.cross_validation.purged_kfold import PurgedKFold
from app.services.ml.label_cache import LabelCache
from sklearn.metrics import classification_report

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
        # プロファイルはcalculate_advanced_features()のprofile引数で指定可能
        self.feature_service = FeatureEngineeringService()
        self.label_cache: Optional[LabelCache] = None

    def close(self) -> None:
        self.db.close()

    # ---- データ取得 ----

    def fetch_data(
        self,
        symbol: str = "BTC/USDT:USDT",
        timeframe: str = "1h",
        limit: int = 200000,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> EvaluationData:
        logger.info(
            f"[Common] データ取得開始: {symbol}, timeframe={timeframe}, limit={limit}, start={start_date}, end={end_date}"
        )

        # 日付文字列をdatetimeオブジェクトに変換
        from datetime import datetime

        start_time = datetime.fromisoformat(start_date) if start_date else None
        end_time = datetime.fromisoformat(end_date) if end_date else None

        ohlcv_df = self.ohlcv_repo.get_ohlcv_dataframe(
            symbol=symbol,
            timeframe=timeframe,
            limit=limit,
            start_time=start_time,
            end_time=end_time,
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
                        "funding_timestamp": "timestamp",  # timestampに統一
                        "funding_rate": "funding_rate",
                    },
                    index_column="timestamp",  # インデックス名も統一
                )
                # funding_timestampカラムが残っている場合は削除
                if "funding_timestamp" in fr_df.columns:
                    fr_df = fr_df.drop(columns=["funding_timestamp"])
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
            logger.info(f"OI records retrieved: {len(oi_records) if oi_records else 0}")
            oi_df: Optional[pd.DataFrame]
            if oi_records:
                oi_df = pd.DataFrame(
                    [
                        {
                            "timestamp": r.data_timestamp,
                            "open_interest_value": r.open_interest_value,
                        }
                        for r in oi_records
                    ]
                )
                oi_df.set_index("timestamp", inplace=True)
                logger.info(f"OI DataFrame created with shape: {oi_df.shape}")
            else:
                logger.warning("No OI records found for the specified period")
                oi_df = None
        except Exception as e:  # pragma: no cover
            logger.error(f"OI取得エラー: {e}", exc_info=True)
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

        # データ存在確認のログ
        logger.info(f"OHLCV data shape: {data.ohlcv.shape}")
        if data.fr is not None:
            logger.info(
                f"FR data shape: {data.fr.shape}, columns: {data.fr.columns.tolist()}"
            )
        else:
            logger.warning("FR data is None")

        if data.oi is not None:
            logger.info(
                f"OI data shape: {data.oi.shape}, columns: {data.oi.columns.tolist()}"
            )
        else:
            logger.warning("OI data is None")

        original_crypto = self.feature_service.crypto_features
        # original_advanced = self.feature_service.advanced_features

        try:
            if skip_crypto_and_advanced:
                self.feature_service.crypto_features = None
                # self.feature_service.advanced_features = None

            # キャッシュをクリアして新しい特徴量計算を強制
            self.feature_service.clear_cache()
            logger.info("Feature service cache cleared")

            features_df = self.feature_service.calculate_advanced_features(
                ohlcv_data=data.ohlcv,
                funding_rate_data=data.fr,
                open_interest_data=data.oi,
            )

            logger.info(f"Generated features columns: {features_df.columns.tolist()}")
        finally:
            self.feature_service.crypto_features = original_crypto
            # self.feature_service.advanced_features = original_advanced

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
        """後方互換性のための旧メソッド（非推奨）

        注意: このメソッドは後方互換性のために残されていますが、
        新しいコードでは create_labels_from_config() を使用してください。
        """
        if close.empty:
            raise ValueError("closeシリーズが空です")
        return close.pct_change(periods).shift(-periods)

    def create_labels_from_config(
        self,
        ohlcv_df: pd.DataFrame,
        preset_name: Optional[str] = None,
        price_column: str = "close",
    ) -> pd.Series:
        """統一設定から分類ラベルを生成

        unified_configのlabel_generation設定を使用してラベルを生成します。
        プリセット名を指定することで、設定を上書きすることも可能です。

        Args:
            ohlcv_df: OHLCVデータフレーム
            preset_name: プリセット名（Noneの場合は設定から読み込み）
            price_column: 価格カラム名（デフォルト: "close"）

        Returns:
            pd.Series: "UP" / "RANGE" / "DOWN" の分類ラベル

        Raises:
            ValueError: データフレームが不正な場合

        Example:
            >>> evaluator = CommonFeatureEvaluator()
            >>> labels = evaluator.create_labels_from_config(ohlcv_df)
            >>> # または特定のプリセットを使用
            >>> labels = evaluator.create_labels_from_config(
            ...     ohlcv_df, preset_name="1h_4bars"
            ... )
        """
        if ohlcv_df.empty:
            raise ValueError("OHLCVデータフレームが空です")

        if price_column not in ohlcv_df.columns:
            raise ValueError(
                f"価格カラム '{price_column}' がデータフレームに存在しません"
            )

        # ラベル生成設定を取得
        label_config = unified_config.ml.training.label_generation

        # プリセット使用の場合
        if label_config.use_preset or preset_name is not None:
            preset_to_use = preset_name or label_config.default_preset
            logger.info(f"プリセット '{preset_to_use}' を使用してラベル生成")

            labels, preset_info = apply_preset_by_name(
                df=ohlcv_df,
                preset_name=preset_to_use,
                price_column=price_column,
            )

            logger.info(f"ラベル生成完了: {preset_info['description']}")
            return labels

    def get_label_config_info(self) -> Dict[str, any]:
        """現在のラベル生成設定情報を取得

        Returns:
            Dict: ラベル生成設定の詳細情報
        """
        label_config = unified_config.ml.training.label_generation

        info = {
            "use_preset": label_config.use_preset,
            "timeframe": label_config.timeframe,
            "horizon_n": label_config.horizon_n,
            "threshold": label_config.threshold,
            "price_column": label_config.price_column,
            "threshold_method": label_config.threshold_method,
        }

        if label_config.use_preset:
            info["preset_name"] = label_config.default_preset

        return info

    def generate_pipeline_compatible_labels(
        self,
        ohlcv_df: pd.DataFrame,
        horizon_n: int = 4,
        pt_factor: float = 1.0,
        sl_factor: float = 1.0,
        use_atr: bool = True,
        atr_period: int = 14,
        binary_label: bool = True,
        timeframe: str = "1h",
    ) -> pd.Series:
        """パイプライン互換のラベル生成（LabelCache使用）

        Args:
            ohlcv_df: OHLCVデータ
            horizon_n: ホライズン
            pt_factor: 利確係数
            sl_factor: 損切係数
            use_atr: ATR使用フラグ
            atr_period: ATR期間
            binary_label: 2値分類フラグ
            timeframe: 時間足

        Returns:
            pd.Series: ラベル
        """
        if self.label_cache is None:
            self.label_cache = LabelCache(ohlcv_df)
        else:
            # データフレームが更新されている可能性があるため再設定
            self.label_cache.ohlcv_df = ohlcv_df

        return self.label_cache.get_labels(
            horizon_n=horizon_n,
            threshold_method="TRIPLE_BARRIER",
            threshold=0.0,
            timeframe=timeframe,
            price_column="close",
            pt_factor=pt_factor,
            sl_factor=sl_factor,
            use_atr=use_atr,
            atr_period=atr_period,
            binary_label=binary_label,
        )

    def get_t1_for_purged_kfold(
        self, indices: pd.DatetimeIndex, horizon_n: int, timeframe: str = "1h"
    ) -> pd.Series:
        """PurgedKFold用のt1を取得"""
        if self.label_cache is None:
            # ダミーのLabelCacheを作成してメソッドを利用
            self.label_cache = LabelCache(pd.DataFrame())

        return self.label_cache.get_t1(indices, horizon_n, timeframe)

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

    @staticmethod
    def purged_kfold_cv(
        X: pd.DataFrame,
        y: pd.Series,
        t1: pd.Series,
        n_splits: int = 5,
        embargo_pct: float = 0.01,
        use_meta_labeling_metrics: bool = False,
    ) -> Dict[str, float]:
        """PurgedKFoldによるCV評価（パイプライン準拠 + メタラベリング対応）

        Args:
            X: 特徴量DataFrame
            y: ターゲットSeries
            t1: ラベルの終了時刻Series
            n_splits: CV分割数
            embargo_pct: エンバーゴ比率
            use_meta_labeling_metrics: メタラベリング評価指標を使用するか

        Returns:
            Dict[str, float]: 評価指標の辞書
        """
        if X.empty:
            raise ValueError("特徴量が空です")

        cv = PurgedKFold(n_splits=n_splits, t1=t1, pct_embargo=embargo_pct)

        # シンプルなモデル（LightGBM）で評価
        import lightgbm as lgb

        f1_scores = []
        precisions = []
        recalls = []
        pipeline_scores = []

        # メタラベリング用追加指標
        if use_meta_labeling_metrics:
            win_rates = []
            expected_values = []
            signal_adoption_rates = []

        start_time = time.perf_counter()

        for train_idx, val_idx in cv.split(X, y):
            if len(train_idx) == 0 or len(val_idx) == 0:
                continue

            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # バランス調整
            if len(y_train.unique()) > 1:
                class_weight = "balanced"
            else:
                class_weight = None

            model = lgb.LGBMClassifier(
                n_estimators=100,
                random_state=42,
                verbosity=-1,
                class_weight=class_weight,
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            # 評価指標計算
            report = classification_report(
                y_val, y_pred, output_dict=True, zero_division=0
            )

            # 2値分類の "1" (Positive) のスコアを参照
            if "1" in report:
                metrics = report["1"]
                f1 = metrics["f1-score"]
                prec = metrics["precision"]
                rec = metrics["recall"]
                n_trades = y_pred.sum()

                # パイプラインスコア: F1 * log(1 + trades)
                # ただし精度要件あり
                if prec < 0.55 or n_trades < 10:
                    p_score = 0.0
                else:
                    p_score = f1 * np.log1p(n_trades)

                # メタラベリング指標の計算
                if use_meta_labeling_metrics:
                    # Win Rate = Precision（勝率）
                    win_rate = prec

                    # Expected Value（期待値）
                    # 勝ち時+1、負け時-1と仮定
                    expected_value = (prec * 1.0) + ((1 - prec) * -1.0)

                    # Signal Adoption Rate（シグナル採択率）
                    signal_adoption_rate = (
                        n_trades / len(y_val) if len(y_val) > 0 else 0.0
                    )

                    win_rates.append(win_rate)
                    expected_values.append(expected_value)
                    signal_adoption_rates.append(signal_adoption_rate)
            else:
                # クラス1が存在しない場合など
                f1 = 0.0
                prec = 0.0
                rec = 0.0
                p_score = 0.0

                if use_meta_labeling_metrics:
                    win_rates.append(0.0)
                    expected_values.append(-1.0)
                    signal_adoption_rates.append(0.0)

            f1_scores.append(f1)
            precisions.append(prec)
            recalls.append(rec)
            pipeline_scores.append(p_score)

        elapsed = time.perf_counter() - start_time

        result = {
            "cv_f1": float(np.mean(f1_scores)) if f1_scores else 0.0,
            "cv_precision": float(np.mean(precisions)) if precisions else 0.0,
            "cv_recall": float(np.mean(recalls)) if recalls else 0.0,
            "cv_pipeline_score": (
                float(np.mean(pipeline_scores)) if pipeline_scores else 0.0
            ),
            "train_time_sec": float(elapsed),
        }

        # メタラベリング指標を追加
        if use_meta_labeling_metrics:
            result.update(
                {
                    "cv_win_rate": float(np.mean(win_rates)) if win_rates else 0.0,
                    "cv_expected_value": (
                        float(np.mean(expected_values)) if expected_values else -1.0
                    ),
                    "cv_signal_adoption_rate": (
                        float(np.mean(signal_adoption_rates))
                        if signal_adoption_rates
                        else 0.0
                    ),
                }
            )

            logger.info("📊 Meta-Labeling Evaluation Results:")
            logger.info(f"  Win Rate (Precision): {result['cv_win_rate']:.2%}")
            logger.info(f"  Expected Value: {result['cv_expected_value']:.4f}")
            logger.info(
                f"  Signal Adoption Rate: {result['cv_signal_adoption_rate']:.2%}"
            )

        return result
