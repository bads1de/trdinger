"""
ML共通ユーティリティ

データ検証、時系列処理、ボラティリティ計算など、
ML処理で頻繁に使用される共通ロジックを提供します。
"""

import hashlib
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# --- データ操作・検証 ---


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    DataFrame のデータ型を最適化してメモリ消費量を劇的に削減
    """
    try:
        df = df.copy()
        # float64を一括でfloat32に変換
        float_cols = [c for c in df.columns if df[c].dtype == "float64"]
        for col in float_cols:
            df[col] = df[col].astype("float32")

        # int64を条件付きでint32に変換
        for col in df.columns:
            if df[col].dtype == "int64" and col != "timestamp":
                vals = df[col].values.tolist()
                if not vals:
                    continue
                c_min, c_max = min(vals), max(vals)
                if c_min >= -2147483648 and c_max <= 2147483647:
                    df[col] = df[col].astype("int32")
        return df
    except Exception as e:
        logger.warning(f"データ型最適化エラー: {e}")
        return df


def generate_cache_key(
    ohlcv_data: pd.DataFrame,
    funding_rate_data: Optional[pd.DataFrame] = None,
    open_interest_data: Optional[pd.DataFrame] = None,
    long_short_ratio_data: Optional[pd.DataFrame] = None,  # Added
    extra_params: Optional[dict] = None,
) -> str:
    """
    データの内容とパラメータセットから一意なキャッシュキーを生成
    """

    def _hash(obj: Any) -> str:
        try:
            if isinstance(obj, pd.DataFrame):
                summary = f"{obj.shape}_{obj.iloc[0].values.tolist() if len(obj) > 0 else ''}_{obj.iloc[-1].values.tolist() if len(obj) > 1 else ''}"
                return hashlib.md5(summary.encode()).hexdigest()[:8]
            return hashlib.md5(str(obj).encode()).hexdigest()[:8]
        except Exception:
            return "hash_error"

    h1 = _hash(ohlcv_data)
    h2 = _hash(funding_rate_data.shape if funding_rate_data is not None else None)
    h3 = _hash(open_interest_data.shape if open_interest_data is not None else None)
    h4 = _hash(long_short_ratio_data.shape if long_short_ratio_data is not None else None)  # Added
    h5 = _hash(sorted(extra_params.items()) if extra_params else None)

    return f"features_{h1}_{h2}_{h3}_{h4}_{h5}"


def validate_training_inputs(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame | None = None,
    y_test: pd.Series | None = None,
    log_info: bool = True,
) -> None:
    """学習用データの検証を行う共通関数"""
    if X_train is None or len(X_train) == 0:
        raise ValueError("学習用特徴量データが空です")
    if y_train is None or len(y_train) == 0:
        raise ValueError("学習用ターゲットデータが空です")
    if len(X_train) != len(y_train):
        raise ValueError("特徴量とターゲットの長さが一致しません")

    if log_info:
        logger.info(f"学習データサイズ: {len(X_train)}行")


def prepare_data_for_prediction(
    features_df: pd.DataFrame,
    expected_columns: List[str],
    scaler=None,
) -> pd.DataFrame:
    """予測用のデータを前処理（カラム調整、スケーリング）"""
    try:
        data_dict = {}
        for col in expected_columns:
            if col in features_df.columns:
                data_dict[col] = features_df[col].values
            else:
                data_dict[col] = np.zeros(len(features_df))

        processed_features = pd.DataFrame(data_dict, index=features_df.index)
        processed_features = processed_features.ffill().fillna(0.0)

        if scaler is not None:
            try:
                scaled_values = scaler.transform(processed_features)
                processed_features = pd.DataFrame(
                    scaled_values,
                    columns=expected_columns,
                    index=features_df.index,
                )
            except Exception as e:
                logger.warning(f"スケーリングをスキップ: {e}")

        return processed_features

    except Exception as e:
        logger.error(f"データ前処理エラー: {e}")
        return features_df


def predict_class_from_proba(
    predictions_proba: np.ndarray, threshold: float = 0.5
) -> np.ndarray:
    """モデルの出力確率からバイナリクラスを推定"""
    if predictions_proba.ndim == 2:
        return np.argmax(predictions_proba, axis=1)
    return (predictions_proba > threshold).astype(int)


def get_feature_importance_unified(
    model,
    feature_columns: List[str],
    top_n: int = 10,
) -> Dict[str, float]:
    """様々なモデルから特徴量重要度を統一的に取得"""
    if model is None or not feature_columns:
        return {}

    try:
        importance_scores = None
        if hasattr(model, "feature_importances_"):
            importance_scores = model.feature_importances_
        elif hasattr(model, "feature_importance") and callable(
            model.feature_importance
        ):
            try:
                importance_scores = model.feature_importance(importance_type="gain")
            except Exception:
                try:
                    importance_scores = model.feature_importance()
                except Exception:
                    importance_scores = None

        if importance_scores is not None:
            if hasattr(importance_scores, "tolist") and callable(
                importance_scores.tolist
            ):
                scores = importance_scores.tolist()
            elif isinstance(importance_scores, (list, tuple)):
                scores = importance_scores
            else:
                scores = [float(x) for x in importance_scores]

            if len(scores) != len(feature_columns):
                logger.warning(
                    f"長さ不一致: scores({len(scores)}) != cols({len(feature_columns)})"
                )
                return {}

            feature_importance = {
                feature_columns[i]: float(scores[i])
                for i in range(len(feature_columns))
            }
            sorted_importance = sorted(
                feature_importance.items(), key=lambda x: x[1], reverse=True
            )[:top_n]
            return dict(sorted_importance)

        elif hasattr(model, "get_feature_importance") and callable(
            model.get_feature_importance
        ):
            try:
                res = model.get_feature_importance(top_n=top_n)
                if isinstance(res, dict):
                    return dict(
                        sorted(res.items(), key=lambda x: x[1], reverse=True)[:top_n]
                    )
                return {}
            except TypeError:
                all_imp = model.get_feature_importance()
                if isinstance(all_imp, dict):
                    return dict(
                        sorted(all_imp.items(), key=lambda x: x[1], reverse=True)[
                            :top_n
                        ]
                    )
                return {}
        return {}
    except Exception as e:
        logger.error(f"特徴量重要度取得エラー: {e}")
        return {}


# --- 時系列処理 ---


def infer_timeframe(index: pd.DatetimeIndex) -> str:
    """DatetimeIndex の時間間隔から時間足を自動推定"""
    if len(index) < 2:
        return "1h"
    diffs = index.to_series().diff().dropna()
    if diffs.empty:
        return "1h"

    sec = diffs.mode().iloc[0].total_seconds()
    mapping = {900: "15m", 1800: "30m", 3600: "1h", 14400: "4h", 86400: "1d"}
    if sec in mapping:
        return mapping[sec]

    if sec >= 3600 and (sec / 3600).is_integer():
        return f"{int(sec / 3600)}h"
    if sec >= 60 and (sec / 60).is_integer():
        return f"{int(sec / 60)}m"
    return "1h"


def get_t1_series(
    indices: pd.DatetimeIndex, horizon_n: int, timeframe: Optional[str] = None
) -> pd.Series:
    """PurgedKFold 用の t1（ラベリング終了時刻）を計算"""
    tf = timeframe or infer_timeframe(indices)

    if tf.endswith("m"):
        delta = pd.Timedelta(minutes=int(tf[:-1]) * horizon_n)
    elif tf.endswith("h"):
        delta = pd.Timedelta(hours=int(tf[:-1]) * horizon_n)
    elif tf.endswith("d"):
        delta = pd.Timedelta(days=int(tf[:-1]) * horizon_n)
    else:
        logger.warning(f"Unknown tf format: {tf}, default to 1h")
        delta = pd.Timedelta(hours=horizon_n)

    return pd.Series(indices + delta, index=indices)


# --- 価格計算・ボラティリティ ---


def calculate_price_change(
    series: pd.Series,
    periods: int = 1,
    shift: int = 0,
    fill_na: bool = True,
) -> pd.Series:
    """価格変化率を計算"""
    try:
        if shift != 0:
            pct_change = series.pct_change(periods=periods).shift(shift)
        else:
            pct_change = series.pct_change(periods=periods)

        if fill_na:
            pct_change = pct_change.fillna(0)
        return pct_change
    except Exception as e:
        logger.error(f"価格変化率計算エラー: {e}")
        raise


def calculate_volatility_std(
    returns: pd.Series,
    window: int = 24,
    min_periods: Optional[int] = None,
) -> pd.Series:
    """標準偏差ベースのボラティリティ計算"""
    if len(returns) == 0:
        return pd.Series([], dtype=float)
    if min_periods is None:
        min_periods = window
    return returns.rolling(window=window, min_periods=min_periods).std()


def calculate_volatility_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
    as_percentage: bool = False,
) -> pd.Series:
    """ATRベースのボラティリティ計算"""
    if len(high) == 0:
        return pd.Series([], dtype=float)
    pc = close.shift(1)
    tr = pd.concat([high - low, (high - pc).abs(), (low - pc).abs()], axis=1).max(
        axis=1
    )
    atr = tr.rolling(window=window).mean()
    return atr / close if as_percentage else atr


def calculate_historical_volatility(
    returns: pd.Series,
    window: int = 20,
    annualize: bool = True,
    periods_per_year: int = 252,
) -> pd.Series:
    """年率換算のヒストリカルボラティリティ計算"""
    vol = returns.rolling(window=window).std()
    if annualize:
        vol = vol * np.sqrt(periods_per_year)
    return vol


def calculate_realized_volatility(
    returns: pd.Series,
    window: int = 24,
    periods_per_day: int = 24,
) -> pd.Series:
    """実現ボラティリティ計算"""
    vol = returns.rolling(window=window).std()
    return vol * np.sqrt(periods_per_day)
