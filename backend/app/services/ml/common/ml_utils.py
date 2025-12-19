"""
ML共通ユーティリティ関数

データ検証、ログ出力など、ML処理で頻繁に使用される共通ロジックを提供します。
"""

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


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
                # NumPyのmin/max破損対策のためPython標準のmin/maxを使用
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
    extra_params: Optional[dict] = None,
) -> str:
    """
    データの内容とパラメータセットから一意なキャッシュキーを生成
    """
    import hashlib

    def _hash(obj: Any) -> str:
        try:
            if isinstance(obj, pd.DataFrame):
                # 衝突を避けるため、形状だけでなく末尾の値も加える
                summary = f"{obj.shape}_{obj.iloc[0].values.tolist() if len(obj) > 0 else ''}_{obj.iloc[-1].values.tolist() if len(obj) > 1 else ''}"
                return hashlib.md5(summary.encode()).hexdigest()[:8]
            return hashlib.md5(str(obj).encode()).hexdigest()[:8]
        except Exception:
            return "hash_error"

    h1 = _hash(ohlcv_data)
    h2 = _hash(funding_rate_data.shape if funding_rate_data is not None else None)
    h3 = _hash(open_interest_data.shape if open_interest_data is not None else None)
    h4 = _hash(sorted(extra_params.items()) if extra_params else None)

    return f"features_{h1}_{h2}_{h3}_{h4}"


def calculate_price_change(
    series: pd.Series,
    periods: int = 1,
    shift: int = 0,
    fill_na: bool = True,
) -> pd.Series:
    """
    価格変化率を計算

    Args:
        series: 価格シリーズ
        periods: 期間（デフォルト1）
        shift: シフト量（負の値で未来を参照、正の値で過去を参照）
        fill_na: NaNを0で埋めるか

    Returns:
        価格変化率シリーズ
    """
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


def validate_training_inputs(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame | None = None,
    y_test: pd.Series | None = None,
    log_info: bool = True,
) -> None:
    """
    学習用データの検証を行う共通関数
    """
    # 入力データの検証 (len() は安全)
    if X_train is None or len(X_train) == 0:
        raise ValueError("学習用特徴量データが空です")
    if y_train is None or len(y_train) == 0:
        raise ValueError("学習用ターゲットデータが空です")
    if len(X_train) != len(y_train):
        raise ValueError("特徴量とターゲットの長さが一致しません")

    # 情報ログ
    if log_info:
        logger.info(f"学習データサイズ: {len(X_train)}行")


def get_feature_importance_unified(
    model,
    feature_columns: list[str],
    top_n: int = 10,
) -> dict[str, float]:
    """
    様々なモデルから特徴量重要度を統一的に取得
    """
    if model is None or not feature_columns:
        return {}

    try:
        importance_scores = None
        
        # 1. feature_importances_ 属性 (sklearn style)
        if hasattr(model, "feature_importances_"):
            importance_scores = model.feature_importances_
            
        # 2. LightGBM/XGBoostスタイルのモデル
        elif hasattr(model, "feature_importance") and callable(model.feature_importance):
            # mock環境では引数なしで呼ぶ、または例外を回避
            try:
                importance_scores = model.feature_importance(importance_type="gain")
            except Exception:
                try:
                    importance_scores = model.feature_importance()
                except Exception:
                    importance_scores = None

        if importance_scores is not None:
            # 配列からPythonリストへ安全に変換 (NumPyの破損対策)
            if hasattr(importance_scores, "tolist") and callable(importance_scores.tolist):
                scores = importance_scores.tolist()
            elif isinstance(importance_scores, (list, tuple)):
                scores = importance_scores
            else:
                # イテレータとして処理
                scores = [float(x) for x in importance_scores]

            if len(scores) != len(feature_columns):
                logger.warning(f"長さ不一致: scores({len(scores)}) != cols({len(feature_columns)})")
                return {}

            feature_importance = {feature_columns[i]: float(scores[i]) for i in range(len(feature_columns))}
            sorted_importance = sorted(
                feature_importance.items(), key=lambda x: x[1], reverse=True
            )[:top_n]
            return dict(sorted_importance)

        # 3. get_feature_importance メソッド
        elif hasattr(model, "get_feature_importance") and callable(model.get_feature_importance):
            try:
                # top_n引数をサポートしているか試す
                res = model.get_feature_importance(top_n=top_n)
                # メソッド側でtop_nが処理されていても、念のためこちらでもソートとスライスを行う
                if isinstance(res, dict):
                    sorted_res = sorted(res.items(), key=lambda x: x[1], reverse=True)[:top_n]
                    return dict(sorted_res)
                return {}
            except TypeError:
                # top_n引数をサポートしていない場合
                all_imp = model.get_feature_importance()
                if isinstance(all_imp, dict):
                    sorted_res = sorted(all_imp.items(), key=lambda x: x[1], reverse=True)[:top_n]
                    return dict(sorted_res)
                return {}

        return {}

    except Exception as e:
        logger.error(f"特徴量重要度取得エラー: {e}")
        return {}


def prepare_data_for_prediction(
    features_df: pd.DataFrame,
    expected_columns: list[str],
    scaler=None,
) -> pd.DataFrame:
    """
    予測用のデータを前処理（カラム調整、スケーリング）
    """
    try:
        # インデックス操作によるエラーを避けるため、辞書構築からDFを作成
        data_dict = {}
        for col in expected_columns:
            if col in features_df.columns:
                data_dict[col] = features_df[col].values
            else:
                data_dict[col] = np.zeros(len(features_df))

        processed_features = pd.DataFrame(data_dict, index=features_df.index)

        # 欠損値の簡易補完
        processed_features = processed_features.ffill().fillna(0.0)

        # スケーリング
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
    """
    モデルの出力確率または多クラス確率行列からバイナリクラスを推定

    Args:
        predictions_proba: 予測確率（1次元配列またはクラスごとの2次元配列）
        threshold: 0/1判定の閾値（バイナリの場合）

    Returns:
        0または1の整数配列（多クラスの場合はargmaxされたクラスID）
    """
    if predictions_proba.ndim == 2:
        return np.argmax(predictions_proba, axis=1)
    return (predictions_proba > threshold).astype(int)
