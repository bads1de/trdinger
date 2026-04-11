"""
ML共通ユーティリティ

データ検証、時系列処理、ボラティリティ計算など、
ML処理で頻繁に使用される共通ロジックを提供します。
"""

import glob
import hashlib
import logging
import os
from typing import Any, Dict, List, Optional, cast

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    DataFrame のデータ型を最適化してメモリ消費量を劇的に削減

    float64をfloat32に、int64を条件付きでint32に変換します。
    これによりメモリ使用量を大幅に削減できます。

    Args:
        df: 最適化対象のDataFrame

    Returns:
        pd.DataFrame: データ型が最適化されたDataFrame

    変換ルール:
        - float64 -> float32（全カラム）
        - int64 -> int32（値がint32の範囲内の場合、timestampカラムを除く）

    Note:
        エラーが発生した場合は元のDataFrameを返します。
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
                c_min = df[col].min()
                c_max = df[col].max()
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
    long_short_ratio_data: Optional[pd.DataFrame] = None,
    extra_params: Optional[dict] = None,
) -> str:
    """
    データの内容とパラメータセットから一意なキャッシュキーを生成

    DataFrameの内容（データとインデックス）、カラム名、
    追加パラメータからMD5ハッシュを生成してキャッシュキーを作成します。

    Args:
        ohlcv_data: OHLCVデータ（必須）
        funding_rate_data: ファンディングレートデータ（オプション）
        open_interest_data: オープンインタレストデータ（オプション）
        long_short_ratio_data: ロング/ショート比率データ（オプション）
        extra_params: 追加パラメータ辞書（オプション）

    Returns:
        str: キャッシュキー文字列（形式: features_h1_h2_h3_h4_h5）

    Note:
        DataFrameのハッシュはデータ内容とインデックス、カラム名の両方を考慮します。
    """

    def _hash(obj: Any) -> str:
        try:
            if isinstance(obj, pd.DataFrame):
                # 1. データ内容とインデックスのハッシュ
                data_hash = pd.util.hash_pandas_object(obj, index=True).values.tobytes()  # type: ignore[reportAttributeAccessIssue]
                # 2. カラム名のハッシュ（カラム名が変われば結果も変わる可能性があるため）
                col_hash = str(list(obj.columns)).encode()

                combined = data_hash + col_hash
                return hashlib.md5(combined).hexdigest()[:8]
            return hashlib.md5(str(obj).encode()).hexdigest()[:8]
        except Exception:
            return "hash_error"

    h1 = _hash(ohlcv_data)
    h2 = _hash(funding_rate_data)  # 直接渡してハッシュ化
    h3 = _hash(open_interest_data)  # 直接渡してハッシュ化
    h4 = _hash(long_short_ratio_data)  # 直接渡してハッシュ化
    h5 = _hash(sorted(extra_params.items()) if extra_params else None)

    return f"features_{h1}_{h2}_{h3}_{h4}_{h5}"


def collect_unique_files(patterns: List[str]) -> List[str]:
    """
    複数パターンに一致するファイルを重複なく収集する。

    globパターンのリストを受け取り、一致するファイルパスを
    重複なしで収集します。

    Args:
        patterns: globパターンのリスト（例: ["*.py", "test_*.py"]）

    Returns:
        List[str]: 一意なファイルパスのリスト

    Note:
        パスは絶対パスに正規化されて重複チェックされます。
    """
    files: List[str] = []
    seen_files = set()

    for pattern in patterns:
        for file_path in glob.glob(pattern):
            normalized_path = os.path.abspath(file_path)
            if normalized_path in seen_files:
                continue
            seen_files.add(normalized_path)
            files.append(file_path)

    return files


def validate_training_inputs(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame | None = None,
    y_test: pd.Series | None = None,
    log_info: bool = True,
) -> None:
    """
    学習用データの検証を行う共通関数

    学習データが空でないこと、特徴量とターゲットの長さが一致することを検証します。

    Args:
        X_train: 学習用特徴量DataFrame
        y_train: 学習用ターゲットSeries
        X_test: テスト用特徴量DataFrame（オプション）
        y_test: テスト用ターゲットSeries（オプション）
        log_info: 検証情報をログ出力するか（デフォルト: True）

    Raises:
        ValueError: 学習データが空、または特徴量とターゲットの長さが不一致の場合
    """
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
    """
    予測用のデータを前処理（カラム調整、スケーリング）

    期待されるカラムリストに基づいてDataFrameを再構築し、
    スケーラーが提供されている場合はスケーリングを適用します。

    Args:
        features_df: 予測用特徴量DataFrame
        expected_columns: 期待されるカラム名のリスト
        scaler: スケーラーオブジェクト（オプション）

    Returns:
        pd.DataFrame: 前処理されたDataFrame

    処理手順:
        1. 期待されるカラムごとにDataFrameから値を抽出
        2. カラムが存在しない場合はゼロで埋める
        3. 欠損値を前方埋めとゼロ埋めで補完
        4. スケーラーが提供されている場合はスケーリングを適用

    Note:
        スケーリングに失敗した場合はスキップして警告を出力します。
    """
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
    """
    モデルの出力確率からバイナリクラスを推定

    確率値をクラスラベルに変換します。
    2次元配列（マルチクラス）の場合はargmaxを使用し、
    1次元配列（バイナリ）の場合は閾値を使用します。

    Args:
        predictions_proba: モデルの出力確率（1次元または2次元配列）
        threshold: バイナリ分類の閾値（デフォルト: 0.5）

    Returns:
        np.ndarray: 推定されたクラスラベル

    変換ルール:
        - 2次元配列: argmax(axis=1)
        - 1次元配列: (predictions_proba > threshold).astype(int)
    """
    if predictions_proba.ndim == 2:
        return np.argmax(predictions_proba, axis=1)
    return (predictions_proba > threshold).astype(int)


def get_feature_importance_unified(
    model,
    feature_columns: List[str],
    top_n: int = 10,
) -> Dict[str, float]:
    """
    異なる機械学習ライブラリ（LightGBM, XGBoost, Scikit-learn等）の間で異なる特徴量重要度の定義を、
    統一されたフォーマットに変換して取得します。

    主な対応ロジック：
    - `feature_importances_` 属性を持つモデル（Scikit-learn, XGBoostのsklearnラッパー等）からの取得。
    - `feature_importance()` メソッドを持つモデル（LightGBMのBooster等）からの取得。
    - 取得したスコアを特徴量名とマッピングし、降順にソートして上位 `top_n` 件を抽出。

    Args:
        model: 学習済みモデルオブジェクト。
        feature_columns (List[str]): 学習に使用された特徴量名のリスト。順序がモデル内部の状態と一致している必要があります。
        top_n (int): 取得する上位特徴量の数。デフォルトは10。

    Returns:
        Dict[str, float]: 特徴量名をキー、生の重要度スコア（または正規化されたスコア）を値とする辞書。
    """
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
                importance_scores.tolist  # type: ignore[reportAttributeAccessIssue]
            ):
                scores = importance_scores.tolist()  # type: ignore[reportAttributeAccessIssue]
            elif isinstance(importance_scores, (list, tuple)):
                scores = importance_scores
            else:
                scores = [float(x) for x in importance_scores]  # type: ignore

            scores = cast(list, scores)

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


# --- 価格計算・ボラティリティ ---


def calculate_price_change(
    series: pd.Series,
    periods: int = 1,
    shift: int = 0,
    fill_na: bool = True,
) -> pd.Series:
    """
    価格変化率を計算

    指定された期間のパーセント変化率を計算します。

    Args:
        series: 価格データのSeries
        periods: 変化率を計算する期間（デフォルト: 1）
        shift: シフト量（デフォルト: 0）
        fill_na: 欠損値を埋めるか（デフォルト: True）

    Returns:
        pd.Series: 価格変化率のSeries

    Note:
        fill_na=Trueの場合、欠損値は0で埋められます。
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


def calculate_volatility_std(
    returns: pd.Series,
    window: int = 24,
    min_periods: Optional[int] = None,
) -> pd.Series:
    """
    標準偏差ベースのボラティリティ計算

    リターンの移動標準偏差を計算してボラティリティを求めます。

    Args:
        returns: リターンデータのSeries
        window: 移動窓のサイズ（デフォルト: 24）
        min_periods: 最小期間（デフォルト: window）

    Returns:
        pd.Series: ボラティリティのSeries

    Note:
        入力が空の場合は空のSeriesを返します。
    """
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
    """
    ATRベースのボラティリティ計算

    Average True Range（ATR）を計算してボラティリティを求めます。
    True Rangeは以下の最大値です：
    - High - Low
    - |High - 前回のClose|
    - |Low - 前回のClose|

    Args:
        high: 高値のSeries
        low: 安値のSeries
        close: 終値のSeries
        window: 移動窓のサイズ（デフォルト: 14）
        as_percentage: 終値に対するパーセンテージで返すか（デフォルト: False）

    Returns:
        pd.Series: ATRボラティリティのSeries

    Note:
        as_percentage=Trueの場合、ATRを終値で割ってパーセンテージに変換します。
    """
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
    """
    年率換算のヒストリカルボラティリティ計算

    リターンの移動標準偏差を計算し、年率換算します。

    Args:
        returns: リターンデータのSeries
        window: 移動窓のサイズ（デフォルト: 20）
        annualize: 年率換算するか（デフォルト: True）
        periods_per_year: 1年あたりの期間数（デフォルト: 252）

    Returns:
        pd.Series: ヒストリカルボラティリティのSeries

    Note:
        annualize=Trueの場合、標準偏差にsqrt(periods_per_year)を乗算します。
    """
    vol = returns.rolling(window=window).std()
    if annualize:
        vol = vol * np.sqrt(periods_per_year)
    return vol


def calculate_realized_volatility(
    returns: pd.Series,
    window: int = 24,
    periods_per_day: int = 24,
) -> pd.Series:
    """
    実現ボラティリティ計算

    リターンの移動標準偏差を計算し、1日あたりの期間数で換算します。

    Args:
        returns: リターンデータのSeries
        window: 移動窓のサイズ（デフォルト: 24）
        periods_per_day: 1日あたりの期間数（デフォルト: 24）

    Returns:
        pd.Series: 実現ボラティリティのSeries

    Note:
        標準偏差にsqrt(periods_per_day)を乗算して換算します。
    """
    vol = returns.rolling(window=window).std()
    return vol * np.sqrt(periods_per_day)
