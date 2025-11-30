"""
ML共通ユーティリティ関数

データ検証、ログ出力など、ML処理で頻繁に使用される共通ロジックを提供します。
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    データ型を最適化してメモリ使用量を削減

    Args:
        df: 最適化するDataFrame

    Returns:
        最適化されたDataFrame
    """
    try:
        optimized_df = df.copy()

        for col in optimized_df.columns:
            if col == "timestamp":
                continue

            if optimized_df[col].dtype == "float64":
                # float64をfloat32に変換（精度は十分）
                optimized_df[col] = optimized_df[col].astype("float32")
            elif optimized_df[col].dtype == "int64":
                # int64をint32に変換（範囲が十分な場合）
                col_min = float(optimized_df[col].min())
                col_max = float(optimized_df[col].max())
                if col_min >= -2147483648 and col_max <= 2147483647:
                    optimized_df[col] = optimized_df[col].astype("int32")

        return optimized_df

    except Exception as e:
        logger.warning(f"データ型最適化エラー: {e}")
        return df


def generate_cache_key(
    ohlcv_data: pd.DataFrame,
    funding_rate_data: pd.DataFrame | None = None,
    open_interest_data: pd.DataFrame | None = None,
    extra_params: dict | None = None,
) -> str:
    """
    データとパラメータからキャッシュキーを生成

    Args:
        ohlcv_data: OHLCV価格データ
        funding_rate_data: ファンディングレートデータ（オプション）
        open_interest_data: 建玉残高データ（オプション）
        extra_params: その他のパラメータ（辞書）

    Returns:
        生成されたキャッシュキー文字列
    """
    import hashlib

    try:
        # pandas.util.hash_pandas_object はインデックスと値をハッシュ化する
        ohlcv_hash = hashlib.md5(
            pd.util.hash_pandas_object(ohlcv_data, index=True).values.tobytes()
        ).hexdigest()[:8]
    except Exception:
        # フォールバック
        data_str = (
            str(ohlcv_data.shape)
            + str(ohlcv_data.iloc[0].values)
            + str(ohlcv_data.iloc[-1].values)
        )
        ohlcv_hash = hashlib.md5(data_str.encode()).hexdigest()[:8]

    fr_hash = hashlib.md5(
        str(
            funding_rate_data.shape if funding_rate_data is not None else "None"
        ).encode()
    ).hexdigest()[:8]

    oi_hash = hashlib.md5(
        str(
            open_interest_data.shape if open_interest_data is not None else "None"
        ).encode()
    ).hexdigest()[:8]

    params_hash = hashlib.md5(
        str(
            sorted(extra_params.items()) if extra_params is not None else "None"
        ).encode()
    ).hexdigest()[:8]

    return f"features_{ohlcv_hash}_{fr_hash}_{oi_hash}_{params_hash}"


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

    Args:
        X_train: 学習用特徴量
        y_train: 学習用ターゲット
        X_test: テスト用特徴量（オプション）
        y_test: テスト用ターゲット（オプション）
        log_info: データサイズをログ出力するか

    Raises:
        ValueError: データが無効な場合
    """
    # 入力データの検証
    if X_train is None or X_train.empty:
        raise ValueError("学習用特徴量データが空です")
    if y_train is None or len(y_train) == 0:
        raise ValueError("学習用ターゲットデータが空です")
    if len(X_train) != len(y_train):
        raise ValueError("特徴量とターゲットの長さが一致しません")

    # 情報ログ
    if log_info:
        logger.info(f"学習データサイズ: {len(X_train)}行, {len(X_train.columns)}特徴量")
        logger.info(f"ターゲット分布: {y_train.value_counts().to_dict()}")

        if X_test is not None:
            logger.info(f"テストデータサイズ: {len(X_test)}行")


def get_feature_importance_unified(
    model,
    feature_columns: list[str],
    top_n: int = 10,
) -> dict[str, float]:
    """
    様々なモデルから特徴量重要度を統一的に取得

    Args:
        model: 学習済みモデル
        feature_columns: 特徴量カラムのリスト
        top_n: 上位N個の特徴量を返す

    Returns:
        特徴量重要度の辞書（降順）
    """
    if model is None or not feature_columns:
        logger.warning("モデルまたは特徴量カラムが無効です")
        return {}

    try:
        # LightGBM/XGBoostスタイルのモデル（feature_importanceメソッド）
        if hasattr(model, "feature_importance") and callable(model.feature_importance):
            importance_scores = model.feature_importance(importance_type="gain")
            if len(importance_scores) != len(feature_columns):
                logger.warning("特徴量重要度と特徴量カラム数が一致しません")
                return {}

            feature_importance = dict(zip(feature_columns, importance_scores))
            sorted_importance = sorted(
                feature_importance.items(), key=lambda x: x[1], reverse=True
            )[:top_n]
            return dict(sorted_importance)

        # get_feature_importanceメソッドを持つモデル
        elif hasattr(model, "get_feature_importance") and callable(
            model.get_feature_importance
        ):
            try:
                # top_n引数をサポートしているか試す
                return model.get_feature_importance(top_n)
            except TypeError:
                # top_n引数をサポートしていない場合
                all_importance = model.get_feature_importance()
                if isinstance(all_importance, dict):
                    sorted_importance = sorted(
                        all_importance.items(), key=lambda x: x[1], reverse=True
                    )[:top_n]
                    return dict(sorted_importance)
                return {}

        # feature_importances_属性を持つモデル（scikit-learn style）
        elif hasattr(model, "feature_importances_"):
            importance_scores = model.feature_importances_
            if len(importance_scores) != len(feature_columns):
                logger.warning("特徴量重要度と特徴量カラム数が一致しません")
                return {}

            feature_importance = dict(zip(feature_columns, importance_scores))
            sorted_importance = sorted(
                feature_importance.items(), key=lambda x: x[1], reverse=True
            )[:top_n]
            return dict(sorted_importance)

        else:
            logger.debug("モデルは特徴量重要度をサポートしていません")
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

    Args:
        features_df: 入力特徴量
        expected_columns: 期待されるカラムリスト
        scaler: スケーラー（オプション）

    Returns:
        前処理済みDataFrame
    """
    try:
        # 1. 存在するカラムのみ抽出
        available_columns = [
            col for col in expected_columns if col in features_df.columns
        ]
        processed_features = features_df[available_columns].copy()

        # 2. 欠損カラムを0で補完
        missing_columns = [
            col for col in expected_columns if col not in features_df.columns
        ]

        if missing_columns:
            logger.debug(f"欠損特徴量を補完します: {len(missing_columns)}個")
            missing_df = pd.DataFrame(
                0.0, index=processed_features.index, columns=missing_columns
            )
            processed_features = pd.concat([processed_features, missing_df], axis=1)

        # 3. カラムの順序を学習時と合わせる
        processed_features = processed_features[expected_columns]

        # 4. 欠損値の簡易補完（予測時）
        processed_features = processed_features.ffill().fillna(0)

        # 5. スケーリング
        if scaler is not None:
            try:
                processed_features = pd.DataFrame(
                    scaler.transform(processed_features),
                    columns=processed_features.columns,
                    index=processed_features.index,
                )
            except Exception as e:
                logger.warning(f"スケーリングをスキップ: {e}")

        return processed_features

    except Exception as e:
        logger.error(f"データ前処理エラー: {e}")
        return features_df
