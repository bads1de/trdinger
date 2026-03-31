"""MLトレーニング軽くテストスクリプト"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.config.unified_config import unified_config
from app.services.ml.orchestration.ml_training_orchestration_service import MLTrainingService


def get_ohlcv_data(engine, symbol, timeframe, limit):
    """DBからOHLCVデータを取得してDataFrameに変換"""
    query = text(
        "SELECT timestamp, open, high, low, close, volume "
        "FROM ohlcv_data "
        "WHERE symbol = :symbol AND timeframe = :timeframe "
        "ORDER BY timestamp ASC "
        "LIMIT :limit"
    )
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"symbol": symbol, "timeframe": timeframe, "limit": limit})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")
    return df


def main():
    # DB接続
    # .envのDATABASE_URLを直接読み込む
    import os
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
    db_url = os.environ.get("DATABASE_URL", "sqlite:///./trdinger.db")
    logger.info(f"DB URL: {db_url}")
    engine = create_engine(db_url)

    try:
        # OHLCVデータ取得
        df = get_ohlcv_data(engine, "BTC/USDT:USDT", "1h", 1000)
        logger.info(f"データ取得: {len(df)}行")
        logger.info(f"カラム: {list(df.columns)}")
        logger.info(f"期間: {df.index.min()} ~ {df.index.max()}")

        if len(df) < 100:
            logger.error("データが不足しています")
            return 1

        # MLトレーニング（LightGBM直接使用）
        from app.services.ml.models.lightgbm import LightGBMModel
        import numpy as np

        logger.info("MLトレーニング開始（LightGBM直接）...")

        # 特徴量エンジニアリング
        from app.services.ml.feature_engineering.feature_engineering_service import FeatureEngineeringService
        fe_service = FeatureEngineeringService()
        features_df = fe_service.calculate_advanced_features(df)
        logger.info(f"特徴量計算完了: {features_df.shape}")

        # ターゲット生成（将来リターン）
        df_pct = df["close"].pct_change(1).shift(-1).dropna()
        common_idx = features_df.index.intersection(df_pct.index)
        X = features_df.loc[common_idx]
        y = df_pct.loc[common_idx]
        valid_mask = X.notna().all(axis=1) & y.notna()
        X = X.loc[valid_mask]
        y = y.loc[valid_mask]
        logger.info(f"学習データ: X={X.shape}, y={y.shape}")

        # 特徴量選択（回帰対応の手法を使用）
        from app.services.ml.feature_selection.feature_selector import FeatureSelector
        selector = FeatureSelector(
            staged_methods=["variance", "random_forest", "lasso", "permutation"],
            correlation_threshold=0.95,
            min_relative_importance=0.01,
            importance_threshold=0.001,
            max_features=50,
        )
        X_selected = selector.fit_transform(X, y)
        selected_features = selector.get_feature_names_out()
        logger.info(f"特徴量選択: {X.shape[1]} -> {len(selected_features)}個")

        # 学習/テスト分割
        split_idx = int(len(X_selected) * 0.8)
        X_train, X_test = X_selected.iloc[:split_idx], X_selected.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # モデル学習
        model = LightGBMModel(n_estimators=50, max_depth=5, random_state=42)
        model.task_type = "volatility_regression"
        result = model.fit(X_train, y_train)
        logger.info(f"学習完了: is_trained={model.is_trained}")
        logger.info(f"特徴量数: {len(model.feature_columns) if model.feature_columns else 0}")

        # テスト評価
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        logger.info(f"テスト評価: RMSE={rmse:.6f}, MAE={mae:.6f}, R2={r2:.4f}")

        # モデル保存
        from app.services.ml.models.model_manager import ModelManager
        manager = ModelManager()
        path = manager.save_model(model, "test_lightgbm_model")
        logger.info(f"モデル保存先: {path}")

        return {"success": True, "message": "OK", "model_path": path, "rmse": rmse, "r2": r2}

        logger.info(f"結果: success={result.get('success', False)}")
        logger.info(f"メッセージ: {result.get('message', '')}")

        if result.get("success"):
            metrics = result.get("metrics", {})
            logger.info(f"メトリクス: {metrics}")
            model_path = result.get("model_path")
            if model_path:
                logger.info(f"モデル保存先: {model_path}")
        else:
            logger.error(f"トレーニング失敗: {result.get('error', 'unknown')}")

        return 0 if result.get("success") else 1

    except Exception as e:
        logger.error(f"エラー発生: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
