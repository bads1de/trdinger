"""
ML改善のテストスクリプト

動的ラベル生成、特徴量拡張、LightGBMトレーナーの改善をテストします。
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.connection import SessionLocal
from database.repositories.ohlcv_repository import OHLCVRepository
from app.core.services.ml.ml_training_service import MLTrainingService
from app.core.utils.label_generation import LabelGenerator, ThresholdMethod

# ログ設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_label_generation():
    """ラベル生成機能のテスト"""
    logger.info("=== ラベル生成機能テスト ===")

    try:
        with SessionLocal() as db:
            ohlcv_repo = OHLCVRepository(db)

            # テスト用データを取得
            ohlcv_data = ohlcv_repo.get_ohlcv_data(
                symbol="BTC/USDT:USDT",
                timeframe="1h",
                start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_time=datetime(2024, 2, 1, tzinfo=timezone.utc),
            )

            if not ohlcv_data:
                logger.error("テスト用データが見つかりません")
                return False

            # DataFrameに変換
            df = pd.DataFrame(
                [
                    {
                        "timestamp": data.timestamp,
                        "close": data.close,
                    }
                    for data in ohlcv_data
                ]
            )

            df.set_index("timestamp", inplace=True)

            # ラベル生成器をテスト
            label_generator = LabelGenerator()

            # 複数の方法をテスト
            methods_to_test = [
                (ThresholdMethod.FIXED, {"threshold": 0.02}),
                (ThresholdMethod.STD_DEVIATION, {"std_multiplier": 0.25}),
                (ThresholdMethod.QUANTILE, {}),
                (ThresholdMethod.ADAPTIVE, {}),
            ]

            for method, params in methods_to_test:
                try:
                    logger.info(f"テスト中: {method.value}")
                    labels, threshold_info = label_generator.generate_labels(
                        df["close"], method=method, **params
                    )

                    logger.info(f"  方法: {threshold_info['description']}")
                    logger.info(
                        f"  閾値: {threshold_info['threshold_down']:.6f} ～ {threshold_info['threshold_up']:.6f}"
                    )
                    logger.info(
                        f"  分布: 上昇{threshold_info['up_ratio']*100:.1f}% / 下落{threshold_info['down_ratio']*100:.1f}% / レンジ{threshold_info['range_ratio']*100:.1f}%"
                    )

                    # 分布検証
                    validation_result = LabelGenerator.validate_label_distribution(
                        labels
                    )
                    if validation_result["is_valid"]:
                        logger.info("  ✅ ラベル分布は有効です")
                    else:
                        logger.warning("  ⚠️ ラベル分布に問題があります")
                        for error in validation_result["errors"]:
                            logger.warning(f"    エラー: {error}")

                except Exception as e:
                    logger.error(f"  ❌ {method.value} でエラー: {e}")

            logger.info("✅ ラベル生成機能テスト完了")
            return True

    except Exception as e:
        logger.error(f"ラベル生成機能テストエラー: {e}")
        return False


def test_ml_training_with_improvements():
    """改善されたMLトレーニングのテスト"""
    logger.info("=== 改善されたMLトレーニングテスト ===")

    try:
        with SessionLocal() as db:
            ohlcv_repo = OHLCVRepository(db)

            # テスト用データを取得（少し多めに）
            ohlcv_data = ohlcv_repo.get_ohlcv_data(
                symbol="BTC/USDT:USDT",
                timeframe="1h",
                start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_time=datetime(2024, 3, 1, tzinfo=timezone.utc),
            )

            if not ohlcv_data or len(ohlcv_data) < 1000:
                logger.error(
                    f"十分なテスト用データが見つかりません: {len(ohlcv_data) if ohlcv_data else 0}件"
                )
                return False

            # DataFrameに変換（カラム名は大文字で統一）
            df = pd.DataFrame(
                [
                    {
                        "timestamp": data.timestamp,
                        "Open": data.open,
                        "High": data.high,
                        "Low": data.low,
                        "Close": data.close,
                        "Volume": data.volume,
                    }
                    for data in ohlcv_data
                ]
            )

            df.set_index("timestamp", inplace=True)

            # MLトレーニングサービスを初期化
            ml_service = MLTrainingService()

            # 改善されたパラメータでトレーニング
            training_params = {
                "threshold_method": "std_deviation",  # 動的閾値を使用
                "std_multiplier": 0.25,  # 分析結果から最適な値
                "test_size": 0.2,
                "random_state": 42,
            }

            logger.info("MLモデルトレーニング開始...")
            logger.info(f"使用データ: {len(df)}行")
            logger.info(f"パラメータ: {training_params}")

            result = ml_service.train_model(
                training_data=df,
                save_model=False,  # テストなので保存しない
                **training_params,
            )

            # 結果を確認
            logger.info("✅ MLトレーニング成功!")

            # 安全な値の取得と表示
            accuracy = result.get("accuracy", "N/A")
            if isinstance(accuracy, (int, float)):
                logger.info(f"精度: {accuracy:.4f}")
            else:
                logger.info(f"精度: {accuracy}")

            logger.info(f"クラス数: {result.get('num_classes', 'N/A')}")
            logger.info(f"学習サンプル数: {result.get('train_samples', 'N/A')}")
            logger.info(f"テストサンプル数: {result.get('test_samples', 'N/A')}")

            # 分類レポートを表示
            if "classification_report" in result:
                class_report = result["classification_report"]
                logger.info("分類レポート:")
                for class_name, metrics in class_report.items():
                    if isinstance(metrics, dict) and "precision" in metrics:
                        logger.info(
                            f"  クラス {class_name}: precision={metrics['precision']:.3f}, recall={metrics['recall']:.3f}, f1-score={metrics['f1-score']:.3f}"
                        )

            # 特徴量重要度の上位を表示
            if "feature_importance" in result and result["feature_importance"]:
                logger.info("特徴量重要度（上位10）:")
                sorted_features = sorted(
                    result["feature_importance"].items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:10]
                for feature, importance in sorted_features:
                    logger.info(f"  {feature}: {importance:.3f}")

            return True

    except Exception as e:
        logger.error(f"MLトレーニングテストエラー: {e}")
        return False


def test_feature_engineering():
    """特徴量エンジニアリングのテスト"""
    logger.info("=== 特徴量エンジニアリングテスト ===")

    try:
        from app.core.services.ml.feature_engineering.feature_engineering_service import (
            FeatureEngineeringService,
        )

        with SessionLocal() as db:
            ohlcv_repo = OHLCVRepository(db)

            # テスト用データを取得
            ohlcv_data = ohlcv_repo.get_ohlcv_data(
                symbol="BTC/USDT:USDT",
                timeframe="1h",
                start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_time=datetime(2024, 1, 15, tzinfo=timezone.utc),
            )

            if not ohlcv_data:
                logger.error("テスト用データが見つかりません")
                return False

            # DataFrameに変換
            df = pd.DataFrame(
                [
                    {
                        "timestamp": data.timestamp,
                        "Open": data.open,
                        "High": data.high,
                        "Low": data.low,
                        "Close": data.close,
                        "Volume": data.volume,
                    }
                    for data in ohlcv_data
                ]
            )

            df.set_index("timestamp", inplace=True)

            # 特徴量エンジニアリングサービスをテスト
            feature_service = FeatureEngineeringService()

            logger.info("特徴量計算開始...")
            features_df = feature_service.calculate_advanced_features(df)

            logger.info(f"✅ 特徴量計算成功!")
            logger.info(f"元データ: {len(df)}行, {len(df.columns)}列")
            logger.info(
                f"特徴量データ: {len(features_df)}行, {len(features_df.columns)}列"
            )

            # 特徴量名を表示
            feature_names = feature_service.get_feature_names()
            logger.info(f"生成された特徴量数: {len(feature_names)}")

            # 外部市場特徴量とFear & Greed特徴量が含まれているかチェック
            external_features = [
                name
                for name in features_df.columns
                if "SP500" in name or "NASDAQ" in name or "DXY" in name or "VIX" in name
            ]
            fear_greed_features = [
                name for name in features_df.columns if "FG_" in name
            ]

            logger.info(f"外部市場特徴量: {len(external_features)}個")
            if external_features:
                logger.info(f"  例: {external_features[:5]}")

            logger.info(f"Fear & Greed特徴量: {len(fear_greed_features)}個")
            if fear_greed_features:
                logger.info(f"  例: {fear_greed_features[:5]}")

            return True

    except Exception as e:
        logger.error(f"特徴量エンジニアリングテストエラー: {e}")
        return False


def main():
    """メイン関数"""
    logger.info("ML改善テスト開始")

    test_results = []

    # 各テストを実行
    test_results.append(("ラベル生成機能", test_label_generation()))
    test_results.append(("特徴量エンジニアリング", test_feature_engineering()))
    test_results.append(("MLトレーニング", test_ml_training_with_improvements()))

    # 結果サマリー
    logger.info("\n" + "=" * 60)
    logger.info("テスト結果サマリー")
    logger.info("=" * 60)

    all_passed = True
    for test_name, result in test_results:
        status = "✅ 成功" if result else "❌ 失敗"
        logger.info(f"{test_name}: {status}")
        if not result:
            all_passed = False

    if all_passed:
        logger.info("\n🎉 すべてのテストが成功しました！")
    else:
        logger.error("\n⚠️ 一部のテストが失敗しました。")

    return all_passed


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"テスト実行エラー: {e}")
        sys.exit(1)
