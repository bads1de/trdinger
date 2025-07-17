#!/usr/bin/env python3
"""
Fear & Greed Index 特徴量エンジニアリング テストスクリプト

実装したFear & Greed Index特徴量エンジニアリング機能の包括的なテストを実行します。
"""

import asyncio
import logging
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

# プロジェクトルートをパスに追加
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from database.connection import SessionLocal
from database.repositories.fear_greed_repository import FearGreedIndexRepository
from database.repositories.external_market_repository import ExternalMarketRepository
from database.repositories.ohlcv_repository import OHLCVRepository
from app.core.services.ml.feature_engineering.feature_engineering_service import (
    FeatureEngineeringService,
)
from app.core.services.ml.feature_engineering.fear_greed_features import (
    FearGreedFeatureCalculator,
)

# ログ設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_sample_ohlcv_data(num_rows: int = 100) -> pd.DataFrame:
    """サンプルOHLCVデータを作成"""
    dates = pd.date_range(start="2024-01-01", periods=num_rows, freq="D")

    # ランダムな価格データを生成
    np.random.seed(42)
    base_price = 50000
    returns = np.random.normal(0, 0.02, num_rows)
    prices = [base_price]

    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))

    data = []
    for i, date in enumerate(dates):
        open_price = prices[i]
        close_price = prices[i] * (1 + np.random.normal(0, 0.01))
        high_price = max(open_price, close_price) * (
            1 + abs(np.random.normal(0, 0.005))
        )
        low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.005)))
        volume = np.random.uniform(1000, 10000)

        data.append(
            {
                "Open": open_price,
                "High": high_price,
                "Low": low_price,
                "Close": close_price,
                "Volume": volume,
            }
        )

    df = pd.DataFrame(data, index=dates)
    return df


def create_sample_fear_greed_data(num_rows: int = 100) -> pd.DataFrame:
    """サンプルFear & Greed Indexデータを作成"""
    dates = pd.date_range(start="2024-01-01", periods=num_rows, freq="D")

    # Fear & Greed Index値を生成（0-100の範囲）
    np.random.seed(42)
    values = []
    current_value = 50  # 中立から開始

    for _ in range(num_rows):
        # ランダムウォークで値を変化
        change = np.random.normal(0, 5)
        current_value = max(0, min(100, current_value + change))
        values.append(int(current_value))

    # 分類を決定
    classifications = []
    for value in values:
        if value <= 25:
            classifications.append("Extreme Fear")
        elif value <= 45:
            classifications.append("Fear")
        elif value <= 54:
            classifications.append("Neutral")
        elif value <= 74:
            classifications.append("Greed")
        else:
            classifications.append("Extreme Greed")

    data = []
    for i, date in enumerate(dates):
        data.append(
            {
                "value": values[i],
                "value_classification": classifications[i],
                "data_timestamp": date,
            }
        )

    df = pd.DataFrame(data)
    df.set_index("data_timestamp", inplace=True)
    return df


async def test_fear_greed_feature_calculator():
    """FearGreedFeatureCalculator単体テスト"""
    logger.info("=== FearGreedFeatureCalculator 単体テスト ===")

    try:
        # テストデータ作成
        ohlcv_data = create_sample_ohlcv_data(50)
        fear_greed_data = create_sample_fear_greed_data(50)
        lookback_periods = {"short_ma": 7, "long_ma": 30, "volatility": 20}

        # 計算クラス初期化
        calculator = FearGreedFeatureCalculator()

        # 特徴量計算実行
        result = calculator.calculate_fear_greed_features(
            ohlcv_data, fear_greed_data, lookback_periods
        )

        # 結果検証
        expected_features = calculator.get_feature_names()
        logger.info(f"期待される特徴量数: {len(expected_features)}")
        logger.info(
            f"実際の特徴量数: {len([col for col in result.columns if col.startswith('FG_')])}"
        )

        # 各特徴量の存在確認
        missing_features = []
        for feature in expected_features:
            if feature not in result.columns:
                missing_features.append(feature)

        if missing_features:
            logger.error(f"❌ 不足している特徴量: {missing_features}")
            return False

        # データ品質チェック
        for feature in expected_features:
            if feature in result.columns:
                non_null_count = result[feature].notna().sum()
                logger.info(f"{feature}: 非NULL値数 = {non_null_count}/{len(result)}")

        logger.info("✅ FearGreedFeatureCalculator 単体テスト完了")
        return True

    except Exception as e:
        logger.error(f"❌ FearGreedFeatureCalculator テストエラー: {e}")
        return False


async def test_feature_engineering_service_integration():
    """FeatureEngineeringService統合テスト"""
    logger.info("=== FeatureEngineeringService 統合テスト ===")

    try:
        # テストデータ作成
        ohlcv_data = create_sample_ohlcv_data(50)
        fear_greed_data = create_sample_fear_greed_data(50)
        lookback_periods = {"short_ma": 10, "long_ma": 50, "volatility": 20}

        # サービス初期化
        service = FeatureEngineeringService()

        # 特徴量計算実行（Fear & Greed データ含む）
        result = service.calculate_advanced_features(
            ohlcv_data=ohlcv_data,
            fear_greed_data=fear_greed_data,
            lookback_periods=lookback_periods,
        )

        # 結果検証
        logger.info(f"結果のデータ形状: {result.shape}")
        logger.info(f"総特徴量数: {len(result.columns)}")

        # Fear & Greed特徴量の存在確認
        fg_features = [col for col in result.columns if col.startswith("FG_")]
        logger.info(f"Fear & Greed特徴量数: {len(fg_features)}")
        logger.info(f"Fear & Greed特徴量: {fg_features}")

        if len(fg_features) == 0:
            logger.error("❌ Fear & Greed特徴量が生成されていません")
            return False

        # データ品質チェック
        null_counts = result.isnull().sum()
        high_null_features = null_counts[null_counts > len(result) * 0.8]
        if len(high_null_features) > 0:
            logger.warning(f"⚠️ 高いNULL率の特徴量: {high_null_features.to_dict()}")

        logger.info("✅ FeatureEngineeringService 統合テスト完了")
        return True

    except Exception as e:
        logger.error(f"❌ FeatureEngineeringService テストエラー: {e}")
        return False


async def test_with_real_data():
    """実際のデータベースデータを使用したテスト"""
    logger.info("=== 実データテスト ===")

    try:
        with SessionLocal() as db:
            # リポジトリ初期化
            fg_repo = FearGreedIndexRepository(db)
            ext_repo = ExternalMarketRepository(db)
            ohlcv_repo = OHLCVRepository(db)

            # データ存在確認
            fg_count = fg_repo.get_data_count()
            ext_stats = ext_repo.get_data_statistics()
            ext_count = ext_stats.get("count", 0)

            logger.info(f"Fear & Greed データ件数: {fg_count}")
            logger.info(f"外部市場データ件数: {ext_count}")

            if fg_count == 0:
                logger.warning(
                    "⚠️ Fear & Greed データが存在しません。サンプルデータでテストします。"
                )
                return await test_feature_engineering_service_integration()

            # 実データ取得
            fg_data_raw = fg_repo.get_latest_fear_greed_data(limit=100)

            if not fg_data_raw:
                logger.warning("⚠️ Fear & Greed データの取得に失敗しました")
                return False

            # DataFrameに変換
            fg_data = pd.DataFrame(
                [
                    {
                        "value": item.value,
                        "value_classification": item.value_classification,
                        "data_timestamp": item.data_timestamp,
                    }
                    for item in fg_data_raw
                ]
            )

            fg_data.set_index("data_timestamp", inplace=True)

            # OHLCVデータ（サンプル）
            ohlcv_data = create_sample_ohlcv_data(len(fg_data))

            # 特徴量計算
            service = FeatureEngineeringService()
            result = service.calculate_advanced_features(
                ohlcv_data=ohlcv_data,
                fear_greed_data=fg_data,
                lookback_periods={"short_ma": 7, "long_ma": 30},
            )

            logger.info(f"実データテスト結果: {result.shape}")
            fg_features = [col for col in result.columns if col.startswith("FG_")]
            logger.info(f"生成されたFear & Greed特徴量: {len(fg_features)}")

            # 統計情報
            for feature in fg_features[:5]:  # 最初の5つの特徴量
                if feature in result.columns:
                    stats = result[feature].describe()
                    logger.info(
                        f"{feature}: mean={stats['mean']:.3f}, std={stats['std']:.3f}"
                    )

            logger.info("✅ 実データテスト完了")
            return True

    except Exception as e:
        logger.error(f"❌ 実データテストエラー: {e}")
        return False


async def main():
    """メインテスト実行"""
    logger.info("🚀 Fear & Greed Index 特徴量エンジニアリング テスト開始")

    tests = [
        ("FearGreedFeatureCalculator 単体テスト", test_fear_greed_feature_calculator),
        (
            "FeatureEngineeringService 統合テスト",
            test_feature_engineering_service_integration,
        ),
        ("実データテスト", test_with_real_data),
    ]

    results = []
    for test_name, test_func in tests:
        logger.info(f"\n📋 {test_name} 実行中...")
        try:
            result = await test_func()
            results.append((test_name, result))
            if result:
                logger.info(f"✅ {test_name} 成功")
            else:
                logger.error(f"❌ {test_name} 失敗")
        except Exception as e:
            logger.error(f"❌ {test_name} 例外: {e}")
            results.append((test_name, False))

    # 結果サマリー
    logger.info("\n" + "=" * 50)
    logger.info("📊 テスト結果サマリー")
    logger.info("=" * 50)

    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status}: {test_name}")
        if result:
            passed += 1

    logger.info(f"\n🎯 総合結果: {passed}/{len(results)} テスト成功")

    if passed == len(results):
        logger.info("🎉 全てのテストが成功しました！")
        return True
    else:
        logger.error("💥 一部のテストが失敗しました")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
