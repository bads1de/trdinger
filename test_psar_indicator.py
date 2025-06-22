#!/usr/bin/env python3
"""
PSAR指標のテストスクリプト

修正後のPSAR指標が正しく動作するかをテストします。
"""

import sys
import os
import logging
import pandas as pd
import numpy as np

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), "backend"))

from app.core.services.indicators.adapters.volatility_adapter import VolatilityAdapter
from app.core.services.auto_strategy.factories.indicator_initializer import (
    IndicatorInitializer,
)

# ログ設定
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_test_data():
    """テスト用のOHLCVデータを作成"""
    dates = pd.date_range("2024-01-01", periods=100, freq="1H")

    # 簡単な価格データを生成
    np.random.seed(42)
    base_price = 50000
    price_changes = np.random.normal(0, 100, 100)
    close_prices = base_price + np.cumsum(price_changes)

    # OHLC データを生成
    high_prices = close_prices + np.random.uniform(0, 200, 100)
    low_prices = close_prices - np.random.uniform(0, 200, 100)
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = base_price

    volume = np.random.uniform(1000, 10000, 100)

    data = pd.DataFrame(
        {
            "timestamp": dates,
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "volume": volume,
        }
    )

    return data


def test_psar_adapter():
    """VolatilityAdapterのPSARメソッドをテスト"""
    print("=== PSAR Adapter テスト ===")

    try:
        data = create_test_data()
        high_series = pd.Series(data["high"].values, index=data.index)
        low_series = pd.Series(data["low"].values, index=data.index)

        # PSARを計算
        result = VolatilityAdapter.psar(
            high_series, low_series, acceleration=0.02, maximum=0.2
        )

        print(f"PSAR計算成功: {len(result)}個のデータポイント")
        print(f"最初の5つの値: {result.head().tolist()}")
        print(f"最後の5つの値: {result.tail().tolist()}")

        return True

    except Exception as e:
        print(f"PSAR Adapter テストエラー: {e}")
        logger.error(f"PSAR Adapter テストエラー: {e}", exc_info=True)
        return False


def test_psar_initializer():
    """IndicatorInitializerのPSAR処理をテスト"""
    print("\n=== PSAR Initializer テスト ===")

    try:
        data = create_test_data()
        initializer = IndicatorInitializer()

        # PSAR指標を初期化
        parameters = {"period": 12}
        result, indicator_name = initializer.calculate_indicator_only(
            "PSAR", parameters, data
        )

        print(f"PSAR初期化成功: {indicator_name}")
        print(f"結果データ数: {len(result)}")
        print(f"最初の5つの値: {result.head().tolist()}")
        print(f"最後の5つの値: {result.tail().tolist()}")

        return True

    except Exception as e:
        print(f"PSAR Initializer テストエラー: {e}")
        logger.error(f"PSAR Initializer テストエラー: {e}", exc_info=True)
        return False


def test_psar_strategy_generation():
    """PSAR指標を使った戦略生成をテスト"""
    print("\n=== PSAR戦略生成テスト ===")

    try:
        from app.core.services.auto_strategy.generators.random_gene_generator import (
            RandomGeneGenerator,
        )
        from app.core.services.auto_strategy.models.gene_encoding import GeneEncoder
        from app.core.services.auto_strategy.models.strategy_gene import StrategyGene

        # PSAR指標を含む戦略を生成
        generator = RandomGeneGenerator()

        # 複数回試行してPSAR指標が生成されるかをテスト
        psar_found = False
        for i in range(20):
            gene = generator.generate_random_gene()

            for indicator in gene.indicators:
                if indicator.type == "PSAR":
                    psar_found = True
                    print(f"PSAR指標発見: {indicator.type} - {indicator.parameters}")

                    # エンコード・デコードテスト
                    encoder = GeneEncoder()
                    encoded = encoder.encode_strategy_gene_to_list(gene)
                    decoded = encoder.decode_list_to_strategy_gene(
                        encoded, StrategyGene
                    )

                    print(f"エンコード・デコード成功")
                    print(f"デコード後の指標数: {len(decoded.indicators)}")

                    for dec_indicator in decoded.indicators:
                        if dec_indicator.type == "PSAR":
                            print(
                                f"デコード後PSAR: {dec_indicator.type} - {dec_indicator.parameters}"
                            )

                    break

            if psar_found:
                break

        if psar_found:
            print("PSAR戦略生成テスト成功")
            return True
        else:
            print("PSAR指標が生成されませんでした（20回試行）")
            return False

    except Exception as e:
        print(f"PSAR戦略生成テストエラー: {e}")
        logger.error(f"PSAR戦略生成テストエラー: {e}", exc_info=True)
        return False


def main():
    """メイン実行関数"""
    print("PSAR指標テスト開始")
    print("=" * 50)

    results = []

    # 1. PSAR Adapter テスト
    results.append(test_psar_adapter())

    # 2. PSAR Initializer テスト
    results.append(test_psar_initializer())

    # 3. PSAR戦略生成テスト
    results.append(test_psar_strategy_generation())

    # 結果まとめ
    print("\n" + "=" * 50)
    print("テスト結果まとめ")
    print("=" * 50)

    test_names = ["PSAR Adapter", "PSAR Initializer", "PSAR戦略生成"]

    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✓ 成功" if result else "✗ 失敗"
        print(f"{i+1}. {name}: {status}")

    all_passed = all(results)
    print(f"\n総合結果: {'✓ 全テスト成功' if all_passed else '✗ 一部テスト失敗'}")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
