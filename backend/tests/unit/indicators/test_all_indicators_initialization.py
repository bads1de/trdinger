"""
全指標の初期化テスト

constants.pyで定義されている全58個の指標について、
初期化が正常に動作するかを包括的にテストします。
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

from app.core.services.auto_strategy.factories.indicator_initializer import (
    IndicatorInitializer,
)
from app.core.services.indicators.constants import ALL_INDICATORS, INDICATOR_INFO

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_data(length: int = 200) -> pd.DataFrame:
    """テスト用の価格データを作成（十分な長さ）"""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=length, freq="1h")

    # より現実的な価格データを生成
    price = 100
    prices = []
    highs = []
    lows = []
    volumes = []

    for i in range(length):
        # トレンドとボラティリティを含む価格生成
        trend = 0.01 * np.sin(i / 20)  # 長期トレンド
        noise = np.random.normal(0, 0.5)  # ランダムノイズ
        price += trend + noise

        # OHLC生成
        high = price + abs(np.random.normal(0, 0.3))
        low = price - abs(np.random.normal(0, 0.3))
        volume = 1000 + abs(np.random.normal(0, 200))

        prices.append(price)
        highs.append(high)
        lows.append(low)
        volumes.append(volume)

    df = pd.DataFrame(
        {
            "open": prices,
            "high": highs,
            "low": lows,
            "close": prices,
            "volume": volumes,
        },
        index=dates,
    )

    return df


def get_default_parameters(indicator_type: str) -> Dict:
    """指標タイプに応じたデフォルトパラメータを取得"""
    # 基本的なパラメータマッピング
    param_mapping = {
        # 期間のみの指標
        "SMA": {"period": 20},
        "EMA": {"period": 20},
        "WMA": {"period": 20},
        "HMA": {"period": 20},
        "KAMA": {"period": 20},
        "TEMA": {"period": 20},
        "DEMA": {"period": 20},
        "T3": {"period": 20},
        "ZLEMA": {"period": 20},
        "TRIMA": {"period": 20},
        "VWMA": {"period": 20},
        "MIDPOINT": {"period": 20},
        "RSI": {"period": 14},
        "CCI": {"period": 20},
        "WILLR": {"period": 14},
        "WILLIAMS": {"period": 14},
        "MOM": {"period": 10},
        "MOMENTUM": {"period": 10},
        "ROC": {"period": 10},
        "ADX": {"period": 14},
        "MFI": {"period": 14},
        "CMO": {"period": 14},
        "TRIX": {"period": 14},
        "ULTOSC": {"period": 14},
        "BOP": {"period": 1},
        "APO": {"period": 12},
        "PPO": {"period": 12},
        "DX": {"period": 14},
        "ADXR": {"period": 14},
        "ATR": {"period": 14},
        "NATR": {"period": 14},
        "STDDEV": {"period": 20},
        "EMV": {"period": 14},
        # 複数パラメータの指標
        "MACD": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
        "BB": {"period": 20, "std_dev": 2.0},
        "STOCH": {"k_period": 14, "d_period": 3},
        "STOCHRSI": {"period": 14, "fastk_period": 3, "fastd_period": 3},
        "AROON": {"period": 14},
        "KELTNER": {"period": 20},
        "DONCHIAN": {"period": 20},
        "ADOSC": {"fast_period": 3, "slow_period": 10},
        "MAMA": {"fast_limit": 0.5, "slow_limit": 0.05},
        "PSAR": {"period": 12},
        "ULTOSC": {"period1": 7, "period2": 14, "period3": 28},
        "APO": {"fast_period": 12, "slow_period": 26},
        "PPO": {"fast_period": 12, "slow_period": 26},
        "VWAP": {"period": 20},
        # High/Low/Closeが必要な指標
        "MIDPRICE": {"period": 14},
        "TRANGE": {},
        # ボリューム系指標
        "OBV": {},
        "AD": {},
        "PVT": {},
        "VWAP": {},
        # 価格変換系指標
        "AVGPRICE": {},
        "MEDPRICE": {},
        "TYPPRICE": {},
        "WCLPRICE": {},
        # 未実装指標（代替される予定）
        "STOCHF": {"period": 14},
        "ROCP": {"period": 10},
        "ROCR": {"period": 10},
        "AROONOSC": {"period": 14},
        "PLUS_DI": {"period": 14},
        "MINUS_DI": {"period": 14},
    }

    return param_mapping.get(indicator_type, {"period": 14})


def test_single_indicator(
    indicator_type: str, initializer: IndicatorInitializer, test_data: pd.DataFrame
) -> Tuple[bool, str]:
    """単一指標の初期化テスト"""
    try:
        parameters = get_default_parameters(indicator_type)

        # 指標計算のみテスト
        result, indicator_name = initializer.calculate_indicator_only(
            indicator_type, parameters, test_data
        )

        if result is not None and indicator_name is not None:
            # 結果の検証
            if hasattr(result, "__len__") and len(result) > 0:
                return True, f"成功 - {indicator_name} (データ数: {len(result)})"
            else:
                return False, f"失敗 - 空の結果"
        else:
            return False, f"失敗 - 計算結果がNone"

    except Exception as e:
        return False, f"エラー - {str(e)}"


def test_all_indicators():
    """全指標の初期化テスト"""
    logger.info("🔍 全指標初期化テスト開始")
    logger.info(f"テスト対象: {len(ALL_INDICATORS)}個の指標")

    initializer = IndicatorInitializer()
    test_data = create_test_data(200)  # 十分な長さのデータ

    # サポートされている指標とされていない指標を分類
    supported_indicators = initializer.get_supported_indicators()
    fallback_indicators = initializer.fallback_indicators

    results = {
        "success": [],
        "fallback_success": [],
        "fallback_failed": [],
        "unsupported": [],
        "error": [],
    }

    logger.info(f"直接サポート: {len(supported_indicators)}個")
    logger.info(f"代替サポート: {len(fallback_indicators)}個")

    for indicator_type in ALL_INDICATORS:
        logger.info(f"\n--- {indicator_type} ---")

        success, message = test_single_indicator(indicator_type, initializer, test_data)

        if success:
            if indicator_type in supported_indicators:
                results["success"].append((indicator_type, message))
                logger.info(f"✅ {indicator_type}: {message}")
            else:
                results["fallback_success"].append((indicator_type, message))
                logger.info(f"🔄 {indicator_type}: {message} (代替)")
        else:
            if indicator_type in fallback_indicators:
                results["fallback_failed"].append((indicator_type, message))
                logger.warning(f"⚠️ {indicator_type}: {message} (代替失敗)")
            elif indicator_type not in supported_indicators:
                results["unsupported"].append((indicator_type, message))
                logger.error(f"❌ {indicator_type}: {message} (未サポート)")
            else:
                results["error"].append((indicator_type, message))
                logger.error(f"💥 {indicator_type}: {message}")

    return results


def print_detailed_results(results: Dict):
    """詳細な結果レポートを出力"""
    logger.info("\n" + "=" * 80)
    logger.info("📊 詳細テスト結果レポート")
    logger.info("=" * 80)

    # 成功した指標
    logger.info(f"\n✅ 直接サポート成功 ({len(results['success'])}個):")
    for indicator, message in results["success"]:
        logger.info(f"  {indicator}: {message}")

    # 代替成功した指標
    logger.info(f"\n🔄 代替サポート成功 ({len(results['fallback_success'])}個):")
    for indicator, message in results["fallback_success"]:
        logger.info(f"  {indicator}: {message}")

    # 代替失敗した指標
    if results["fallback_failed"]:
        logger.info(f"\n⚠️ 代替サポート失敗 ({len(results['fallback_failed'])}個):")
        for indicator, message in results["fallback_failed"]:
            logger.info(f"  {indicator}: {message}")

    # 未サポート指標
    if results["unsupported"]:
        logger.info(f"\n❌ 未サポート指標 ({len(results['unsupported'])}個):")
        for indicator, message in results["unsupported"]:
            logger.info(f"  {indicator}: {message}")

    # エラー指標
    if results["error"]:
        logger.info(f"\n💥 エラー指標 ({len(results['error'])}個):")
        for indicator, message in results["error"]:
            logger.info(f"  {indicator}: {message}")

    # サマリー
    total_success = len(results["success"]) + len(results["fallback_success"])
    total_indicators = len(ALL_INDICATORS)
    success_rate = (total_success / total_indicators) * 100

    logger.info(f"\n📈 サマリー:")
    logger.info(f"  総指標数: {total_indicators}")
    logger.info(f"  成功数: {total_success}")
    logger.info(f"  成功率: {success_rate:.1f}%")

    if success_rate >= 90:
        logger.info("🎉 優秀な成功率です！")
    elif success_rate >= 80:
        logger.info("👍 良好な成功率です")
    else:
        logger.warning("⚠️ 成功率の改善が必要です")


def main():
    """メイン実行"""
    try:
        results = test_all_indicators()
        print_detailed_results(results)

        # 成功率の計算
        total_success = len(results["success"]) + len(results["fallback_success"])
        total_indicators = len(ALL_INDICATORS)
        success_rate = (total_success / total_indicators) * 100

        return success_rate >= 80  # 80%以上を成功とする

    except Exception as e:
        logger.error(f"テスト実行エラー: {e}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
