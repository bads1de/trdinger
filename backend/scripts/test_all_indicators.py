#!/usr/bin/env python3
"""
全テクニカル指標の動作確認スクリプト

全ての利用可能なテクニカル指標を実際のデータで計算し、
エラーが発生する指標を特定します。
"""

import sys
import os
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Tuple

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from app.core.services.auto_strategy.factories.indicator_initializer import (
    IndicatorInitializer,
)

# ログ設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_sample_data(length: int = 100) -> pd.DataFrame:
    """
    テスト用のOHLCVデータを生成

    Args:
        length: データの長さ

    Returns:
        OHLCVデータのDataFrame
    """
    np.random.seed(42)  # 再現性のため

    # 基準価格から始めて、ランダムウォークで価格を生成
    base_price = 100.0
    price_changes = np.random.normal(0, 0.02, length)  # 2%の標準偏差
    prices = [base_price]

    for change in price_changes[1:]:
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 1.0))  # 最低価格を1に設定

    # OHLC生成
    data = []
    for i, close in enumerate(prices):
        # 高値・安値の範囲を設定
        volatility = 0.01  # 1%の日中変動
        high = close * (1 + np.random.uniform(0, volatility))
        low = close * (1 - np.random.uniform(0, volatility))

        # 始値は前日終値付近
        if i == 0:
            open_price = close
        else:
            open_price = prices[i - 1] * (1 + np.random.normal(0, 0.005))

        # 出来高は価格変動に連動
        volume = np.random.uniform(1000, 10000) * (1 + abs(price_changes[i]) * 10)

        data.append(
            {
                "open": open_price,
                "high": max(open_price, high, close),
                "low": min(open_price, low, close),
                "close": close,
                "volume": volume,
            }
        )

    df = pd.DataFrame(data)
    df.index = pd.date_range(start="2023-01-01", periods=length, freq="D")

    return df


def get_all_indicators() -> Dict[str, Dict[str, Any]]:
    """
    全ての利用可能な指標とそのデフォルトパラメータを取得

    Returns:
        指標名とパラメータの辞書
    """
    indicators = {
        # トレンド系
        "SMA": {"period": 20},
        "EMA": {"period": 20},
        "WMA": {"period": 20},
        "DEMA": {"period": 20},
        "TEMA": {"period": 20},
        "TRIMA": {"period": 20},
        "KAMA": {"period": 20},
        "T3": {"period": 5, "vfactor": 0.7},
        "HMA": {"period": 20},
        "ZLEMA": {"period": 20},
        "VWMA": {"period": 20},
        "MIDPOINT": {"period": 14},
        "MIDPRICE": {"period": 14},
        # モメンタム系
        "RSI": {"period": 14},
        "STOCH": {"k_period": 14, "d_period": 3},
        "STOCHRSI": {"period": 14, "fastk_period": 3, "fastd_period": 3},
        "CCI": {"period": 14},
        "WILLR": {"period": 14},
        "MOM": {"period": 10},
        "ROC": {"period": 10},
        "ADX": {"period": 14},
        "AROON": {"period": 14},
        "MFI": {"period": 14},
        "ULTOSC": {"period1": 7, "period2": 14, "period3": 28},
        "CMO": {"period": 14},
        "TRIX": {"period": 14},
        "BOP": {"period": 1},
        "APO": {"fast_period": 12, "slow_period": 26},
        "PPO": {"fast_period": 12, "slow_period": 26},
        "AROONOSC": {"period": 14},
        "DX": {"period": 14},
        "ADXR": {"period": 14},
        "PLUS_DI": {"period": 14},
        "MINUS_DI": {"period": 14},
        "ROCP": {"period": 10},
        "ROCR": {"period": 10},
        "STOCHF": {"fastk_period": 5, "fastd_period": 3},
        # ボラティリティ系
        "BB": {"period": 20, "std_dev": 2},
        "ATR": {"period": 14},
        "NATR": {"period": 14},
        "TRANGE": {"period": 1},
        "KELTNER": {"period": 20},
        "STDDEV": {"period": 20},
        "DONCHIAN": {"period": 20},
        # ボリューム系
        "OBV": {"period": 1},
        "AD": {"period": 1},
        "PVT": {"period": 1},
        "EMV": {"period": 14},
        "VWAP": {"period": 20},
        "ADOSC": {"fast_period": 3, "slow_period": 10},
        # 価格変換系
        "AVGPRICE": {"period": 1},
        "MEDPRICE": {"period": 1},
        "TYPPRICE": {"period": 1},
        "WCLPRICE": {"period": 1},
        # 複合指標
        "MACD": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
        "PSAR": {"period": 12},
        "MAMA": {"fast_limit": 0.5, "slow_limit": 0.05},
    }

    return indicators


def test_indicator(
    initializer: IndicatorInitializer,
    indicator_type: str,
    parameters: Dict[str, Any],
    data: pd.DataFrame,
) -> Tuple[bool, str]:
    """
    単一指標のテスト

    Args:
        initializer: 指標初期化器
        indicator_type: 指標タイプ
        parameters: パラメータ
        data: テストデータ

    Returns:
        (成功フラグ, エラーメッセージ)
    """
    try:
        result, indicator_name = initializer.calculate_indicator_only(
            indicator_type, parameters, data
        )

        if result is None or indicator_name is None:
            return False, "計算結果がNone"

        if hasattr(result, "__len__") and len(result) == 0:
            return False, "計算結果が空"

        # NaNチェック
        if hasattr(result, "isna"):
            # DataFrameの場合は全ての列をチェック
            if isinstance(result, pd.DataFrame):
                if result.isna().all().all():
                    return False, "全ての値がNaN"
            else:
                if result.isna().all():
                    return False, "全ての値がNaN"

        return True, "成功"

    except Exception as e:
        return False, str(e)


def main():
    """メイン実行関数"""
    logger.info("=== 全テクニカル指標動作確認開始 ===")

    # テストデータ生成
    logger.info("テストデータを生成中...")
    test_data = create_sample_data(100)
    logger.info(f"テストデータ生成完了: {len(test_data)}行")

    # 指標初期化器を作成
    initializer = IndicatorInitializer()

    # 全指標を取得
    all_indicators = get_all_indicators()
    logger.info(f"テスト対象指標数: {len(all_indicators)}")

    # 結果記録用
    success_indicators = []
    failed_indicators = []

    # 各指標をテスト
    for indicator_type, parameters in all_indicators.items():
        logger.info(f"テスト中: {indicator_type}")

        success, message = test_indicator(
            initializer, indicator_type, parameters, test_data
        )

        if success:
            success_indicators.append(indicator_type)
            logger.info(f"✓ {indicator_type}: {message}")
        else:
            failed_indicators.append((indicator_type, message))
            logger.error(f"✗ {indicator_type}: {message}")

    # 結果サマリー
    logger.info("\n=== テスト結果サマリー ===")
    logger.info(f"成功: {len(success_indicators)}/{len(all_indicators)}")
    logger.info(f"失敗: {len(failed_indicators)}/{len(all_indicators)}")

    if success_indicators:
        logger.info("\n成功した指標:")
        for indicator in success_indicators:
            logger.info(f"  ✓ {indicator}")

    if failed_indicators:
        logger.info("\n失敗した指標:")
        for indicator, error in failed_indicators:
            logger.info(f"  ✗ {indicator}: {error}")

    # 詳細レポート
    logger.info(f"\n=== 詳細レポート ===")
    logger.info(f"テストデータ形状: {test_data.shape}")
    logger.info(f"テストデータ列: {list(test_data.columns)}")
    logger.info(f"テストデータ期間: {test_data.index[0]} - {test_data.index[-1]}")

    return len(failed_indicators) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
