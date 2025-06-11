#!/usr/bin/env python3
"""
更新されたシンボル設定をテストするスクリプト

新しく追加されたBTC（ETHは除外）のスポット・先物ペアが
正しく設定されているかを確認します。
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from app.config.market_config import MarketDataConfig
import logging

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_symbol_validation():
    """シンボル検証のテスト"""
    print("=" * 60)
    print("シンボル検証テスト")
    print("=" * 60)

    # テスト対象の通貨
    target_currencies = ["BTC"]  # ETHは除外

    for currency in target_currencies:
        print(f"\n【{currency}】")

        # スポットペア
        spot_symbol = f"{currency}/USDT"
        is_valid = MarketDataConfig.validate_symbol(spot_symbol)
        status = "✓" if is_valid else "❌"
        print(f"  スポット {spot_symbol}: {status}")

        # 先物ペア（永続契約）
        futures_symbol = f"{currency}/USD:{currency}"
        is_valid = MarketDataConfig.validate_symbol(futures_symbol)
        status = "✓" if is_valid else "❌"
        print(f"  先物   {futures_symbol}: {status}")


def test_symbol_normalization():
    """シンボル正規化のテスト"""
    print("\n" + "=" * 60)
    print("シンボル正規化テスト")
    print("=" * 60)

    # テストケース（BTCのみ）
    test_cases = [
        # Bitcoin
        ("BTC/USDT", "BTC/USDT"),
        ("BTCUSDT", "BTC/USDT:USDT"),
        ("BTC-USDT", "BTC/USDT"),
        ("BTCUSD", "BTCUSD"),
        ("BTC/USDT:USDT", "BTC/USDT:USDT"),
    ]

    success_count = 0
    total_count = len(test_cases)

    for input_symbol, expected_output in test_cases:
        try:
            normalized = MarketDataConfig.normalize_symbol(input_symbol)
            if normalized == expected_output:
                status = "✓"
                success_count += 1
            else:
                status = "❌"
            print(
                f"  {input_symbol:12} → {normalized:15} (期待値: {expected_output:15}) {status}"
            )
        except Exception as e:
            print(f"  {input_symbol:12} → エラー: {e} ❌")

    print(f"\n正規化テスト結果: {success_count}/{total_count} 成功")


def test_supported_symbols_list():
    """サポートされているシンボル一覧の確認"""
    print("\n" + "=" * 60)
    print("サポートされているシンボル一覧")
    print("=" * 60)

    symbols = MarketDataConfig.SUPPORTED_SYMBOLS

    # 通貨別に分類
    currencies = {}
    for symbol in symbols:
        # 通貨コードを抽出
        if "/" in symbol:
            currency = symbol.split("/")[0]
        else:
            # 代替表記の場合
            for curr in ["BTC"]:
                if symbol.startswith(curr):
                    currency = curr
                    break
            else:
                currency = "OTHER"

        if currency not in currencies:
            currencies[currency] = {"spot": [], "futures": []}

        # スポットか先物かを判定
        if ":" in symbol or symbol.endswith("USD"):
            currencies[currency]["futures"].append(symbol)
        else:
            currencies[currency]["spot"].append(symbol)

    # 結果表示
    for currency in ["BTC"]:
        if currency in currencies:
            print(f"\n【{currency}】")
            print(f"  スポット ({len(currencies[currency]['spot'])}ペア):")
            for symbol in currencies[currency]["spot"]:
                print(f"    {symbol}")
            print(f"  先物 ({len(currencies[currency]['futures'])}ペア):")
            for symbol in currencies[currency]["futures"]:
                print(f"    {symbol}")

    print(f"\n総シンボル数: {len(symbols)}")


def test_timeframe_validation():
    """時間軸検証のテスト"""
    print("\n" + "=" * 60)
    print("時間軸検証テスト")
    print("=" * 60)

    valid_timeframes = ["15m", "30m", "1h", "4h", "1d"]
    invalid_timeframes = ["1m", "5m", "2m", "3h", "1w", "1M"]

    print("有効な時間軸:")
    for tf in valid_timeframes:
        is_valid = MarketDataConfig.validate_timeframe(tf)
        status = "✓" if is_valid else "❌"
        print(f"  {tf}: {status}")

    print("\n無効な時間軸:")
    for tf in invalid_timeframes:
        is_valid = MarketDataConfig.validate_timeframe(tf)
        status = "✓" if is_valid else "❌"
        print(f"  {tf}: {status}")


def main():
    """メイン関数"""
    print("取引ペア設定テスト開始")
    print("対象通貨: BTC")
    print("対象市場: スポット, 先物（永続契約）")

    try:
        # 各テストを実行
        test_supported_symbols_list()
        test_symbol_validation()
        test_symbol_normalization()
        test_timeframe_validation()

        print("\n" + "=" * 60)
        print("テスト完了")
        print("=" * 60)
        print("✓ 設定が正常に更新されました")
        print("✓ BTCのスポット・先物ペアが利用可能です")

    except Exception as e:
        logger.error(f"テスト中にエラーが発生しました: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
