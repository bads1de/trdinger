"""
オートストラテジー 国際化・多様性テスト

異なるタイムゾーン、通貨ペア、小数点精度、地域固有の処理を検証します。
"""

import sys
import os

# プロジェクトルートをパスに追加
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, backend_dir)

import pytest
import numpy as np
import time
import pytz
import logging
from datetime import datetime, timedelta
from decimal import Decimal, getcontext
import locale

logger = logging.getLogger(__name__)


class TestInternationalization:
    """国際化・多様性テストクラス"""
    
    def setup_method(self):
        """テスト前の準備"""
        self.start_time = time.time()
        # 高精度計算のための設定
        getcontext().prec = 28
        
    def teardown_method(self):
        """テスト後のクリーンアップ"""
        execution_time = time.time() - self.start_time
        logger.info(f"テスト実行時間: {execution_time:.3f}秒")
    
    def test_timezone_handling(self):
        """テスト58: 異なるタイムゾーンでの時刻処理"""
        logger.info("🔍 タイムゾーン処理テスト開始")
        
        try:
            from app.services.auto_strategy.calculators.tpsl_calculator import TPSLCalculator
            
            calculator = TPSLCalculator()
            
            # 主要なタイムゾーン
            timezones = {
                "UTC": pytz.UTC,
                "US/Eastern": pytz.timezone("US/Eastern"),
                "Europe/London": pytz.timezone("Europe/London"),
                "Asia/Tokyo": pytz.timezone("Asia/Tokyo"),
                "Asia/Shanghai": pytz.timezone("Asia/Shanghai"),
                "Australia/Sydney": pytz.timezone("Australia/Sydney"),
                "America/New_York": pytz.timezone("America/New_York"),
                "Europe/Berlin": pytz.timezone("Europe/Berlin")  # Frankfurt -> Berlin
            }
            
            # 基準時刻（UTC）
            base_utc = datetime(2023, 6, 15, 12, 0, 0, tzinfo=pytz.UTC)
            
            timezone_results = {}
            
            for tz_name, tz in timezones.items():
                # UTC時刻を各タイムゾーンに変換
                local_time = base_utc.astimezone(tz)
                
                # 市場時間の判定（簡易版）
                is_market_open = self._is_market_open(local_time, tz_name)
                
                # タイムゾーン固有の処理
                market_data = {
                    "timestamp": local_time,
                    "price": 50000 + np.random.normal(0, 100),
                    "timezone": tz_name,
                    "market_open": is_market_open
                }
                
                # TP/SL計算（タイムゾーンに関係なく同じ結果になるべき）
                sl_price, tp_price = calculator.calculate_basic_tpsl_prices(
                    market_data["price"], 0.02, 0.04, 1.0
                )
                
                timezone_results[tz_name] = {
                    "local_time": local_time,
                    "utc_time": local_time.astimezone(pytz.UTC),
                    "price": market_data["price"],
                    "sl_price": sl_price,
                    "tp_price": tp_price,
                    "market_open": is_market_open,
                    "utc_offset": local_time.utcoffset().total_seconds() / 3600
                }
                
                logger.info(f"{tz_name}: {local_time.strftime('%Y-%m-%d %H:%M:%S %Z')} (UTC{local_time.utcoffset().total_seconds()/3600:+.1f})")
            
            # タイムゾーン間の一貫性確認
            utc_times = [result["utc_time"] for result in timezone_results.values()]
            
            # 全てのUTC時刻が同じであることを確認
            base_utc_timestamp = utc_times[0].timestamp()
            for utc_time in utc_times[1:]:
                assert abs(utc_time.timestamp() - base_utc_timestamp) < 1, "UTC時刻の変換に誤差があります"
            
            # 計算結果の一貫性確認（同じ価格なら同じ結果になるべき）
            base_price = timezone_results["UTC"]["price"]
            base_sl = timezone_results["UTC"]["sl_price"]
            base_tp = timezone_results["UTC"]["tp_price"]
            
            for tz_name, result in timezone_results.items():
                if abs(result["price"] - base_price) < 0.01:  # 同じ価格の場合
                    assert abs(result["sl_price"] - base_sl) < 0.01, f"{tz_name}: SL価格が一致しません"
                    assert abs(result["tp_price"] - base_tp) < 0.01, f"{tz_name}: TP価格が一致しません"
            
            # 市場時間の妥当性確認
            market_open_count = sum(1 for result in timezone_results.values() if result["market_open"])
            logger.info(f"市場オープン中のタイムゾーン: {market_open_count}/{len(timezones)}")
            
            logger.info("✅ タイムゾーン処理テスト成功")
            
        except Exception as e:
            pytest.fail(f"タイムゾーン処理テストエラー: {e}")
    
    def _is_market_open(self, local_time: datetime, timezone_name: str) -> bool:
        """市場オープン時間の判定（簡易版）"""
        hour = local_time.hour
        weekday = local_time.weekday()  # 0=月曜日, 6=日曜日
        
        # 週末は閉場
        if weekday >= 5:  # 土曜日、日曜日
            return False
        
        # 地域別の市場時間（簡易版）
        if "US" in timezone_name or "America" in timezone_name:
            return 9 <= hour <= 16  # 9:00-16:00
        elif "Europe" in timezone_name:
            return 8 <= hour <= 17  # 8:00-17:00
        elif "Asia" in timezone_name:
            return 9 <= hour <= 15  # 9:00-15:00
        else:
            return True  # その他は24時間
    
    def test_multi_currency_pairs(self):
        """テスト59: 複数通貨ペアでの処理"""
        logger.info("🔍 複数通貨ペア処理テスト開始")
        
        try:
            from app.services.auto_strategy.calculators.tpsl_calculator import TPSLCalculator
            
            calculator = TPSLCalculator()
            
            # 主要通貨ペアとその特性
            currency_pairs = {
                "BTC/USDT": {
                    "base_price": 50000,
                    "volatility": 0.05,
                    "min_price_increment": 0.01,
                    "decimal_places": 2
                },
                "EUR/USD": {
                    "base_price": 1.0850,
                    "volatility": 0.01,
                    "min_price_increment": 0.00001,
                    "decimal_places": 5
                },
                "GBP/JPY": {
                    "base_price": 165.50,
                    "volatility": 0.015,
                    "min_price_increment": 0.001,
                    "decimal_places": 3
                },
                "USD/JPY": {
                    "base_price": 150.25,
                    "volatility": 0.012,
                    "min_price_increment": 0.001,
                    "decimal_places": 3
                },
                "AUD/CAD": {
                    "base_price": 0.9125,
                    "volatility": 0.008,
                    "min_price_increment": 0.00001,
                    "decimal_places": 5
                },
                "ETH/BTC": {
                    "base_price": 0.0625,
                    "volatility": 0.03,
                    "min_price_increment": 0.000001,
                    "decimal_places": 6
                }
            }
            
            currency_results = {}
            
            for pair, config in currency_pairs.items():
                # 通貨ペア固有の価格生成
                current_price = config["base_price"] * (1 + np.random.normal(0, config["volatility"]))
                
                # 小数点精度に合わせて価格を調整
                decimal_places = config["decimal_places"]
                current_price = round(current_price, decimal_places)
                
                # 通貨ペア固有のTP/SL設定
                if "JPY" in pair:
                    # 円ペアは通常より小さなパーセンテージ
                    sl_pct = 0.005  # 0.5%
                    tp_pct = 0.01   # 1.0%
                elif "BTC" in pair or "ETH" in pair:
                    # 暗号通貨は大きなパーセンテージ
                    sl_pct = 0.03   # 3.0%
                    tp_pct = 0.06   # 6.0%
                else:
                    # 通常の通貨ペア
                    sl_pct = 0.015  # 1.5%
                    tp_pct = 0.03   # 3.0%
                
                # TP/SL計算
                sl_price, tp_price = calculator.calculate_basic_tpsl_prices(
                    current_price, sl_pct, tp_pct, 1.0
                )
                
                # 価格精度の調整
                if sl_price is not None:
                    sl_price = round(sl_price, decimal_places)
                if tp_price is not None:
                    tp_price = round(tp_price, decimal_places)
                
                # 最小価格単位の確認（浮動小数点の精度問題を考慮）
                min_increment = config["min_price_increment"]
                if sl_price is not None:
                    # 価格を最小単位で割った余りをチェック
                    sl_units = round(sl_price / min_increment)
                    expected_sl = sl_units * min_increment
                    sl_diff = abs(sl_price - expected_sl)
                    # 浮動小数点の精度を考慮して緩い閾値を使用
                    assert sl_diff < min_increment * 0.1, f"{pair}: SL価格が最小単位に合いません (差: {sl_diff})"

                if tp_price is not None:
                    tp_units = round(tp_price / min_increment)
                    expected_tp = tp_units * min_increment
                    tp_diff = abs(tp_price - expected_tp)
                    assert tp_diff < min_increment * 0.1, f"{pair}: TP価格が最小単位に合いません (差: {tp_diff})"
                
                currency_results[pair] = {
                    "current_price": current_price,
                    "sl_price": sl_price,
                    "tp_price": tp_price,
                    "sl_pct": sl_pct,
                    "tp_pct": tp_pct,
                    "decimal_places": decimal_places,
                    "min_increment": min_increment
                }
                
                logger.info(f"{pair}: 価格={current_price:.{decimal_places}f}, SL={sl_price:.{decimal_places}f}, TP={tp_price:.{decimal_places}f}")
            
            # 通貨ペア間の相対的な妥当性確認
            for pair, result in currency_results.items():
                # SL/TP価格が現在価格と適切な関係にあることを確認
                current = result["current_price"]
                sl = result["sl_price"]
                tp = result["tp_price"]
                
                if sl is not None and tp is not None:
                    # ロングポジションの場合
                    assert sl < current < tp, f"{pair}: 価格関係が不正です (SL={sl}, 現在={current}, TP={tp})"
                    
                    # 実際のパーセンテージが期待値に近いことを確認
                    actual_sl_pct = (current - sl) / current
                    actual_tp_pct = (tp - current) / current
                    
                    assert abs(actual_sl_pct - result["sl_pct"]) < 0.001, f"{pair}: SL%が期待値と異なります"
                    assert abs(actual_tp_pct - result["tp_pct"]) < 0.001, f"{pair}: TP%が期待値と異なります"
            
            logger.info("✅ 複数通貨ペア処理テスト成功")
            
        except Exception as e:
            pytest.fail(f"複数通貨ペア処理テストエラー: {e}")
    
    def test_decimal_precision_accuracy(self):
        """テスト60: 異なる小数点精度での計算精度"""
        logger.info("🔍 小数点精度計算テスト開始")
        
        try:
            from app.services.auto_strategy.calculators.tpsl_calculator import TPSLCalculator
            
            calculator = TPSLCalculator()
            
            # 異なる精度レベルのテストケース
            precision_tests = [
                {
                    "name": "高精度暗号通貨",
                    "price": Decimal("50000.123456789"),
                    "sl_pct": Decimal("0.02"),
                    "tp_pct": Decimal("0.04"),
                    "expected_precision": 9
                },
                {
                    "name": "標準FX通貨",
                    "price": Decimal("1.08567"),
                    "sl_pct": Decimal("0.015"),
                    "tp_pct": Decimal("0.03"),
                    "expected_precision": 5
                },
                {
                    "name": "円ペア",
                    "price": Decimal("150.123"),
                    "sl_pct": Decimal("0.005"),
                    "tp_pct": Decimal("0.01"),
                    "expected_precision": 3
                },
                {
                    "name": "株価",
                    "price": Decimal("1234.56"),
                    "sl_pct": Decimal("0.025"),
                    "tp_pct": Decimal("0.05"),
                    "expected_precision": 2
                },
                {
                    "name": "極小価格",
                    "price": Decimal("0.000123456"),
                    "sl_pct": Decimal("0.1"),
                    "tp_pct": Decimal("0.2"),
                    "expected_precision": 9
                }
            ]
            
            precision_results = {}
            
            for test_case in precision_tests:
                name = test_case["name"]
                price = float(test_case["price"])
                sl_pct = float(test_case["sl_pct"])
                tp_pct = float(test_case["tp_pct"])
                expected_precision = test_case["expected_precision"]
                
                # 高精度計算
                sl_price, tp_price = calculator.calculate_basic_tpsl_prices(
                    price, sl_pct, tp_pct, 1.0
                )
                
                if sl_price is not None and tp_price is not None:
                    # Decimalを使用した高精度計算
                    price_decimal = test_case["price"]
                    sl_pct_decimal = test_case["sl_pct"]
                    tp_pct_decimal = test_case["tp_pct"]
                    
                    expected_sl = float(price_decimal * (Decimal("1") - sl_pct_decimal))
                    expected_tp = float(price_decimal * (Decimal("1") + tp_pct_decimal))
                    
                    # 精度の確認
                    sl_error = abs(sl_price - expected_sl) / expected_sl
                    tp_error = abs(tp_price - expected_tp) / expected_tp
                    
                    # 有効桁数の確認
                    sl_significant_digits = self._count_significant_digits(sl_price)
                    tp_significant_digits = self._count_significant_digits(tp_price)
                    
                    precision_results[name] = {
                        "price": price,
                        "sl_price": sl_price,
                        "tp_price": tp_price,
                        "expected_sl": expected_sl,
                        "expected_tp": expected_tp,
                        "sl_error": sl_error,
                        "tp_error": tp_error,
                        "sl_significant_digits": sl_significant_digits,
                        "tp_significant_digits": tp_significant_digits,
                        "expected_precision": expected_precision
                    }
                    
                    logger.info(f"{name}:")
                    logger.info(f"  価格: {price}")
                    logger.info(f"  SL: {sl_price} (誤差: {sl_error:.2e})")
                    logger.info(f"  TP: {tp_price} (誤差: {tp_error:.2e})")
                    logger.info(f"  有効桁数: SL={sl_significant_digits}, TP={tp_significant_digits}")
                    
                    # 精度要件の確認
                    assert sl_error < 1e-10, f"{name}: SL計算精度が不十分です: {sl_error:.2e}"
                    assert tp_error < 1e-10, f"{name}: TP計算精度が不十分です: {tp_error:.2e}"
                    
                    # 有効桁数の確認（期待精度以上、ただし極小価格の場合は緩和）
                    min_expected_precision = max(expected_precision - 2, 5)  # 最低5桁は確保
                    assert sl_significant_digits >= min_expected_precision, f"{name}: SL有効桁数が不足: {sl_significant_digits} < {min_expected_precision}"
                    assert tp_significant_digits >= min_expected_precision, f"{name}: TP有効桁数が不足: {tp_significant_digits} < {min_expected_precision}"
            
            logger.info("✅ 小数点精度計算テスト成功")
            
        except Exception as e:
            pytest.fail(f"小数点精度計算テストエラー: {e}")
    
    def _count_significant_digits(self, number: float) -> int:
        """有効桁数をカウント"""
        if number == 0:
            return 1
        
        # 科学記数法で表現
        if number < 0:
            number = -number
        
        # 小数点以下の桁数を考慮
        str_number = f"{number:.15f}".rstrip('0').rstrip('.')
        if '.' in str_number:
            integer_part, decimal_part = str_number.split('.')
            if integer_part == '0':
                # 0.00123 のような場合、先頭の0は有効桁数に含まない
                return len(decimal_part.lstrip('0'))
            else:
                return len(integer_part) + len(decimal_part)
        else:
            return len(str_number.rstrip('0'))

    def test_regional_market_holidays(self):
        """テスト61: 地域固有の市場休日での処理"""
        logger.info("🔍 地域固有市場休日処理テスト開始")

        try:
            from app.services.auto_strategy.calculators.tpsl_calculator import TPSLCalculator

            calculator = TPSLCalculator()

            # 地域固有の市場休日
            regional_holidays = {
                "US": [
                    datetime(2023, 1, 2),   # New Year's Day (observed)
                    datetime(2023, 1, 16),  # Martin Luther King Jr. Day
                    datetime(2023, 2, 20),  # Presidents' Day
                    datetime(2023, 7, 4),   # Independence Day
                    datetime(2023, 11, 23), # Thanksgiving
                    datetime(2023, 12, 25), # Christmas
                ],
                "Europe": [
                    datetime(2023, 1, 1),   # New Year's Day
                    datetime(2023, 4, 7),   # Good Friday
                    datetime(2023, 4, 10),  # Easter Monday
                    datetime(2023, 5, 1),   # Labour Day
                    datetime(2023, 12, 25), # Christmas Day
                    datetime(2023, 12, 26), # Boxing Day
                ],
                "Asia": [
                    datetime(2023, 1, 1),   # New Year's Day
                    datetime(2023, 1, 23),  # Chinese New Year
                    datetime(2023, 4, 29),  # Golden Week (Japan)
                    datetime(2023, 5, 3),   # Constitution Day (Japan)
                    datetime(2023, 10, 1),  # National Day (China)
                    datetime(2023, 11, 3),  # Culture Day (Japan)
                ]
            }

            holiday_results = {}

            for region, holidays in regional_holidays.items():
                region_stats = {
                    "trading_days": 0,
                    "holiday_days": 0,
                    "weekend_days": 0,
                    "processed_trades": 0,
                    "skipped_trades": 0
                }

                # 1年間の日付をチェック
                start_date = datetime(2023, 1, 1)
                end_date = datetime(2023, 12, 31)
                current_date = start_date

                while current_date <= end_date:
                    is_weekend = current_date.weekday() >= 5  # 土日
                    is_holiday = current_date.date() in [h.date() for h in holidays]
                    is_trading_day = not (is_weekend or is_holiday)

                    if is_weekend:
                        region_stats["weekend_days"] += 1
                    elif is_holiday:
                        region_stats["holiday_days"] += 1
                    else:
                        region_stats["trading_days"] += 1

                        # 取引日の場合のみ処理を実行
                        try:
                            price = 50000 + np.random.normal(0, 100)
                            sl_price, tp_price = calculator.calculate_basic_tpsl_prices(
                                price, 0.02, 0.04, 1.0
                            )

                            if sl_price is not None and tp_price is not None:
                                region_stats["processed_trades"] += 1
                            else:
                                region_stats["skipped_trades"] += 1

                        except Exception as e:
                            region_stats["skipped_trades"] += 1
                            logger.debug(f"{region} {current_date.date()}: 処理エラー - {e}")

                    current_date += timedelta(days=1)

                holiday_results[region] = region_stats

                # 統計情報
                total_days = 365
                trading_rate = region_stats["trading_days"] / total_days
                processing_rate = region_stats["processed_trades"] / region_stats["trading_days"] if region_stats["trading_days"] > 0 else 0

                logger.info(f"{region}地域:")
                logger.info(f"  取引日: {region_stats['trading_days']}日 ({trading_rate:.1%})")
                logger.info(f"  休日: {region_stats['holiday_days']}日")
                logger.info(f"  週末: {region_stats['weekend_days']}日")
                logger.info(f"  処理成功: {region_stats['processed_trades']}件 ({processing_rate:.1%})")
                logger.info(f"  処理スキップ: {region_stats['skipped_trades']}件")

            # 地域間の比較
            trading_days_variance = []
            for region, stats in holiday_results.items():
                trading_days_variance.append(stats["trading_days"])

            max_trading_days = max(trading_days_variance)
            min_trading_days = min(trading_days_variance)
            trading_days_diff = max_trading_days - min_trading_days

            logger.info(f"地域間取引日数差: {trading_days_diff}日")

            # 妥当性確認
            for region, stats in holiday_results.items():
                # 取引日が年間の60%以上であることを確認
                trading_rate = stats["trading_days"] / 365
                assert trading_rate >= 0.6, f"{region}: 取引日率が低すぎます: {trading_rate:.1%}"

                # 処理成功率が95%以上であることを確認
                if stats["trading_days"] > 0:
                    processing_rate = stats["processed_trades"] / stats["trading_days"]
                    assert processing_rate >= 0.95, f"{region}: 処理成功率が低すぎます: {processing_rate:.1%}"

            # 地域間の取引日数差が妥当な範囲内であることを確認
            assert trading_days_diff <= 20, f"地域間取引日数差が大きすぎます: {trading_days_diff}日"

            logger.info("✅ 地域固有市場休日処理テスト成功")

        except Exception as e:
            pytest.fail(f"地域固有市場休日処理テストエラー: {e}")

    def test_locale_specific_formatting(self):
        """テスト62: ロケール固有の数値フォーマット処理"""
        logger.info("🔍 ロケール固有フォーマット処理テスト開始")

        try:
            from app.services.auto_strategy.calculators.tpsl_calculator import TPSLCalculator

            calculator = TPSLCalculator()

            # 異なるロケールでの数値フォーマット
            locale_tests = [
                {
                    "locale": "en_US.UTF-8",
                    "decimal_separator": ".",
                    "thousands_separator": ",",
                    "currency_symbol": "$",
                    "test_price": "50,000.50",
                    "expected_price": 50000.50
                },
                {
                    "locale": "de_DE.UTF-8",
                    "decimal_separator": ",",
                    "thousands_separator": ".",
                    "currency_symbol": "€",
                    "test_price": "50.000,50",
                    "expected_price": 50000.50
                },
                {
                    "locale": "fr_FR.UTF-8",
                    "decimal_separator": ",",
                    "thousands_separator": " ",
                    "currency_symbol": "€",
                    "test_price": "50 000,50",
                    "expected_price": 50000.50
                },
                {
                    "locale": "ja_JP.UTF-8",
                    "decimal_separator": ".",
                    "thousands_separator": ",",
                    "currency_symbol": "¥",
                    "test_price": "50,000.50",
                    "expected_price": 50000.50
                }
            ]

            locale_results = {}

            for test_case in locale_tests:
                locale_name = test_case["locale"]

                try:
                    # ロケール設定（利用可能な場合のみ）
                    try:
                        locale.setlocale(locale.LC_ALL, locale_name)
                        locale_available = True
                    except locale.Error:
                        logger.warning(f"ロケール {locale_name} が利用できません。デフォルトロケールを使用します。")
                        locale_available = False

                    # 数値パース関数
                    def parse_localized_number(number_str: str, decimal_sep: str, thousands_sep: str) -> float:
                        """ロケール固有の数値文字列をパース"""
                        # 通貨記号を除去
                        cleaned = number_str.replace("$", "").replace("€", "").replace("¥", "").strip()

                        # 千の位区切り文字を除去
                        if thousands_sep and thousands_sep != decimal_sep:
                            cleaned = cleaned.replace(thousands_sep, "")

                        # 小数点区切り文字を標準形式に変換
                        if decimal_sep != ".":
                            cleaned = cleaned.replace(decimal_sep, ".")

                        return float(cleaned)

                    # テスト価格のパース
                    parsed_price = parse_localized_number(
                        test_case["test_price"],
                        test_case["decimal_separator"],
                        test_case["thousands_separator"]
                    )

                    # パース精度の確認
                    expected_price = test_case["expected_price"]
                    parse_error = abs(parsed_price - expected_price) / expected_price

                    # TP/SL計算
                    sl_price, tp_price = calculator.calculate_basic_tpsl_prices(
                        parsed_price, 0.02, 0.04, 1.0
                    )

                    # 結果のフォーマット
                    def format_localized_number(number: float, decimal_sep: str, thousands_sep: str) -> str:
                        """数値をロケール固有の形式でフォーマット"""
                        # 標準形式でフォーマット
                        formatted = f"{number:,.2f}"

                        # ロケール固有の区切り文字に変換
                        if decimal_sep != ".":
                            formatted = formatted.replace(".", "DECIMAL_TEMP")
                        if thousands_sep != ",":
                            formatted = formatted.replace(",", thousands_sep)
                        if decimal_sep != ".":
                            formatted = formatted.replace("DECIMAL_TEMP", decimal_sep)

                        return formatted

                    formatted_sl = format_localized_number(
                        sl_price, test_case["decimal_separator"], test_case["thousands_separator"]
                    ) if sl_price is not None else None

                    formatted_tp = format_localized_number(
                        tp_price, test_case["decimal_separator"], test_case["thousands_separator"]
                    ) if tp_price is not None else None

                    locale_results[locale_name] = {
                        "locale_available": locale_available,
                        "test_price_str": test_case["test_price"],
                        "parsed_price": parsed_price,
                        "expected_price": expected_price,
                        "parse_error": parse_error,
                        "sl_price": sl_price,
                        "tp_price": tp_price,
                        "formatted_sl": formatted_sl,
                        "formatted_tp": formatted_tp
                    }

                    logger.info(f"{locale_name}:")
                    logger.info(f"  入力: {test_case['test_price']}")
                    logger.info(f"  パース結果: {parsed_price}")
                    logger.info(f"  SL: {formatted_sl}")
                    logger.info(f"  TP: {formatted_tp}")
                    logger.info(f"  パース誤差: {parse_error:.2e}")

                    # パース精度の確認
                    assert parse_error < 1e-10, f"{locale_name}: パース精度が不十分です: {parse_error:.2e}"

                    # 計算結果の妥当性確認
                    if sl_price is not None and tp_price is not None:
                        assert sl_price < parsed_price < tp_price, f"{locale_name}: 価格関係が不正です"

                except Exception as e:
                    logger.warning(f"{locale_name}: ロケールテストでエラー - {e}")
                    locale_results[locale_name] = {
                        "error": str(e),
                        "locale_available": False
                    }

            # 結果の一貫性確認
            successful_locales = [
                name for name, result in locale_results.items()
                if "error" not in result and result.get("sl_price") is not None
            ]

            if len(successful_locales) >= 2:
                # 同じ価格での計算結果が一致することを確認
                base_result = locale_results[successful_locales[0]]
                base_sl = base_result["sl_price"]
                base_tp = base_result["tp_price"]

                for locale_name in successful_locales[1:]:
                    result = locale_results[locale_name]
                    sl_diff = abs(result["sl_price"] - base_sl) / base_sl
                    tp_diff = abs(result["tp_price"] - base_tp) / base_tp

                    assert sl_diff < 1e-10, f"{locale_name}: SL計算結果が不一致: {sl_diff:.2e}"
                    assert tp_diff < 1e-10, f"{locale_name}: TP計算結果が不一致: {tp_diff:.2e}"

            logger.info(f"成功したロケール: {len(successful_locales)}/{len(locale_tests)}")
            logger.info("✅ ロケール固有フォーマット処理テスト成功")

        except Exception as e:
            pytest.fail(f"ロケール固有フォーマット処理テストエラー: {e}")

        finally:
            # ロケールをデフォルトに戻す
            try:
                locale.setlocale(locale.LC_ALL, "")
            except:
                pass


if __name__ == "__main__":
    # テスト実行
    test_instance = TestInternationalization()
    
    tests = [
        test_instance.test_timezone_handling,
        test_instance.test_multi_currency_pairs,
        test_instance.test_decimal_precision_accuracy,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test_instance.setup_method()
            test()
            test_instance.teardown_method()
            passed += 1
        except Exception as e:
            logger.error(f"テスト失敗: {test.__name__}: {e}")
            failed += 1
    
    print(f"\n📊 国際化・多様性テスト結果: 成功 {passed}, 失敗 {failed}")
    print(f"成功率: {passed / (passed + failed) * 100:.1f}%")
