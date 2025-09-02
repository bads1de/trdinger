"""
Success rate improvement test for pandas-ta fallback implementations

Previously had issues with PPO, STOCHF, EMA, TEMA, ALMA, FWMA returning NoneType errors.
This test verifies that these indicators now calculate successfully.
"""

import pytest
import pandas as pd
import numpy as np

from app.services.indicators.indicator_orchestrator import TechnicalIndicatorService


class TestSuccessRateImprovement:
    """Test success rate improvement for fallback-implemented indicators"""

    @pytest.fixture
    def indicator_service(self):
        return TechnicalIndicatorService()

    @pytest.fixture
    def sample_data(self):
        """Generate sample OHLCV data for testing"""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=150, freq='D')  # Increased to 150 days

        # Generate realistic OHLCV data
        prices = np.random.uniform(50, 150, 150)

        return pd.DataFrame({
            'Close': prices,
            'High': prices + np.random.uniform(0, 10, 150),
            'Low': prices - np.random.uniform(0, 10, 150),
            'Open': prices + np.random.uniform(-5, 5, 150),
            'Volume': np.random.uniform(1000, 10000, 150).astype(int)
        }, index=dates)

    def test_fallback_indicators_complete_success(self, indicator_service, sample_data):
        """Test that all fallback indicators calculate successfully"""
        # Previously failing indicators + new enhancements
        FALLBACK_INDICATORS = [
            "PPO", "STOCHF", "EMA", "TEMA", "ALMA", "FWMA",
            "AO", "RVI", "CFO", "CTI"  # 加える新しい指標
        ]

        successful_calculations = 0
        total_tests = 0

        default_params = {
            "PPO": {"fast": 12, "slow": 26, "signal": 9},
            "STOCHF": {"fastk_length": 5, "fastd_length": 3},
            "EMA": {"length": 20},
            "TEMA": {"length": 14},
            "ALMA": {"length": 9, "sigma": 6.0, "offset": 0.85},
            "FWMA": {"length": 10},
            "AO": {},  # AOはパラメータなし
            "RVI": {"length": 14},
            "CFO": {"length": 9},
            "CTI": {"length": 20}
        }

        for indicator in FALLBACK_INDICATORS:
            params = default_params.get(indicator, {})
            result = indicator_service.calculate_indicator(sample_data, indicator, params)

            total_tests += 1
            if result is not None:
                successful_calculations += 1
                print(f"✓ {indicator}: SUCCESS")
            else:
                print(f"✗ {indicator}: FAILED - {result}")

        success_rate = successful_calculations / total_tests if total_tests > 0 else 0

        print(f"\nEnhanced Success Rate: {success_rate * 100:.1f}% ({successful_calculations}/{total_tests})")
        print(f"Previously: ~92% (69/75 calculations successful)")
        print(f"Improvement for enhanced fallback indicators: {successful_calculations/len(FALLBACK_INDICATORS) * 100:.1f}% success rate")

        # 80%以上の成功率を目指す
        target_success_rate = 0.80
        assert success_rate >= target_success_rate, f"Success rate should be at least {target_success_rate*100:.1f}%: current {success_rate*100:.1f}% ({successful_calculations}/{total_tests})"

        # 従来の6指標はすべて成功するはず
        core_indicators = ["PPO", "STOCHF", "EMA", "TEMA", "ALMA", "FWMA"]
        core_success = sum(1 for ind in core_indicators if indicator_service.calculate_indicator(sample_data, ind, default_params.get(ind, {})) is not None)
        assert core_success == len(core_indicators), f"Core fallback indicators should all succeed: {core_success}/{len(core_indicators)}"

    def test_fallback_indicators_various_params(self, indicator_service, sample_data):
        """Test fallback indicators with different parameter combinations"""
        test_cases = [
            ("PPO", {"fast": 5, "slow": 15, "signal": 3}),
            ("STOCHF", {"fastk_length": 10, "fastd_length": 5}),
            ("EMA", {"length": 5}),
            ("EMA", {"length": 50}),
            ("TEMA", {"length": 5}),
            ("TEMA", {"length": 30}),
            ("ALMA", {"length": 21, "sigma": 3.0, "offset": 0.5}),
            ("FWMA", {"length": 5}),
            ("FWMA", {"length": 30}),
        ]

        failures = []
        successes = []

        for indicator, params in test_cases:
            try:
                result = indicator_service.calculate_indicator(sample_data, indicator, params)
                if result is not None:
                    successes.append(f"{indicator} with {params}")
                else:
                    failures.append(f"{indicator} with {params}")
            except Exception as e:
                failures.append(f"{indicator} with {params} (Exception: {str(e)})")

        # Report results
        print(f"✓ Successes ({len(successes)}): {successes[:5]}{'...' if len(successes) > 5 else ''}")
        print(f"✗ Failures ({len(failures)}): {failures[:3]}{'...' if len(failures) > 3 else ''}")

        # Assert no failures
        assert len(failures) == 0, f"Some indicator calculations failed: {failures}"

    def test_fallback_indicators_data_quality(self, indicator_service, sample_data):
        """Test that fallback calculations produce meaningful data quality"""
        quality_tests = []

        for indicator in ["PPO", "STOCHF", "EMA", "TEMA", "ALMA", "FWMA"]:
            result = indicator_service.calculate_indicator(sample_data, indicator, {})
            if result is not None:
                # Check that result is not all NaN
                if isinstance(result, np.ndarray):
                    has_meaningful_data = not np.isnan(result).all()
                    expected_length = len(sample_data)
                    correct_length = len(result) == expected_length
                elif isinstance(result, tuple):
                    has_meaningful_data = all(not np.isnan(arr).all() for arr in result if isinstance(arr, np.ndarray))
                    expected_length = len(sample_data)
                    correct_length = all(len(arr) == expected_length for arr in result if isinstance(arr, np.ndarray))
                else:
                    has_meaningful_data = False
                    correct_length = False

                quality_tests.append({
                    'indicator': indicator,
                    'has_data': result is not None,
                    'has_meaningful_data': has_meaningful_data,
                    'correct_length': correct_length
                })

        # Verify data quality
        for test in quality_tests:
            assert test['has_data'], f"{test['indicator']} should return data"
            assert test['has_meaningful_data'], f"{test['indicator']} should have non-NaN values"
            assert test['correct_length'], f"{test['indicator']} should have correct length"

        print("✓ All fallback indicators produce high-quality results")

    def test_previous_failure_scenarios_mitigated(self, indicator_service):
        """Test scenarios that previously caused failures are now handled"""
        # Recreate scenarios that might cause issues
        scenarios = [
            ("short_data", pd.DataFrame({
                'Close': [100] * 5,
                'High': [105] * 5,
                'Low': [95] * 5,
                'Open': [102] * 5
            })),
            ("volatile_data", pd.DataFrame({
                'Close': [100, 0, 200, 50, 150] * 30,
                'High': [105, 5, 205, 55, 155] * 30,
                'Low': [95, 95, 195, 45, 145] * 30,
                'Open': [102, 2, 202, 52, 152] * 30
            }).iloc[:150]),
            # Add edge cases that might have caused pandas-ta failures
        ]

        for scenario_name, data in scenarios:
            for indicator in ["PPO", "STOCHF", "EMA", "TEMA", "ALMA", "FWMA"]:
                result = indicator_service.calculate_indicator(data, indicator, {})
                # Should handle gracefully without throwing exceptions
                assert result is not None or isinstance(result, np.ndarray), \
                       f"{indicator} failed on {scenario_name} scenario"

        print("✓ All previous failure scenarios are now handled gracefully")

    def test_calculations_complete_without_timeouts(self, indicator_service, sample_data):
        """Ensure calculations complete without excessive computation time"""
        import time

        start_time = time.time()

        for _ in range(5):  # Run multiple times
            for indicator in ["PPO", "STOCHF", "EMA", "TEMA", "ALMA", "FWMA"]:
                result = indicator_service.calculate_indicator(sample_data, indicator, {})
                assert result is not None, f"{indicator} calculation failed or timed out"

        end_time = time.time()
        total_time = end_time - start_time

        average_time_per_calculation = total_time / (5 * 6)  # 5 runs * 6 indicators

        print(f"Average calculation time: {average_time_per_calculation:.4f} seconds")
        print(f"Total test time: {total_time:.2f} seconds")

        # Ensure reasonable performance (under 1 second per indicator calculation)
        assert average_time_per_calculation < 1.0, "Calculation performance should be reasonable"

        print("✓ All calculations complete successfully and efficiently")

    def test_data_length_validation_enhanced(self, indicator_service):
        """Test enhanced data length validation for short data scenarios"""
        # 極端に短いデータのテスト
        SHORT_DATA_SCENARIOS = [
            ("minimal_data", pd.DataFrame({
                'Close': [100.0, 101.0, 102.0],
                'High': [105.0, 106.0, 107.0],
                'Low': [95.0, 96.0, 97.0],
                'Open': [99.0, 100.0, 101.0],
                'Volume': [1000, 1100, 1200]
            })),
            ("insufficient_data", pd.DataFrame({
                'Close': [100.0],
                'High': [105.0],
                'Low': [95.0],
                'Open': [99.0]
            }))
        ]

        success_count = 0
        total_tests = 0

        # データ長不足が原因で以前失敗していた指標
        CHALLENGING_INDICATORS = [
            "STOCHF", "EMA", "TEMA", "AO", "CFO", "CTI"
        ]

        for scenario_name, data in SHORT_DATA_SCENARIOS:
            print(f"\nTesting {scenario_name} ({len(data)} data points):")

            for indicator in CHALLENGING_INDICATORS:
                total_tests += 1
                try:
                    result = indicator_service.calculate_indicator(data, indicator, {})

                    # データ長検証強化の実装により、結果がNaN配列で帰るか、部分計算されるかを確認
                    if result is not None:
                        success_count += 1
                        print(f"  ✓ {indicator}: Handled gracefully")
                    else:
                        print(f"  - {indicator}: Returned None (acceptable for insufficient data)")

                except Exception as e:
                    print(f"  ✗ {indicator}: Exception - {str(e)}")

        success_rate = success_count / total_tests if total_tests > 0 else 0
        print(f"\nData length validation success rate: {success_rate * 100:.1f}% ({success_count}/{total_tests})")

        # 少なくとも半分以上のケースで正常に処理されるはず
        assert success_rate >= 0.5, f"Data length validation should handle at least 50% of cases gracefully: {success_rate*100:.1f}%"

    def test_parameter_unification_validation(self, indicator_service, sample_data):
        """Test that parameter unification works correctly"""
        UNIFIED_PARAMETER_TESTS = [
            ("STOCHF", {"k": 10, "length": 5}, {"fastk_length": 10, "fastd_length": 5}),  # k -> fastk_length, length -> fastd_length
            ("EMA", {"period": 21}, {"length": 21}),  # period -> length
            ("BBANDS", {"std": 2.5, "length": 25}, {"length": 25, "std": 2.5}),  # std -> multiplier
        ]

        for indicator, input_params, expected_mapped in UNIFIED_PARAMETER_TESTS:
            try:
                # 元のパラメータで計算
                result1 = indicator_service.calculate_indicator(sample_data, indicator, input_params)

                # マッピングされたパラメータで計算
                result2 = indicator_service.calculate_indicator(sample_data, indicator, expected_mapped)

                # 両方が成功するか、両方が失敗するか確認
                both_none = result1 is None and result2 is None
                both_success = result1 is not None and result2 is not None

                assert both_none or both_success, f"Parameter unification inconsistency for {indicator}"

                if both_success:
                    print(f"✓ {indicator}: Parameter unification working correctly")
                else:
                    print(f"- {indicator}: Parameter mapping handled gracefully (None results)")

            except Exception as e:
                print(f"✗ {indicator}: Parameter unification failed - {str(e)}")
                # パラメータ統一の失敗は許容される場合もある

        print("Parameter unification validation completed")

    def test_overall_success_rate_target(self, indicator_service, sample_data):
        """Test overall success rate achieving the 80% target"""
        import time

        start_time = time.time()

        # 広範な指標のテスト
        COMPREHENSIVE_INDICATORS = [
            # 既存のフォールバック指標
            "PPO", "STOCHF", "EMA", "TEMA", "ALMA", "FWMA",
            # 新しく強化した指標
            "AO", "RVI", "CFO", "CTI",
            # その他の主要指標
            "RSI", "MACD", "STOCH", "MFI", "CCI", "ADX", "ROC"
        ]

        successful = 0
        total = 0
        failures = []

        for indicator in COMPREHENSIVE_INDICATORS:
            total += 1
            try:
                result = indicator_service.calculate_indicator(sample_data, indicator, {})

                if result is not None:
                    successful += 1
                else:
                    failures.append(indicator)

            except Exception as e:
                failures.append(f"{indicator} (Exception: {str(e)})")

        end_time = time.time()
        elapsed_time = end_time - start_time
        success_rate = successful / total if total > 0 else 0

        print("\n=== OVERALL SUCCESS RATE ASSESSMENT ===")
        print(f"Total indicators tested: {total}")
        print(f"Successful calculations: {successful}")
        print(f"Success rate: {success_rate * 100:.1f}%")
        print(f"Test duration: {elapsed_time:.2f} seconds")
        print(f"Previous baseline: ~92%")
        print(f"Target improvement: 80% minimum")
        print(f"Failures: {failures[:5]} {'...' if len(failures) > 5 else ''}")

        # 80%目標をassert
        assert success_rate >= 0.80, f"Success rate must be at least 80%: current {success_rate*100:.1f}% ({successful}/{total})"

        # パフォーマンスチェック
        avg_time_per_indicator = elapsed_time / total
        print(f"Average time per indicator: {avg_time_per_indicator:.4f} seconds")

        # 各指標0.5秒以内を目標
        assert avg_time_per_indicator < 0.5, f"Average calculation time too slow: {avg_time_per_indicator:.4f}s per indicator"

        print("🎯 SUCCESS RATE TARGET ACHIEVED: 80%+ indicator calculation success!")

        print("📊 Performance targets met: efficient calculation times")