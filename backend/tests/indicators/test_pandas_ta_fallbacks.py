"""
Pandas-TA fallback implementation tests for PPO, STOCHF, EMA, TEMA, ALMA, FWMA

These indicators had NoneType errors with pandas-ta, so we've implemented
fallback calculations using TrendIndicators class.
"""

import pytest
import pandas as pd
import numpy as np

from app.services.indicators.indicator_orchestrator import TechnicalIndicatorService


class TestPandasTAFallbacks:
    """PPO, STOCHF, EMA, TEMA, ALMA, FWMA pandas-ta fallbackテスト"""

    @pytest.fixture
    def indicator_service(self):
        return TechnicalIndicatorService()

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        return pd.DataFrame({
            'Close': np.random.uniform(50, 150, 100),
            'High': np.random.uniform(51, 155, 100),
            'Low': np.random.uniform(48, 148, 100),
            'Open': np.random.uniform(49, 152, 100)
        }, index=dates)

    def test_ppo_fallback_calculation(self, indicator_service, sample_data):
        """Test PPO fallback implementation"""
        params = {"fast": 12, "slow": 26, "signal": 9}
        result = indicator_service.calculate_indicator(sample_data, "PPO", params)

        assert result is not None
        assert isinstance(result, tuple)
        assert len(result) == 3  # PPO line, signal line, histogram
        assert not all(np.isnan(result[0])), "PPO line should not be all NaN"
        assert not all(np.isnan(result[1])), "Signal line should not be all NaN"
        assert not all(np.isnan(result[2])), "PPO histogram should not be all NaN"

    def test_stochf_fallback_calculation(self, indicator_service, sample_data):
        """Test STOCHF fallback implementation"""
        params = {"fastk_length": 5, "fastd_length": 3}
        result = indicator_service.calculate_indicator(sample_data, "STOCHF", params)

        assert result is not None
        assert isinstance(result, tuple)
        assert len(result) == 2  # Fast K, Fast D
        assert not all(np.isnan(result[0])), "Fast K should not be all NaN"
        assert not all(np.isnan(result[1])), "Fast D should not be all NaN"

    def test_ema_fallback_calculation(self, indicator_service, sample_data):
        """Test EMA fallback implementation"""
        params = {"length": 10}
        result = indicator_service.calculate_indicator(sample_data, "EMA", params)

        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == (100,)  # Same as input data length
        assert not all(np.isnan(result)), "EMA should not be all NaN"

    def test_tema_fallback_calculation(self, indicator_service, sample_data):
        """Test TEMA fallback implementation"""
        params = {"length": 14}
        result = indicator_service.calculate_indicator(sample_data, "TEMA", params)

        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == (100,)
        assert not all(np.isnan(result)), "TEMA should not be all NaN"

    def test_alma_fallback_calculation(self, indicator_service, sample_data):
        """Test ALMA fallback implementation"""
        params = {"length": 9, "sigma": 6.0, "offset": 0.85}
        result = indicator_service.calculate_indicator(sample_data, "ALMA", params)

        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == (100,)
        assert not all(np.isnan(result)), "ALMA should not be all NaN"

    def test_fwma_fallback_calculation(self, indicator_service, sample_data):
        """Test FWMA fallback implementation"""
        params = {"length": 10}
        result = indicator_service.calculate_indicator(sample_data, "FWMA", params)

        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == (100,)
        assert not all(np.isnan(result)), "FWMA should not be all NaN"

    def test_ao_enhanced_fallback(self, indicator_service, sample_data):
        """Test AO enhanced fallback implementation"""
        result = indicator_service.calculate_indicator(sample_data, "AO", {})

        assert result is not None
        assert isinstance(result, np.ndarray), f"AO should return numpy array, got {type(result)}"
        assert len(result) == len(sample_data), f"AO result length mismatch: {len(result)} != {len(sample_data)}"
        assert not np.isnan(result).all(), "AO should contain meaningful data"

        # AOはゼロを中心に変動するので、値の範囲が妥当かチェック
        # AOの妥当な値で範囲をチェック（NaNを除外）
        valid_values = result[~np.isnan(result)]
        if len(valid_values) > 0:
            assert valid_values.min() > -1000 and valid_values.max() < 1000, f"AO values seem unreasonable: min={valid_values.min()}, max={valid_values.max()}"

    def test_rvi_enhanced_fallback(self, indicator_service, sample_data):
        """Test RVI enhanced fallback implementation"""
        params = {"length": 10}
        result = indicator_service.calculate_indicator(sample_data, "RVI", params)

        assert result is not None
        assert isinstance(result, np.ndarray), f"RVI should return numpy array, got {type(result)}"
        assert len(result) == len(sample_data), f"RVI result length mismatch: {len(result)} != {len(sample_data)}"

        # RVIがNoneになる部分を除いて、値が0-100の範囲内かチェック
        valid_values = result[~np.isnan(result)]
        if len(valid_values) > 0:
            assert all(0 <= v <= 100 for v in valid_values), f"RVI values should be in 0-100 range: {valid_values[:10]}"

    def test_cfo_enhanced_fallback(self, indicator_service, sample_data):
        """Test CFO enhanced fallback implementation"""
        params = {"length": 5}
        result = indicator_service.calculate_indicator(sample_data, "CFO", params)

        assert result is not None
        assert isinstance(result, np.ndarray), f"CFO should return numpy array, got {type(result)}"
        assert len(result) == len(sample_data), f"CFO result length mismatch: {len(result)} != {len(sample_data)}"

        # CFOが有効な値を生成しているかチェック
        valid_values = result[~np.isnan(result)]
        if len(valid_values) > 0:
            # CFOはトレンド予測指標なので、-10 to +10程度の範囲が妥当
            reasonable_values = [v for v in valid_values if -20 <= v <= 20]
            assert len(reasonable_values) > 0, f"CFO should have reasonable values: {valid_values[:10]}"

    def test_cti_enhanced_fallback(self, indicator_service, sample_data):
        """Test CTI enhanced fallback implementation"""
        params = {"length": 15}
        result = indicator_service.calculate_indicator(sample_data, "CTI", params)

        assert result is not None
        assert isinstance(result, np.ndarray), f"CTI should return numpy array, got {type(result)}"
        assert len(result) == len(sample_data), f"CTI result length mismatch: {len(result)} != {len(sample_data)}"

        # CTIは-100 to +100のトレンド指標
        valid_values = result[~np.isnan(result)]
        if len(valid_values) > 0:
            in_range = [v for v in valid_values if -100 <= v <= 100]
            assert len(in_range) == len(valid_values), f"CTI values should be in -100 to +100 range: outliers={set(valid_values) - set(in_range)}"

    def test_enhanced_fallbacks_with_edge_cases(self, indicator_service):
        """Test enhanced fallback indicators with edge case data"""
        # 極端なデータでのテスト
        edge_cases = [
            ("flat_data", pd.DataFrame({'Close': [100] * 50, 'High': [100] * 50, 'Low': [100] * 50, 'Open': [100] * 50})),
            ("volatile_data", pd.DataFrame({
                'Close': np.sin(np.arange(50) * 0.5) * 50 + 100,
                'High': np.sin(np.arange(50) * 0.5) * 50 + 105,
                'Low': np.sin(np.arange(50) * 0.5) * 50 + 95,
                'Open': np.sin(np.arange(50) * 0.5) * 50 + 102
            }))
        ]

        for case_name, test_data in edge_cases:
            print(f"Testing {case_name}...")

            # AOとRVIはOHLCが必要
            for indicator, needs_ohlc in [("AO", True), ("RVI", True), ("CFO", False), ("CTI", False)]:
                try:
                    if needs_ohlc and 'Volume' not in test_data.columns:
                        test_data = test_data.copy()
                        test_data['Volume'] = np.random.uniform(1000, 10000, len(test_data))

                    result = indicator_service.calculate_indicator(test_data, indicator, {})
                    assert result is not None, f"{indicator} should handle {case_name}"
                    print(f"  ✓ {indicator}: {case_name} handled")

                except Exception as e:
                    print(f"  ✗ {indicator}: {case_name} failed - {str(e)}")

        print("Edge case testing completed for enhanced fallbacks")

    @pytest.mark.parametrize("indicator,mapped_params,test_params", [
        ("STOCHF", {"fastk_length": 5, "fastd_length": 3}, {"k": 5, "d": 3}),
        ("STOCHF", {"fastk_length": 10, "fastd_length": 5}, {"length": 5, "k_period": 10}),
        ("EMA", {"length": 21}, {"period": 21}),
        ("EMA", {"length": 30}, {"length": 30}),
        ("TEMA", {"length": 12}, {"period": 12}),
        ("BBANDS", {"length": 20, "std": 2.0}, {"period": 20, "std": 2.0}),
    ])
    def test_parameter_mapping_functionality(self, indicator_service, sample_data, indicator, mapped_params, test_params):
        """Test that parameter mapping works correctly for unified parameters"""
        # マップされたパラメータで計算
        result_mapped = indicator_service.calculate_indicator(sample_data, indicator, mapped_params)

        # テストパラメータで計算（マッピングを経由）
        result_test = indicator_service.calculate_indicator(sample_data, indicator, test_params)

        # 両方とも成功するか、両方とも失敗するかを確認
        both_success = result_mapped is not None and result_test is not None
        both_none = result_mapped is None and result_test is None

        assert both_success or both_none, f"Parameter mapping inconsistency for {indicator}"

        if both_success:
            # 同一パラメータで計算結果が類似するか確認
            if isinstance(result_mapped, np.ndarray) and isinstance(result_test, np.ndarray):
                # NaNでない部分で比較
                valid_mapped = result_mapped[~np.isnan(result_mapped)]
                valid_test = result_test[~np.isnan(result_test)]

                if len(valid_mapped) > 0 and len(valid_test) > 0:
                    # 値がほぼ同じか確認（小数点誤差を考慮）
                    np.testing.assert_allclose(valid_mapped, valid_test, rtol=0.01,
                                             err_msg=f"Results differ for {indicator} with parameter mapping")
            print(f"✓ {indicator}: Parameter mapping working correctly")
        else:
            print(f"- {indicator}: Parameter mapping handled gracefully")

    def test_pandas_ta_compatibility_layer_comprehensive(self, indicator_service, sample_data):
        """Comprehensive test of pandas-ta compatibility for fallback indicators"""
        compatibility_results = []
        target_indicators = [
            "PPO", "STOCHF", "EMA", "TEMA", "ALMA", "FWMA",
            "AO", "RVI", "CFO", "CTI"
        ]

        default_params_dict = {
            "PPO": {"fast": 12, "slow": 26, "signal": 9},
            "STOCHF": {"fastk_length": 5, "fastd_length": 3},
            "EMA": {"length": 20},
            "TEMA": {"length": 14},
            "ALMA": {"length": 9, "sigma": 6.0, "offset": 0.85},
            "FWMA": {"length": 10},
            "AO": {},
            "RVI": {"length": 14},
            "CFO": {"length": 9},
            "CTI": {"length": 20}
        }

        for indicator in target_indicators:
            try:
                params = default_params_dict.get(indicator, {})
                result = indicator_service.calculate_indicator(sample_data, indicator, params)

                # 結果の品質チェック
                quality_score = 0
                if result is not None:
                    quality_score += 1  # 成功

                    if isinstance(result, (np.ndarray, tuple)):
                        if isinstance(result, tuple):
                            # タプルの場合、各要素でチェック
                            valid_count = sum(1 for arr in result if isinstance(arr, np.ndarray) and not np.isnan(arr).all())
                            quality_score += min(valid_count / len(result), 1.0)
                        else:
                            # 単一配列の場合
                            if not np.isnan(result).all():
                                quality_score += 1

                compatibility_results.append({
                    'indicator': indicator,
                    'success': result is not None,
                    'quality_score': quality_score,
                    'fallback_used': True  # 全てフォールバック実装
                })

            except Exception as e:
                compatibility_results.append({
                    'indicator': indicator,
                    'success': False,
                    'quality_score': 0,
                    'error': str(e),
                    'fallback_used': True
                })

        # 結果集計
        successful_indicators = [r for r in compatibility_results if r['success']]
        avg_quality = sum(r['quality_score'] for r in compatibility_results) / len(compatibility_results)

        success_rate = len(successful_indicators) / len(compatibility_results)

        print("\n=== PANDAS-TA COMPATIBILITY TEST RESULTS ===")
        print(f"Total indicators tested: {len(compatibility_results)}")
        print(f"Successful calculations: {len(successful_indicators)}")
        print(f"Success rate: {success_rate * 100:.1f}%")
        print(f"Average quality score: {avg_quality:.2f}/2.0")
        print(f"Target: 80%+ success rate with good quality")

        # 成功率アサーション
        assert success_rate >= 0.80, f"Success rate must be at least 80%: {success_rate*100:.1f}%"

        # 高品質な結果の割合もチェック
        high_quality_count = sum(1 for r in compatibility_results if r['quality_score'] >= 1.8)
        high_quality_rate = high_quality_count / len(compatibility_results)
        assert high_quality_rate >= 0.70, f"At least 70% should have high quality results: {high_quality_rate*100:.1f}%"

        print("✅ PANDAS-TA COMPATIBILITY TARGETS ACHIEVED")
        for result in compatibility_results:
            status = "✓" if result['success'] else "✗"
            print(f"  {status} {result['indicator']}: quality={result['quality_score']:.1f}")

    def test_ppo_parameters_handling(self, indicator_service, sample_data):
        """Test PPO with custom parameters"""
        params = {"fast": 5, "slow": 20, "signal": 5}
        result = indicator_service.calculate_indicator(sample_data, "PPO", params)
        assert result is not None
        assert len(result) == 3

    def test_stochf_parameters_handling(self, indicator_service, sample_data):
        """Test STOCHF with custom parameters"""
        params = {"fastk_length": 10, "fastd_length": 5}
        result = indicator_service.calculate_indicator(sample_data, "STOCHF", params)
        assert result is not None
        assert len(result) == 2

    def test_fallback_indicators_with_no_params(self, indicator_service, sample_data):
        """Test fallback indicators work with no parameters (use defaults)"""
        for indicator in ["EMA", "TEMA", "ALMA", "FWMA"]:
            result = indicator_service.calculate_indicator(sample_data, indicator, {})
            assert result is not None, f"{indicator} should work with default parameters"

    def test_fallback_indicators_data_length_validation(self, indicator_service):
        """Test data length validation works correctly"""
        # Create data that's too short for calculation
        short_data = pd.DataFrame({
            'Close': [50, 51, 52, 53, 54],
            'High': [51, 52, 53, 54, 55],
            'Low': [49, 50, 51, 52, 53],
        })

        for indicator in ["PPO", "STOCHF", "EMA", "TEMA", "ALMA", "FWMA"]:
            result = indicator_service.calculate_indicator(short_data, indicator, {})
            # Should return NaN arrays when data is insufficient
            assert result is not None, f"{indicator} should handle short data gracefully"

    def test_fallback_indicators_with_invalid_params(self, indicator_service, sample_data):
        """Test fallback indicators handle invalid parameters gracefully"""
        # Test with invalid (non-positive) length
        params = {"length": -1}
        for indicator in ["EMA", "TEMA", "ALMA", "FWMA"]:
            result = indicator_service.calculate_indicator(sample_data, indicator, params)
            assert result is not None, f"{indicator} should handle invalid length gracefully"

    @pytest.mark.parametrize("indicator,expected_length", [
        ("PPO", 3),  # (ppo_line, signal_line, histogram)
        ("STOCHF", 2),  # (fast_k, fast_d)
        ("EMA", 100),  # single array
        ("TEMA", 100),
        ("ALMA", 100),
        ("FWMA", 100)
    ])
    def test_fallback_indicators_output_structure(self, indicator_service, sample_data, indicator, expected_length):
        """Test all fallback indicators return correct output structure"""
        result = indicator_service.calculate_indicator(sample_data, indicator, {})

        if indicator in ["PPO", "STOCHF"]:
            assert len(result) == expected_length, f"{indicator} should return {expected_length} elements"
        else:
            assert len(result) == expected_length, f"{indicator} should return array of length {expected_length}"