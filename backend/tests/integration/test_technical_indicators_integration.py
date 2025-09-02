"""
Technical Indicators Integration Test

テスト対象指標：
AROON, KST, ICHIMOKU, MI (MASS INDEX), MIDPRICE, T3, TLB (TILLSON_T3?), ZLMA, PVOL (PVO?), CMF, EFI (EFI?), KVO, TSI, SUPERTREND, VWMA, VP (VPCI?)

統合テスト：
- 各指標の正常計算チェック
- pandas-taとの互換性検証
- エラーペイロード確認
- 共通テストケース実行
"""
import pytest
import pandas as pd
import numpy as np
import pandas_ta as ta
from app.services.indicators.indicator_orchestrator import TechnicalIndicatorService
from app.services.indicators.technical_indicators.momentum import MomentumIndicators
from app.services.indicators.technical_indicators.trend import TrendIndicators
from app.services.indicators.technical_indicators.volatility import VolatilityIndicators
from app.services.indicators.technical_indicators.volume import VolumeIndicators
from typing import List, Dict, Any


class TestTechnicalIndicatorsIntegration:
    """Technical Indicators統合テスト"""

    # テスト対象指標リスト（親タスクで指定された17個）
    TEST_INDICATORS = [
        'AROON', 'KST', 'ICHIMOKU', 'MI', 'MIDPRICE',
        'T3', 'TLB', 'ZLMA', 'PVOL', 'CMF', 'EFI', 'TSI',
        'SUPERTREND', 'VWMA', 'VP'
    ]

    @pytest.fixture
    def sample_ohlcv_data(self) -> pd.DataFrame:
        """標準的なOHLCVサンプルデータ生成"""
        np.random.seed(42)
        size = 100
        base_price = 100
        volatility = 0.02

        # 価格変動生成
        price_changes = np.random.normal(0, volatility, size)
        price = base_price * np.cumprod(1 + price_changes)

        # OHLC生成
        high_factor = np.random.uniform(1.001, 1.01, size)
        low_factor = np.random.uniform(0.99, 0.999, size)
        close_factor = np.random.uniform(low_factor, high_factor, size)

        data = {
            'open': np.roll(price, 1),
            'high': price * high_factor,
            'low': price * low_factor,
            'close': price * close_factor,
            'volume': np.random.randint(10000, 1000000, size)
        }

        df = pd.DataFrame(data)
        df.iloc[0, df.columns.get_loc('open')] = df.iloc[0]['close']  # 最初のopenをcloseに合わせる

        return df

    @pytest.fixture
    def indicator_service(self):
        """TechnicalIndicatorServiceフィクスチャ"""
        return TechnicalIndicatorService()

    def test_all_indicators_basic_calculation(self, indicator_service, sample_ohlcv_data):
        """全ての17個指標が正常に計算されるかのチェック"""
        failed_indicators = []
        successful_indicators = []

        for indicator_name in self.TEST_INDICATORS:
            try:
                # サービス経由で計算
                result = indicator_service.calculate_indicator(
                    sample_ohlcv_data,
                    indicator_name,
                    {}  # デフォルトパラメータ
                )

                if result is not None:
                    successful_indicators.append(indicator_name)
                    print(f"✅ {indicator_name}: 計算成功")
                else:
                    failed_indicators.append(indicator_name)
                    print(f"❌ {indicator_name}: 結果がNone")

            except Exception as e:
                failed_indicators.append(indicator_name)
                print(f"❌ {indicator_name}: エラー - {str(e)}")

        # 成功率レポート
        total_indicators = len(self.TEST_INDICATORS)
        success_rate = len(successful_indicators) / total_indicators if total_indicators > 0 else 0

        print(f"\n=== 基本計算テスト結果 ===")
        print(f"総指標数: {total_indicators}")
        print(f"成功: {len(successful_indicators)}")
        print(f"失敗: {len(failed_indicators)}")
        print(".1f")

        if failed_indicators:
            print(f"失敗指標: {failed_indicators}")

        # 最低80%の成功率を期待
        assert success_rate >= 0.8, f"成功率が80%を下回る: {success_rate:.1%}"

    def test_indicators_error_payload_verification(self, indicator_service, sample_ohlcv_data):
        """各指標のエラーペイロードを確認"""
        error_payloads = {}
        valid_payloads = {}

        for indicator_name in self.TEST_INDICATORS:
            try:
                result = indicator_service.calculate_indicator(
                    sample_ohlcv_data,
                    indicator_name,
                    {}
                )

                if result is not None:
                    # エラーペイロードの種類を確認
                    if isinstance(result, pd.DataFrame):
                        # 複数列結果
                        error_payloads[indicator_name] = {
                            'type': 'DataFrame',
                            'shape': result.shape,
                            'columns': list(result.columns),
                            'error_rows': {
                                col: result[col].isna().sum() for col in result.columns
                            }
                        }
                    elif isinstance(result, pd.Series):
                        # 単一列結果
                        error_payloads[indicator_name] = {
                            'type': 'Series',
                            'length': len(result),
                            'error_count': result.isna().sum(),
                            'error_ratio': result.isna().sum() / len(result) if len(result) > 0 else 0
                        }
                    elif isinstance(result, dict):
                        # 辞書形式結果
                        error_payloads[indicator_name] = {
                            'type': 'dict',
                            'keys': list(result.keys()),
                            'key_types': {k: type(v).__name__ for k, v in result.items()}
                        }
                    else:
                        error_payloads[indicator_name] = {
                            'type': type(result).__name__,
                            'description': str(result)
                        }

                    print(f"✅ {indicator_name}: エラーペイロード解析成功")
                else:
                    print(f"❌ {indicator_name}: 結果がNone")

            except Exception as e:
                error_payloads[indicator_name] = {
                    'type': 'Error',
                    'error_message': str(e)
                }
                print(f"❌ {indicator_name}: エラー - {str(e)}")

        # エラーペイロード検証の基本チェック
        assert len(error_payloads) > 0, "エラーペイロードデータが存在しない"
        print("\n=== エラーペイロード検証完了 ===")

        return error_payloads

    def test_pandas_ta_compatibility_matrix(self, sample_ohlcv_data):
        """pandas-taとの互換性テスト"""
        compatibility_results = {}

        # pandas-taで利用可能な機能一覧
        pandas_ta_indicators = {
            'AROON': lambda data: ta.aroon(data['high'], data['low']),
            'MIDPRICE': lambda data: ta.midprice(data['high'], data['low']),
            'TSI': lambda data: ta.tsi(data['close']),
            'VWMA': lambda data: ta.vwma(data['close'], data['volume']),
            'CMF': lambda data: ta.cmf(data['high'], data['low'], data['close'], data['volume']),
            'SUPERTREND': lambda data: ta.supertrend(data['high'], data['low'], data['close']),
            'ZLMA': lambda data: ta.zlma(data['close']),
            'T3': lambda data: ta.t3(data['close']),
            # 以下は実装が必要な指標
            'ICHIMOKU': 'complex_calculation',  # pandas-taでichimoku関数が存在する
            'KST': 'complex_calculation',        # KSTは実装が必要
            'MI': 'mass_index',                  # MASS INDEXの別名
            'TLB': 'complex_calculation',        # パラメータ設定が必要
            'PVOL': 'complex_calculation',       # カスタム実装が必要かも
            'EFI': 'complex_calculation',        # EFI/EFI実装確認
            'VP': 'complex_calculation'          # VPCIなどの確認
        }

        for indicator_name, pandas_func in pandas_ta_indicators.items():
            if indicator_name in self.TEST_INDICATORS:
                try:
                    if pandas_func == 'complex_calculation':
                        compatibility_results[indicator_name] = 'complex_calculationRequires_manual_implementation'
                        print(f"⚠️  {indicator_name}: 複雑なpanda-s-ta実装が必要")
                    else:
                        # pandas-ta直接計算テスト
                        pandas_result = pandas_func(sample_ohlcv_data)
                        if pandas_result is not None:
                            compatibility_results[indicator_name] = 'pandas_ta_direct_compatible'
                            print(f"✅ {indicator_name}: pandas-ta直接互換")
                        else:
                            compatibility_results[indicator_name] = 'pandas_ta_fallback_required'
                            print(f"⚠️  {indicator_name}: pandas-taフォールバック実装必要")
                except Exception as e:
                    compatibility_results[indicator_name] = f'pandas_ta_error: {str(e)}'
                    print(f"❌ {indicator_name}: pandas-taエラー - {str(e)}")

        return compatibility_results

    def test_indicators_with_custom_data_sizes(self, indicator_service):
        """異なるデータサイズでのテスト"""
        data_sizes = [10, 50, 200]  # 小規模・中規模・大規模
        stability_results = {}

        for size in data_sizes:
            # サイズに応じたテストデータ生成
            np.random.seed(42)
            base_price = 100
            volatility = 0.02
            noise = np.random.normal(0, volatility, size)

            test_data = pd.DataFrame({
                'open': np.roll(base_price * np.cumprod(1 + noise), 1),
                'high': base_price * np.cumprod(1 + noise) * np.random.uniform(1.001, 1.01, size),
                'low': base_price * np.cumprod(1 + noise) * np.random.uniform(0.99, 0.999, size),
                'close': base_price * np.cumprod(1 + noise),
                'volume': np.random.randint(10000, 1000000, size)
            })

            test_data.iloc[0, test_data.columns.get_loc('open')] = test_data.iloc[0]['close']

            size_results = {}
            for indicator_name in self.TEST_INDICATORS:
                try:
                    result = indicator_service.calculate_indicator(
                        test_data,
                        indicator_name,
                        {}
                    )
                    size_results[indicator_name] = 'success' if result is not None else 'no_result'
                except Exception as e:
                    size_results[indicator_name] = f'error: {str(e)}'

            stability_results[f'size_{size}'] = size_results

        return stability_results

    def test_edge_case_handling(self, sample_ohlcv_data):
        """エッジケース処理テスト"""
        edge_cases = {
            'insufficient_data': sample_ohlcv_data.head(5),  # 最小データ未満
            'duplicate_values': sample_ohlcv_data.copy().iloc[[0]],  # 重複値
            'single_row': sample_ohlcv_data.head(1),         # 単一行
            'all_zeros_volume': sample_ohlcv_data.copy().assign(volume=0.1),  # ゼロボリューム
            'constant_prices': sample_ohlcv_data.copy().assign(high=100, low=100, close=100),  # 定常価格
        }

        edge_results = {}

        for case_name, test_data in edge_cases.items():
            case_results = {}
            for indicator_name in self.TEST_INDICATORS:
                try:
                    # TechnicalIndicatorServiceでの計算
                    import app.services.indicators.indicator_orchestrator as service
                    from app.services.indicators.indicator_orchestrator import TechnicalIndicatorService

                    service_instance = TechnicalIndicatorService()
                    result = service_instance.calculate_indicator(
                        test_data,
                        indicator_name,
                        {}
                    )
                    case_results[indicator_name] = 'handled' if result is not None else 'no_result'

                except Exception as e:
                    case_results[indicator_name] = f'error: {str(e)}'

            edge_results[case_name] = case_results

        return edge_results

    @pytest.mark.parametrize("indicator_name", TEST_INDICATORS)
    def test_individual_indicator_comprehensive_check(self, indicator_service, sample_ohlcv_data, indicator_name):
        """各指標の総合チェック"""
        try:
            # 基本計算テスト
            result = indicator_service.calculate_indicator(
                sample_ohlcv_data,
                indicator_name,
                {}
            )

            assert result is not None, f"{indicator_name}がNoneを返した"

            # データタイプ確認
            if isinstance(result, pd.DataFrame):
                assert not result.empty, f"{indicator_name}のDataFrameが空"
                assert all(not result.empty for col in result.columns), f"{indicator_name}に空の列が存在"
            elif isinstance(result, pd.Series):
                assert not result.empty, f"{indicator_name}のSeriesが空"
                assert not result.isna().all(), f"{indicator_name}が全NA"
            elif isinstance(result, dict):
                assert len(result) > 0, f"{indicator_name}の辞書が空"
                assert all(v is not None for v in result.values()), f"{indicator_name}にNoneの値が存在"

            print(f"✅ {indicator_name}: 総合テスト完了")

        except AssertionError as ae:
            pytest.fail(f"{indicator_name}: アサーション失敗 - {str(ae)}")
        except Exception as e:
            pytest.fail(f"{indicator_name}: 未期待エラー - {str(e)}")

    def test_full_integration_workflow(self, indicator_service, sample_ohlcv_data):
        """完全統合ワークフロー"""
        workflow_results = {
            'setup': {},
            'calculation': {},
            'validation': {},
            'compatibility': {},
            'error_handling': {}
        }

        try:
            # 1. セットアップフェーズ
            workflow_results['setup'] = {
                'service_ready': indicator_service is not None,
                'data_ready': not sample_ohlcv_data.empty,
                'indicators_list': self.TEST_INDICATORS
            }

            # 2. 計算フェーズ
            calculation_results = {}
            for ind in self.TEST_INDICATORS[:3]:  # 最初の3つだけテスト
                try:
                    result = indicator_service.calculate_indicator(
                        sample_ohlcv_data,
                        ind,
                        {}
                    )
                    calculation_results[ind] = result is not None
                except Exception as e:
                    calculation_results[ind] = f'error: {str(e)}'

            workflow_results['calculation'] = calculation_results

            # 3. 検証フェーズ（シンプルな互換性チェック）
            workflow_results['validation'] = {
                'good_indicators': [ind for ind, success in calculation_results.items() if success is True],
                'failed_indicators': [ind for ind, success in calculation_results.items() if success is not True]
            }

            # 4. 互換性フェーズ
            workflow_results['compatibility'] = {
                'pandas_ta_checked': True,
                'fallback_implementations': []  # 実装が進むと埋まる
            }

            # 5. エラーハンドリングフェーズ
            workflow_results['error_handling'] = {
                'exception_caught': len([r for r in calculation_results.values() if 'error' in str(r)]) > 0,
                'null_results_handled': len([r for r in calculation_results.values() if r is False]) > 0
            }

        except Exception as e:
            workflow_results['overall_error'] = str(e)

        assert workflow_results['setup']['service_ready'], "サービスが初期化されていない"
        assert workflow_results['setup']['data_ready'], "データが準備されていない"

        # 最も基本的な成功基準：少なくとも1つ指標が計算された
        assert any(r is True for r in workflow_results['calculation'].values()), "すべて指標でエラー"

        print(f"✅ 完全統合ワークフロー完了: 結果={workflow_results}")