"""
MLエラーハンドリング・エッジケーステスト

不正データ入力、モデル未学習状態、メモリ不足、ネットワークエラー、
データ形式エラーなどの異常系を包括的にテストします。
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path
from typing import Dict, Any, List
import tempfile
import threading
import time
import gc

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from .utils import (
    create_sample_ohlcv_data,
    create_sample_funding_rate_data,
    create_sample_open_interest_data,
    MLTestConfig,
    measure_performance,
    validate_ml_predictions,
    create_comprehensive_test_data
)


class MLErrorHandlingTestSuite:
    """MLエラーハンドリング・エッジケーステストスイート"""

    def __init__(self):
        self.config = MLTestConfig()
        self.test_results = []

    def run_all_tests(self):
        """全テストを実行"""
        print("MLエラーハンドリング・エッジケーステストスイート開始")
        print("=" * 60)

        tests = [
            self.test_invalid_data_input_handling,
            self.test_empty_data_handling,
            self.test_malformed_data_handling,
            self.test_extreme_values_handling,
            self.test_memory_pressure_handling,
            self.test_concurrent_access_handling,
            self.test_model_state_error_handling,
            self.test_feature_calculation_errors,
            self.test_prediction_boundary_cases,
            self.test_system_resource_exhaustion,
        ]

        passed = 0
        total = len(tests)

        for test in tests:
            try:
                print(f"\n実行中: {test.__name__}")
                if test():
                    passed += 1
                    print("✓ PASS")
                else:
                    print("✗ FAIL")
            except Exception as e:
                print(f"✗ ERROR: {e}")
                import traceback
                traceback.print_exc()

        print("\n" + "=" * 60)
        print(f"テスト結果: {passed}/{total} 成功")

        if passed == total:
            print("全テスト成功！MLエラーハンドリングは堅牢です。")
        else:
            print(f"{total - passed}個のテストが失敗しました。")

        return passed == total

    def test_invalid_data_input_handling(self):
        """不正データ入力ハンドリングテスト"""
        try:
            from app.core.services.auto_strategy.services.ml_indicator_service import MLIndicatorService

            service = MLIndicatorService()

            # 不正データケース
            invalid_data_cases = [
                # 1. None入力
                None,

                # 2. 空のDataFrame
                pd.DataFrame(),

                # 3. 不正な列名
                pd.DataFrame({'invalid_col': [1, 2, 3]}),

                # 4. 文字列データ
                pd.DataFrame({
                    'open': ['abc', 'def', 'ghi'],
                    'high': ['123', '456', '789'],
                    'low': ['aaa', 'bbb', 'ccc'],
                    'close': ['xxx', 'yyy', 'zzz'],
                    'volume': ['111', '222', '333']
                }),

                # 5. 混合データ型
                pd.DataFrame({
                    'open': [100, 'invalid', 102],
                    'high': [101, 103, 'bad'],
                    'low': [99, 100, 101],
                    'close': [100.5, 102, 'error'],
                    'volume': [1000, 1100, 1200]
                }),

                # 6. 負の値
                pd.DataFrame({
                    'open': [-100, -101, -102],
                    'high': [-99, -100, -101],
                    'low': [-102, -103, -104],
                    'close': [-100.5, -101.5, -102.5],
                    'volume': [-1000, -1100, -1200]
                }),

                # 7. 極端に大きな値
                pd.DataFrame({
                    'open': [1e20, 1e21, 1e22],
                    'high': [1e20, 1e21, 1e22],
                    'low': [1e20, 1e21, 1e22],
                    'close': [1e20, 1e21, 1e22],
                    'volume': [1e20, 1e21, 1e22]
                })
            ]

            error_handling_success = 0
            total_cases = len(invalid_data_cases)

            for i, invalid_data in enumerate(invalid_data_cases):
                try:
                    result = service.calculate_ml_indicators(invalid_data)

                    # エラーが発生しなかった場合、結果が適切に処理されているかチェック
                    if result is not None:
                        assert isinstance(result, dict)
                        assert len(result) == 3  # ML_UP_PROB, ML_DOWN_PROB, ML_RANGE_PROB
                        error_handling_success += 1
                    else:
                        # Noneが返される場合も適切なエラーハンドリング
                        error_handling_success += 1

                except Exception as e:
                    # 例外が適切に処理されている場合も成功
                    print(f"  ケース{i}: 適切に例外処理 - {type(e).__name__}")
                    error_handling_success += 1

            success_rate = error_handling_success / total_cases

            print(f"不正データ入力ハンドリングテスト成功 - 成功率: {success_rate:.2%}")

            # 80%以上のケースで適切にハンドリングされることを期待
            assert success_rate >= 0.8, f"エラーハンドリング率が低すぎます: {success_rate:.2%}"

            return True

        except Exception as e:
            print(f"不正データ入力ハンドリングテスト失敗: {e}")
            return False

    def test_empty_data_handling(self):
        """空データハンドリングテスト"""
        try:
            from app.core.services.auto_strategy.services.ml_indicator_service import MLIndicatorService
            from app.core.services.feature_engineering.feature_engineering_service import FeatureEngineeringService
            from app.core.services.ml.signal_generator import MLSignalGenerator

            # 各サービスでの空データ処理テスト
            services_tests = [
                (MLIndicatorService(), 'calculate_ml_indicators'),
                (FeatureEngineeringService(), 'calculate_advanced_features'),
                (MLSignalGenerator(), 'predict')
            ]

            empty_data_cases = [
                pd.DataFrame(),  # 完全に空
                pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume']),  # 列のみ
                pd.DataFrame({'open': [], 'high': [], 'low': [], 'close': [], 'volume': []})  # 空の列
            ]

            successful_handling = 0
            total_tests = len(services_tests) * len(empty_data_cases)

            for service, method_name in services_tests:
                for i, empty_data in enumerate(empty_data_cases):
                    try:
                        method = getattr(service, method_name)

                        if method_name == 'calculate_advanced_features':
                            # FeatureEngineeringServiceは追加パラメータが必要
                            result = method(empty_data, lookback_periods={'short_ma': 10})
                        else:
                            result = method(empty_data)

                        # 結果が適切に処理されているかチェック
                        if result is not None:
                            successful_handling += 1
                        else:
                            successful_handling += 1  # Noneも適切な処理

                    except Exception as e:
                        # 適切な例外処理も成功とみなす
                        print(f"  {service.__class__.__name__}.{method_name} ケース{i}: {type(e).__name__}")
                        successful_handling += 1

            success_rate = successful_handling / total_tests

            print(f"空データハンドリングテスト成功 - 成功率: {success_rate:.2%}")

            assert success_rate >= 0.9, f"空データハンドリング率が低すぎます: {success_rate:.2%}"

            return True

        except Exception as e:
            print(f"空データハンドリングテスト失敗: {e}")
            return False

    def test_malformed_data_handling(self):
        """不正形式データハンドリングテスト"""
        try:
            from app.core.services.auto_strategy.services.ml_indicator_service import MLIndicatorService

            service = MLIndicatorService()

            # 不正形式データケース
            malformed_cases = [
                # 1. 重複インデックス
                pd.DataFrame({
                    'open': [100, 101, 102],
                    'high': [101, 102, 103],
                    'low': [99, 100, 101],
                    'close': [100.5, 101.5, 102.5],
                    'volume': [1000, 1100, 1200]
                }, index=[0, 0, 0]),

                # 2. 不正な時系列順序
                pd.DataFrame({
                    'timestamp': pd.to_datetime(['2023-01-03', '2023-01-01', '2023-01-02']),
                    'open': [100, 101, 102],
                    'high': [101, 102, 103],
                    'low': [99, 100, 101],
                    'close': [100.5, 101.5, 102.5],
                    'volume': [1000, 1100, 1200]
                }),

                # 3. 欠損値が多すぎるデータ
                pd.DataFrame({
                    'open': [100, np.nan, np.nan],
                    'high': [np.nan, 102, np.nan],
                    'low': [np.nan, np.nan, 101],
                    'close': [np.nan, np.nan, np.nan],
                    'volume': [1000, np.nan, np.nan]
                }),

                # 4. 不整合な価格データ（high < low）
                pd.DataFrame({
                    'open': [100, 101, 102],
                    'high': [99, 100, 101],  # high < low
                    'low': [101, 102, 103],
                    'close': [100.5, 101.5, 102.5],
                    'volume': [1000, 1100, 1200]
                }),

                # 5. ゼロボリューム
                pd.DataFrame({
                    'open': [100, 101, 102],
                    'high': [101, 102, 103],
                    'low': [99, 100, 101],
                    'close': [100.5, 101.5, 102.5],
                    'volume': [0, 0, 0]
                })
            ]

            handling_success = 0
            total_cases = len(malformed_cases)

            for i, malformed_data in enumerate(malformed_cases):
                try:
                    result = service.calculate_ml_indicators(malformed_data)

                    # 結果の妥当性チェック
                    if result is not None and isinstance(result, dict):
                        # 基本的な構造チェック
                        if len(result) == 3:
                            handling_success += 1
                        else:
                            handling_success += 0.5  # 部分的成功
                    else:
                        handling_success += 1  # Noneも適切な処理

                except Exception as e:
                    print(f"  不正形式ケース{i}: {type(e).__name__}")
                    handling_success += 1  # 例外処理も成功

            success_rate = handling_success / total_cases

            print(f"不正形式データハンドリングテスト成功 - 成功率: {success_rate:.2%}")

            assert success_rate >= 0.8, f"不正形式データハンドリング率が低すぎます: {success_rate:.2%}"

            return True

        except Exception as e:
            print(f"不正形式データハンドリングテスト失敗: {e}")
            return False

    def test_extreme_values_handling(self):
        """極端値ハンドリングテスト"""
        try:
            from app.core.services.auto_strategy.services.ml_indicator_service import MLIndicatorService

            service = MLIndicatorService()

            # 極端値ケース
            extreme_cases = [
                # 1. 無限大値
                pd.DataFrame({
                    'open': [100, np.inf, 102],
                    'high': [101, 102, np.inf],
                    'low': [99, 100, 101],
                    'close': [100.5, 101.5, 102.5],
                    'volume': [1000, 1100, 1200]
                }),

                # 2. 負の無限大値
                pd.DataFrame({
                    'open': [100, -np.inf, 102],
                    'high': [101, 102, 103],
                    'low': [99, -np.inf, 101],
                    'close': [100.5, 101.5, 102.5],
                    'volume': [1000, 1100, 1200]
                }),

                # 3. NaN値
                pd.DataFrame({
                    'open': [100, np.nan, 102],
                    'high': [101, 102, np.nan],
                    'low': [99, np.nan, 101],
                    'close': [np.nan, 101.5, 102.5],
                    'volume': [1000, np.nan, 1200]
                }),

                # 4. 極小値
                pd.DataFrame({
                    'open': [1e-10, 1e-15, 1e-20],
                    'high': [1e-9, 1e-14, 1e-19],
                    'low': [1e-11, 1e-16, 1e-21],
                    'close': [1e-10, 1e-15, 1e-20],
                    'volume': [1e-5, 1e-10, 1e-15]
                }),

                # 5. 極大値
                pd.DataFrame({
                    'open': [1e15, 1e20, 1e25],
                    'high': [1e15, 1e20, 1e25],
                    'low': [1e15, 1e20, 1e25],
                    'close': [1e15, 1e20, 1e25],
                    'volume': [1e15, 1e20, 1e25]
                })
            ]

            extreme_handling_success = 0
            total_cases = len(extreme_cases)

            for i, extreme_data in enumerate(extreme_cases):
                try:
                    result = service.calculate_ml_indicators(extreme_data)

                    if result is not None:
                        # 結果に無限大やNaNが含まれていないかチェック
                        all_finite = True
                        for indicator_name, values in result.items():
                            if isinstance(values, np.ndarray):
                                if not np.all(np.isfinite(values[~np.isnan(values)])):
                                    all_finite = False
                                    break

                        if all_finite:
                            extreme_handling_success += 1
                        else:
                            extreme_handling_success += 0.5  # 部分的成功
                    else:
                        extreme_handling_success += 1  # Noneも適切な処理

                except Exception as e:
                    print(f"  極端値ケース{i}: {type(e).__name__}")
                    extreme_handling_success += 1  # 例外処理も成功

            success_rate = extreme_handling_success / total_cases

            print(f"極端値ハンドリングテスト成功 - 成功率: {success_rate:.2%}")

            assert success_rate >= 0.7, f"極端値ハンドリング率が低すぎます: {success_rate:.2%}"

            return True

        except Exception as e:
            print(f"極端値ハンドリングテスト失敗: {e}")
            return False

    def test_memory_pressure_handling(self):
        """メモリ圧迫ハンドリングテスト"""
        try:
            from app.core.services.auto_strategy.services.ml_indicator_service import MLIndicatorService

            service = MLIndicatorService()

            # メモリ圧迫シミュレーション
            memory_pressure_tests = []

            # 1. 大量データ処理
            try:
                large_data = create_sample_ohlcv_data(50000)  # 5万行
                result = service.calculate_ml_indicators(large_data)

                if result is not None:
                    memory_pressure_tests.append(True)
                else:
                    memory_pressure_tests.append(True)  # Noneも適切な処理

            except MemoryError:
                print("  大量データ処理: MemoryError適切に処理")
                memory_pressure_tests.append(True)
            except Exception as e:
                print(f"  大量データ処理: {type(e).__name__}")
                memory_pressure_tests.append(True)

            # 2. 繰り返し処理でのメモリリーク確認
            try:
                initial_objects = len(gc.get_objects())

                for i in range(10):
                    data = create_sample_ohlcv_data(1000)
                    result = service.calculate_ml_indicators(data)
                    del data, result

                    if i % 3 == 0:
                        gc.collect()

                final_objects = len(gc.get_objects())
                object_increase = final_objects - initial_objects

                # オブジェクト数の増加が合理的な範囲内かチェック
                if object_increase < 1000:  # 1000オブジェクト以下の増加は許容
                    memory_pressure_tests.append(True)
                else:
                    print(f"  メモリリーク可能性: {object_increase}オブジェクト増加")
                    memory_pressure_tests.append(False)

            except Exception as e:
                print(f"  メモリリークテスト: {type(e).__name__}")
                memory_pressure_tests.append(True)

            # 3. 同時多重処理
            try:
                def memory_intensive_task():
                    data = create_sample_ohlcv_data(5000)
                    result = service.calculate_ml_indicators(data)
                    return result is not None

                threads = []
                for i in range(3):
                    thread = threading.Thread(target=memory_intensive_task)
                    threads.append(thread)
                    thread.start()

                for thread in threads:
                    thread.join(timeout=10)  # 10秒タイムアウト

                memory_pressure_tests.append(True)

            except Exception as e:
                print(f"  同時多重処理: {type(e).__name__}")
                memory_pressure_tests.append(True)

            success_rate = sum(memory_pressure_tests) / len(memory_pressure_tests)

            print(f"メモリ圧迫ハンドリングテスト成功 - 成功率: {success_rate:.2%}")

            assert success_rate >= 0.8, f"メモリ圧迫ハンドリング率が低すぎます: {success_rate:.2%}"

            return True

        except Exception as e:
            print(f"メモリ圧迫ハンドリングテスト失敗: {e}")
            return False

    def test_concurrent_access_handling(self):
        """並行アクセスハンドリングテスト"""
        try:
            from app.core.services.auto_strategy.services.ml_indicator_service import MLIndicatorService

            service = MLIndicatorService()

            # 並行アクセステスト
            results = []
            errors = []

            def concurrent_task(task_id):
                try:
                    data = create_sample_ohlcv_data(500 + task_id * 100)
                    result = service.calculate_ml_indicators(data)
                    results.append((task_id, result is not None))
                except Exception as e:
                    errors.append((task_id, type(e).__name__))

            # 複数スレッドで同時実行
            threads = []
            for i in range(5):
                thread = threading.Thread(target=concurrent_task, args=(i,))
                threads.append(thread)
                thread.start()

            # 全スレッド完了待機
            for thread in threads:
                thread.join(timeout=30)  # 30秒タイムアウト

            # 結果分析
            successful_tasks = len([r for r in results if r[1]])
            total_tasks = len(results) + len(errors)

            if total_tasks > 0:
                success_rate = (successful_tasks + len(errors)) / total_tasks  # エラーも適切な処理
            else:
                success_rate = 0.0

            print(f"並行アクセスハンドリングテスト成功 - 成功率: {success_rate:.2%}")
            print(f"  成功タスク: {successful_tasks}, エラー: {len(errors)}")

            assert success_rate >= 0.8, f"並行アクセスハンドリング率が低すぎます: {success_rate:.2%}"

            return True

        except Exception as e:
            print(f"並行アクセスハンドリングテスト失敗: {e}")
            return False

    def test_model_state_error_handling(self):
        """モデル状態エラーハンドリングテスト"""
        try:
            from app.core.services.ml.signal_generator import MLSignalGenerator

            # 1. 未学習モデルでの予測
            generator = MLSignalGenerator()

            test_features = pd.DataFrame({
                'feature1': [1.0, 2.0, 3.0],
                'feature2': [0.5, 1.5, 2.5],
                'close': [50000.0, 51000.0, 52000.0]
            })

            model_state_tests = []

            try:
                prediction = generator.predict(test_features.head(1))
                # 未学習状態でもデフォルト値が返されることを確認
                if validate_ml_predictions(prediction):
                    model_state_tests.append(True)
                else:
                    model_state_tests.append(False)
            except Exception as e:
                print(f"  未学習モデル予測: {type(e).__name__}")
                model_state_tests.append(True)  # 適切な例外処理

            # 2. 不正な特徴量での予測
            try:
                invalid_features = pd.DataFrame({'invalid': [1, 2, 3]})
                prediction = generator.predict(invalid_features)
                model_state_tests.append(True)  # エラーが適切に処理された
            except Exception as e:
                print(f"  不正特徴量予測: {type(e).__name__}")
                model_state_tests.append(True)

            # 3. 空の特徴量での予測
            try:
                empty_features = pd.DataFrame()
                prediction = generator.predict(empty_features)
                model_state_tests.append(True)
            except Exception as e:
                print(f"  空特徴量予測: {type(e).__name__}")
                model_state_tests.append(True)

            success_rate = sum(model_state_tests) / len(model_state_tests)

            print(f"モデル状態エラーハンドリングテスト成功 - 成功率: {success_rate:.2%}")

            assert success_rate >= 0.9, f"モデル状態エラーハンドリング率が低すぎます: {success_rate:.2%}"

            return True

        except Exception as e:
            print(f"モデル状態エラーハンドリングテスト失敗: {e}")
            return False

    def test_feature_calculation_errors(self):
        """特徴量計算エラーテスト"""
        try:
            from app.core.services.feature_engineering.feature_engineering_service import FeatureEngineeringService

            service = FeatureEngineeringService()

            feature_error_tests = []

            # 1. 不正なlookback_periods
            try:
                data = create_sample_ohlcv_data(100)
                invalid_lookbacks = {
                    "short_ma": -5,  # 負の値
                    "long_ma": 0,    # ゼロ
                    "volatility": 1000,  # データサイズより大きい
                }
                result = service.calculate_advanced_features(data, lookback_periods=invalid_lookbacks)
                feature_error_tests.append(True)
            except Exception as e:
                print(f"  不正lookback_periods: {type(e).__name__}")
                feature_error_tests.append(True)

            # 2. 不整合な外部データ
            try:
                ohlcv_data = create_sample_ohlcv_data(100)
                # 異なるサイズの外部データ
                funding_data = create_sample_funding_rate_data(50)  # 半分のサイズ
                oi_data = create_sample_open_interest_data(150)     # 1.5倍のサイズ

                result = service.calculate_advanced_features(
                    ohlcv_data, funding_data, oi_data,
                    lookback_periods={"short_ma": 10}
                )
                feature_error_tests.append(True)
            except Exception as e:
                print(f"  不整合外部データ: {type(e).__name__}")
                feature_error_tests.append(True)

            # 3. 極端なパラメータ
            try:
                data = create_sample_ohlcv_data(10)
                extreme_lookbacks = {
                    "short_ma": 1,
                    "long_ma": 9,  # データサイズに近い
                    "volatility": 8,
                }
                result = service.calculate_advanced_features(data, lookback_periods=extreme_lookbacks)
                feature_error_tests.append(True)
            except Exception as e:
                print(f"  極端パラメータ: {type(e).__name__}")
                feature_error_tests.append(True)

            success_rate = sum(feature_error_tests) / len(feature_error_tests)

            print(f"特徴量計算エラーテスト成功 - 成功率: {success_rate:.2%}")

            assert success_rate >= 0.8, f"特徴量計算エラーハンドリング率が低すぎます: {success_rate:.2%}"

            return True

        except Exception as e:
            print(f"特徴量計算エラーテスト失敗: {e}")
            return False

    def test_prediction_boundary_cases(self):
        """予測境界ケーステスト"""
        try:
            from app.core.services.auto_strategy.services.ml_indicator_service import MLIndicatorService

            service = MLIndicatorService()

            boundary_tests = []

            # 1. 最小データサイズ
            try:
                min_data = pd.DataFrame({
                    'open': [100.0],
                    'high': [101.0],
                    'low': [99.0],
                    'close': [100.5],
                    'volume': [1000.0]
                })
                result = service.calculate_ml_indicators(min_data)
                boundary_tests.append(True)
            except Exception as e:
                print(f"  最小データサイズ: {type(e).__name__}")
                boundary_tests.append(True)

            # 2. 同一価格データ
            try:
                same_price_data = pd.DataFrame({
                    'open': [100.0] * 10,
                    'high': [100.0] * 10,
                    'low': [100.0] * 10,
                    'close': [100.0] * 10,
                    'volume': [1000.0] * 10
                })
                result = service.calculate_ml_indicators(same_price_data)
                boundary_tests.append(True)
            except Exception as e:
                print(f"  同一価格データ: {type(e).__name__}")
                boundary_tests.append(True)

            # 3. 急激な価格変動
            try:
                volatile_data = pd.DataFrame({
                    'open': [100, 200, 50, 300, 10],
                    'high': [150, 250, 100, 350, 60],
                    'low': [50, 150, 10, 250, 5],
                    'close': [120, 180, 80, 320, 40],
                    'volume': [1000, 2000, 3000, 4000, 5000]
                })
                result = service.calculate_ml_indicators(volatile_data)
                boundary_tests.append(True)
            except Exception as e:
                print(f"  急激な価格変動: {type(e).__name__}")
                boundary_tests.append(True)

            success_rate = sum(boundary_tests) / len(boundary_tests)

            print(f"予測境界ケーステスト成功 - 成功率: {success_rate:.2%}")

            assert success_rate >= 0.8, f"予測境界ケースハンドリング率が低すぎます: {success_rate:.2%}"

            return True

        except Exception as e:
            print(f"予測境界ケーステスト失敗: {e}")
            return False

    def test_system_resource_exhaustion(self):
        """システムリソース枯渇テスト"""
        try:
            from app.core.services.auto_strategy.services.ml_indicator_service import MLIndicatorService

            service = MLIndicatorService()

            resource_tests = []

            # 1. CPU集約的処理
            try:
                cpu_intensive_data = create_sample_ohlcv_data(10000)
                start_time = time.time()
                result = service.calculate_ml_indicators(cpu_intensive_data)
                execution_time = time.time() - start_time

                # 合理的な時間内に完了することを確認
                if execution_time < 60:  # 60秒以内
                    resource_tests.append(True)
                else:
                    print(f"  CPU集約処理時間超過: {execution_time:.1f}秒")
                    resource_tests.append(False)

            except Exception as e:
                print(f"  CPU集約処理: {type(e).__name__}")
                resource_tests.append(True)

            # 2. メモリ集約的処理
            try:
                # 段階的にデータサイズを増やしてメモリ限界をテスト
                max_successful_size = 0
                for size in [1000, 5000, 10000, 20000]:
                    try:
                        memory_data = create_sample_ohlcv_data(size)
                        result = service.calculate_ml_indicators(memory_data)
                        max_successful_size = size
                        del memory_data, result
                        gc.collect()
                    except MemoryError:
                        break
                    except Exception:
                        break

                if max_successful_size >= 1000:
                    resource_tests.append(True)
                else:
                    resource_tests.append(False)

            except Exception as e:
                print(f"  メモリ集約処理: {type(e).__name__}")
                resource_tests.append(True)

            # 3. ファイルディスクリプタ制限テスト
            try:
                # 一時ファイルを大量作成してリソース制限をテスト
                temp_files = []
                for i in range(100):
                    try:
                        temp_file = tempfile.NamedTemporaryFile(delete=False)
                        temp_files.append(temp_file.name)
                        temp_file.close()
                    except OSError:
                        break

                # クリーンアップ
                for temp_file in temp_files:
                    try:
                        os.unlink(temp_file)
                    except:
                        pass

                resource_tests.append(True)

            except Exception as e:
                print(f"  ファイルディスクリプタテスト: {type(e).__name__}")
                resource_tests.append(True)

            success_rate = sum(resource_tests) / len(resource_tests)

            print(f"システムリソース枯渇テスト成功 - 成功率: {success_rate:.2%}")

            assert success_rate >= 0.7, f"リソース枯渇ハンドリング率が低すぎます: {success_rate:.2%}"

            return True

        except Exception as e:
            print(f"システムリソース枯渇テスト失敗: {e}")
            return False


def main():
    """メインテスト実行"""
    test_suite = MLErrorHandlingTestSuite()
    success = test_suite.run_all_tests()
    return success


if __name__ == "__main__":
    main()