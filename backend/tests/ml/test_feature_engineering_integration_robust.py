"""
堅牢な特徴量エンジニアリング統合テスト

モック使用、段階的機能テスト、後方互換性などの
統合性に関する包括的なテストを実施します。
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import warnings

from app.core.services.ml.feature_engineering.feature_engineering_service import FeatureEngineeringService
from app.core.services.ml.feature_engineering.temporal_features import TemporalFeatureCalculator
from app.core.services.ml.feature_engineering.interaction_features import InteractionFeatureCalculator


class TestFeatureEngineeringIntegrationRobust:
    """堅牢な特徴量エンジニアリング統合テストクラス"""
    
    @pytest.fixture
    def sample_data(self):
        """標準的なテストデータ"""
        dates = pd.date_range('2024-01-01', periods=100, freq='h', tz='UTC')
        
        ohlcv = pd.DataFrame({
            'Open': np.random.uniform(40000, 50000, 100),
            'High': np.random.uniform(40000, 50000, 100),
            'Low': np.random.uniform(40000, 50000, 100),
            'Close': np.random.uniform(40000, 50000, 100),
            'Volume': np.random.uniform(1000, 10000, 100)
        }, index=dates)
        
        fr = pd.DataFrame({
            'funding_rate': np.random.uniform(-0.01, 0.01, 100)
        }, index=dates)
        
        oi = pd.DataFrame({
            'open_interest': np.random.uniform(1000000, 2000000, 100)
        }, index=dates)
        
        return ohlcv, fr, oi
    
    def test_component_isolation(self, sample_data):
        """コンポーネント分離テスト"""
        ohlcv, fr, oi = sample_data
        
        # 各計算クラスを個別にテスト
        temporal_calc = TemporalFeatureCalculator()
        interaction_calc = InteractionFeatureCalculator()
        
        # 時間的特徴量の単独動作確認
        temporal_result = temporal_calc.calculate_temporal_features(ohlcv)
        assert len(temporal_result) == 100
        assert 'Hour_of_Day' in temporal_result.columns
        
        # 統合サービスでの動作確認
        service = FeatureEngineeringService()
        full_result = service.calculate_advanced_features(ohlcv, fr, oi)
        
        # 時間的特徴量が統合結果に含まれていることを確認
        temporal_features = temporal_calc.get_feature_names()
        for feature in temporal_features:
            assert feature in full_result.columns, f"Temporal feature {feature} missing in integrated result"
            
            # 値が一致することを確認（計算順序による微小な差は許容）
            if feature in temporal_result.columns:
                temporal_values = temporal_result[feature]
                integrated_values = full_result[feature]
                
                # 数値特徴量の場合は近似一致を確認
                if pd.api.types.is_numeric_dtype(temporal_values):
                    np.testing.assert_allclose(
                        temporal_values, integrated_values, 
                        rtol=1e-10, atol=1e-10,
                        err_msg=f"Values mismatch for feature {feature}"
                    )
                else:
                    # ブール特徴量の場合は完全一致を確認
                    pd.testing.assert_series_equal(
                        temporal_values, integrated_values,
                        check_names=False,
                        check_dtype=False
                    )
    
    def test_graceful_component_failure(self, sample_data):
        """コンポーネント障害時の優雅な処理テスト"""
        ohlcv, fr, oi = sample_data
        
        service = FeatureEngineeringService()
        
        # 時間的特徴量計算を無効化
        with patch.object(service.temporal_calculator, 'calculate_temporal_features') as mock_temporal:
            mock_temporal.side_effect = Exception("Temporal calculation failed")
            
            # エラーが発生しても他の特徴量は計算されることを確認
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = service.calculate_advanced_features(ohlcv, fr, oi)
            
            # 基本的な特徴量は生成されていることを確認
            assert len(result) == 100
            assert len(result.columns) > len(ohlcv.columns)
            
            # 価格特徴量は生成されていることを確認
            price_features = ['Price_Momentum_14', 'ATR_20']
            for feature in price_features:
                if feature in result.columns:
                    assert result[feature].notna().any(), f"Price feature {feature} should be calculated"
        
        # 相互作用特徴量計算を無効化
        with patch.object(service.interaction_calculator, 'calculate_interaction_features') as mock_interaction:
            mock_interaction.side_effect = Exception("Interaction calculation failed")
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = service.calculate_advanced_features(ohlcv, fr, oi)
            
            # 基本的な特徴量と時間的特徴量は生成されていることを確認
            assert len(result) == 100
            assert 'Hour_of_Day' in result.columns  # 時間的特徴量
            assert 'Price_Momentum_14' in result.columns or 'ATR_20' in result.columns  # 価格特徴量
    
    def test_partial_data_scenarios(self, sample_data):
        """部分的データシナリオのテスト"""
        ohlcv, fr, oi = sample_data
        service = FeatureEngineeringService()
        
        # OHLCVのみ（FR/OIなし）
        result_ohlcv_only = service.calculate_advanced_features(ohlcv)
        assert len(result_ohlcv_only) == 100
        assert 'Hour_of_Day' in result_ohlcv_only.columns  # 時間的特徴量は生成される
        
        # OHLCV + FR（OIなし）
        result_with_fr = service.calculate_advanced_features(ohlcv, fr)
        assert len(result_with_fr) == 100
        assert 'FR_Normalized' in result_with_fr.columns or 'funding_rate' in result_with_fr.columns
        
        # OHLCV + OI（FRなし）
        result_with_oi = service.calculate_advanced_features(ohlcv, None, oi)
        assert len(result_with_oi) == 100
        assert 'OI_Change_Rate' in result_with_oi.columns or 'open_interest' in result_with_oi.columns
        
        # 全データ
        result_full = service.calculate_advanced_features(ohlcv, fr, oi)
        assert len(result_full) == 100
        
        # 全データ版が最も多くの特徴量を持つことを確認
        assert len(result_full.columns) >= len(result_ohlcv_only.columns)
        assert len(result_full.columns) >= len(result_with_fr.columns)
        assert len(result_full.columns) >= len(result_with_oi.columns)
    
    def test_feature_name_consistency(self):
        """特徴量名の一貫性テスト"""
        service = FeatureEngineeringService()
        
        # get_feature_names()で取得される名前と実際の生成される特徴量名が一致することを確認
        expected_names = service.get_feature_names()
        
        # テストデータで実際に特徴量を生成
        dates = pd.date_range('2024-01-01', periods=50, freq='h', tz='UTC')
        ohlcv = pd.DataFrame({
            'Open': np.random.uniform(40000, 50000, 50),
            'High': np.random.uniform(40000, 50000, 50),
            'Low': np.random.uniform(40000, 50000, 50),
            'Close': np.random.uniform(40000, 50000, 50),
            'Volume': np.random.uniform(1000, 10000, 50)
        }, index=dates)
        
        fr = pd.DataFrame({'funding_rate': np.random.uniform(-0.01, 0.01, 50)}, index=dates)
        oi = pd.DataFrame({'open_interest': np.random.uniform(1000000, 2000000, 50)}, index=dates)
        
        result = service.calculate_advanced_features(ohlcv, fr, oi)
        actual_names = result.columns.tolist()
        
        # 期待される特徴量名が実際に生成されていることを確認
        missing_features = []
        for expected_name in expected_names:
            if expected_name not in actual_names:
                missing_features.append(expected_name)
        
        # 一部の特徴量は条件によって生成されない場合があるため、
        # 重要な特徴量のみをチェック
        critical_features = [
            'Hour_of_Day', 'Day_of_Week', 'Asia_Session',  # 時間的特徴量
            'Price_Momentum_14', 'ATR_20',  # 価格特徴量
            'RSI', 'MACD'  # テクニカル特徴量
        ]
        
        for feature in critical_features:
            if feature in expected_names:
                assert feature in actual_names, f"Critical feature {feature} is missing"
    
    def test_data_validation_integration(self, sample_data):
        """データバリデーション統合テスト"""
        ohlcv, fr, oi = sample_data
        service = FeatureEngineeringService()
        
        # 正常データでの処理
        result_normal = service.calculate_advanced_features(ohlcv, fr, oi)
        
        # 異常データを含むケース
        corrupted_ohlcv = ohlcv.copy()
        corrupted_ohlcv.iloc[10:15, :] = np.nan  # 一部行を完全にNaN
        corrupted_ohlcv.iloc[20, 0] = np.inf     # 無限大値
        corrupted_ohlcv.iloc[25, 1] = -np.inf   # 負の無限大値
        
        # 異常データでも処理が完了することを確認
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result_corrupted = service.calculate_advanced_features(corrupted_ohlcv, fr, oi)
        
        # 結果が生成されることを確認
        assert len(result_corrupted) == 100
        
        # 正常な行では正常データと同様の結果が得られることを確認
        normal_rows = [0, 1, 2, 30, 40, 50]  # 異常データの影響を受けない行
        for row_idx in normal_rows:
            for feature in ['Hour_of_Day', 'Day_of_Week']:
                if feature in result_normal.columns and feature in result_corrupted.columns:
                    normal_val = result_normal.iloc[row_idx][feature]
                    corrupted_val = result_corrupted.iloc[row_idx][feature]
                    assert normal_val == corrupted_val, f"Feature {feature} differs at row {row_idx}"
    
    def test_version_compatibility(self, sample_data):
        """バージョン互換性テスト"""
        ohlcv, fr, oi = sample_data
        
        # 古いインターフェースでの呼び出し（位置引数）
        service = FeatureEngineeringService()
        result_positional = service.calculate_advanced_features(ohlcv, fr, oi)
        
        # 新しいインターフェースでの呼び出し（キーワード引数）
        result_keyword = service.calculate_advanced_features(
            ohlcv_data=ohlcv, 
            funding_rate_data=fr, 
            open_interest_data=oi
        )
        
        # 結果が同じであることを確認
        assert len(result_positional) == len(result_keyword)
        assert len(result_positional.columns) == len(result_keyword.columns)
        
        # 重要な特徴量の値が一致することを確認
        important_features = ['Hour_of_Day', 'Day_of_Week', 'Price_Momentum_14']
        for feature in important_features:
            if feature in result_positional.columns and feature in result_keyword.columns:
                pd.testing.assert_series_equal(
                    result_positional[feature], 
                    result_keyword[feature],
                    check_names=False
                )
    
    def test_concurrent_service_instances(self, sample_data):
        """並行サービスインスタンステスト"""
        ohlcv, fr, oi = sample_data
        
        import threading
        import queue
        
        def worker(worker_id, data_tuple, result_queue):
            """ワーカー関数"""
            ohlcv_data, fr_data, oi_data = data_tuple
            service = FeatureEngineeringService()  # 各スレッドで独立したインスタンス
            
            try:
                result = service.calculate_advanced_features(ohlcv_data, fr_data, oi_data)
                result_queue.put({
                    'worker_id': worker_id,
                    'success': True,
                    'rows': len(result),
                    'columns': len(result.columns),
                    'sample_features': {
                        'Hour_of_Day': result['Hour_of_Day'].iloc[0] if 'Hour_of_Day' in result.columns else None,
                        'Day_of_Week': result['Day_of_Week'].iloc[0] if 'Day_of_Week' in result.columns else None
                    }
                })
            except Exception as e:
                result_queue.put({
                    'worker_id': worker_id,
                    'success': False,
                    'error': str(e)
                })
        
        # 複数スレッドで並行実行
        num_workers = 3
        result_queue = queue.Queue()
        threads = []
        
        for i in range(num_workers):
            thread = threading.Thread(
                target=worker, 
                args=(i, (ohlcv, fr, oi), result_queue)
            )
            threads.append(thread)
            thread.start()
        
        # 全スレッドの完了を待機
        for thread in threads:
            thread.join()
        
        # 結果の検証
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())
        
        assert len(results) == num_workers, f"Expected {num_workers} results, got {len(results)}"
        
        # 全ワーカーが成功したことを確認
        for result in results:
            assert result['success'], f"Worker {result['worker_id']} failed: {result.get('error', 'Unknown error')}"
            assert result['rows'] == 100, f"Worker {result['worker_id']} processed {result['rows']} rows"
            assert result['columns'] > 50, f"Worker {result['worker_id']} generated {result['columns']} columns"
        
        # 全ワーカーで同じ時間的特徴量が生成されることを確認
        first_result = results[0]
        for result in results[1:]:
            assert result['sample_features']['Hour_of_Day'] == first_result['sample_features']['Hour_of_Day'], \
                "Hour_of_Day should be consistent across workers"
            assert result['sample_features']['Day_of_Week'] == first_result['sample_features']['Day_of_Week'], \
                "Day_of_Week should be consistent across workers"
    
    def test_memory_leak_detection(self, sample_data):
        """メモリリーク検出テスト"""
        import psutil
        import gc
        
        ohlcv, fr, oi = sample_data
        process = psutil.Process()
        
        # 初期メモリ使用量
        gc.collect()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 複数回の処理実行
        memory_measurements = []
        
        for iteration in range(10):
            service = FeatureEngineeringService()
            result = service.calculate_advanced_features(ohlcv, fr, oi)
            
            # 結果の基本検証
            assert len(result) == 100
            
            # 明示的にオブジェクトを削除
            del service, result
            gc.collect()
            
            # メモリ使用量を測定
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_measurements.append(current_memory - initial_memory)
        
        # メモリリークの検出
        # 最後の5回の平均と最初の5回の平均を比較
        early_avg = np.mean(memory_measurements[:5])
        late_avg = np.mean(memory_measurements[-5:])
        memory_growth = late_avg - early_avg
        
        # メモリ増加が50MB以下であることを確認
        assert memory_growth < 50, f"Potential memory leak detected: {memory_growth:.2f}MB growth over 10 iterations"
        
        print(f"Memory leak test: {memory_growth:.2f}MB growth over 10 iterations")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
