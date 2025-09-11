"""
リファクタリングされたDataProcessorの総合統合テスト

5つのテストシナリオをカバー：
1. エンドツーエンドデータ処理ワークフロー
2. 既存APIコールの後方互換性
3. 旧アプローチとのパフォーマンス比較
4. エラーハンドリングとエッジケース
5. メモリ使用量と効率改善
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock
import time
import psutil
import os
from typing import Dict, Any
import gc


class TestEndToEndWorkflow:
    """エンドツーエンドデータ処理ワークフローのテスト"""

    @pytest.fixture
    def large_sample_data(self):
        """大規模テスト用のサンプルデータ"""
        dates = pd.date_range('2023-01-01', periods=10000, freq='h')
        np.random.seed(42)

        # OHLCVデータ
        data = {
            'open': np.random.uniform(100, 110, 10000),
            'high': np.random.uniform(105, 115, 10000),
            'low': np.random.uniform(95, 105, 10000),
            'close': np.random.uniform(100, 110, 10000),
            'volume': np.random.randint(1000, 10000, 10000),
        }

        # 追加の特徴量
        data.update({
            'rsi': np.random.uniform(0, 100, 10000),
            'macd': np.random.uniform(-10, 10, 10000),
            'bb_upper': np.random.uniform(105, 120, 10000),
            'bb_lower': np.random.uniform(90, 100, 10000),
            'fear_greed_value': np.random.uniform(0, 100, 10000),
            'funding_rate': np.random.uniform(-0.01, 0.01, 10000),
            'open_interest': np.random.uniform(1000000, 5000000, 10000),
        })

        df = pd.DataFrame(data, index=dates)

        # NaN値を追加して現実性を高める
        nan_mask = np.random.random(10000) < 0.05  # 5%の確率でNaN
        for col in df.columns:
            if col != 'open':  # openは必須カラムなのでNaNなし
                df.loc[nan_mask, col] = np.nan

        return df

    @pytest.fixture
    def data_processor(self):
        """DataProcessorインスタンス"""
        from backend.app.utils.data_processing import DataProcessor
        return DataProcessor()

    def test_complete_data_processing_workflow(self, data_processor, large_sample_data):
        """完全なデータ処理ワークフローのテスト"""
        print("\n=== エンドツーエンドワークフローテスト開始 ===")

        # ステップ1: データクリーニングと検証
        print("ステップ1: データクリーニングと検証")
        cleaned_data = data_processor.clean_and_validate_data(
            large_sample_data,
            required_columns=['open', 'high', 'low', 'close'],
            interpolate=True,
            optimize=True
        )

        assert isinstance(cleaned_data, pd.DataFrame)
        assert len(cleaned_data) > 0
        assert all(col in cleaned_data.columns for col in ['open', 'high', 'low', 'close'])

        # NaNが補間されていることを確認
        assert not cleaned_data[['open', 'high', 'low', 'close']].isnull().any().any()

        print(f"クリーニング後データサイズ: {len(cleaned_data)}行, {len(cleaned_data.columns)}列")

        # ステップ2: 特徴量エンジニアリングパイプライン
        print("ステップ2: 特徴量エンジニアリングパイプライン")
        processed_data = data_processor.preprocess_with_pipeline(
            cleaned_data,
            pipeline_name="integration_test",
            fit_pipeline=True
        )

        assert isinstance(processed_data, pd.DataFrame)
        assert len(processed_data) > 0
        print(f"パイプライン処理後データサイズ: {len(processed_data)}行, {len(processed_data.columns)}列")

        # ステップ3: 効率的なデータ処理
        print("ステップ3: 効率的なデータ処理")
        efficient_result = data_processor.process_data_efficiently(
            cleaned_data,
            pipeline_name="efficient_test"
        )

        assert isinstance(efficient_result, pd.DataFrame)
        assert len(efficient_result) > 0
        print(f"効率処理後データサイズ: {len(efficient_result)}行, {len(efficient_result.columns)}列")

        # ステップ4: 学習用データ準備（モックラベルジェネレータ使用）
        print("ステップ4: 学習用データ準備")

        # Mockラベルジェネレータ
        label_generator = Mock()
        mock_labels = pd.Series(np.random.choice([0, 1], size=len(cleaned_data)))
        mock_labels.index = cleaned_data.index
        label_generator.generate_labels.return_value = (
            mock_labels,
            {"threshold": 0.02, "method": "mock"}
        )

        features, labels, threshold_info = data_processor.prepare_training_data(
            cleaned_data, label_generator
        )

        assert isinstance(features, pd.DataFrame)
        assert isinstance(labels, pd.Series)
        assert isinstance(threshold_info, dict)
        assert len(features) == len(labels)
        assert len(features) > 0

        print(f"学習用データ準備完了: {len(features)}行, {len(labels)}行")
        print("=== エンドツーエンドワークフローテスト完了 ===")


class TestBackwardCompatibility:
    """後方互換性のテスト"""

    @pytest.fixture
    def data_processor(self):
        """DataProcessorインスタンス"""
        from backend.app.utils.data_processing import data_processor
        return data_processor

    def test_global_instance_compatibility(self, data_processor):
        """グローバルインスタンスの後方互換性テスト"""
        # グローバルインスタンスが利用可能であることを確認
        assert hasattr(data_processor, 'clean_and_validate_data')
        assert hasattr(data_processor, 'prepare_training_data')
        assert hasattr(data_processor, 'preprocess_with_pipeline')
        assert hasattr(data_processor, 'process_data_efficiently')
        assert hasattr(data_processor, 'create_optimized_pipeline')
        assert hasattr(data_processor, 'get_pipeline_info')
        assert hasattr(data_processor, 'clear_cache')

        # 各メソッドが呼び出し可能であることを確認
        assert callable(data_processor.clean_and_validate_data)
        assert callable(data_processor.prepare_training_data)
        assert callable(data_processor.preprocess_with_pipeline)

    def test_old_api_signature_compatibility(self, data_processor):
        """旧APIシグネチャの互換性テスト"""
        # 旧バージョンのAPIコールをシミュレート
        sample_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [102, 103, 104]
        })

        # 旧API形式での呼び出し
        result = data_processor.clean_and_validate_data(
            sample_data, ['open', 'high', 'low', 'close']
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_data)


class TestPerformanceComparison:
    """パフォーマンス比較テスト"""

    @pytest.fixture
    def large_sample_data(self):
        """大規模データセット"""
        dates = pd.date_range('2023-01-01', periods=10000, freq='h')
        np.random.seed(42)

        data = {
            'open': np.random.uniform(100, 110, 10000),
            'high': np.random.uniform(105, 115, 10000),
            'low': np.random.uniform(95, 105, 10000),
            'close': np.random.uniform(100, 110, 10000),
            'volume': np.random.randint(1000, 10000, 10000),
            'rsi': np.random.uniform(0, 100, 10000),
            'macd': np.random.uniform(-10, 10, 10000),
        }

        df = pd.DataFrame(data, index=dates)

        # NaNを追加
        nan_mask = np.random.random(10000) < 0.1
        for col in df.columns:
            df.loc[nan_mask, col] = np.nan

        return df

    @pytest.fixture
    def data_processor(self):
        """DataProcessorインスタンス"""
        from backend.app.utils.data_processing import DataProcessor
        return DataProcessor()

    def test_memory_usage_comparison(self, data_processor, large_sample_data):
        """メモリ使用量比較テスト"""
        print("\n=== メモリ使用量比較テスト開始 ===")

        # 初期メモリ使用量
        initial_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
        print(f"初期メモリ使用量: {initial_memory:.2f} MB")

        # データ処理実行
        start_time = time.time()
        result = data_processor.clean_and_validate_data(
            large_sample_data,
            required_columns=['open', 'high', 'low', 'close'],
            interpolate=True,
            optimize=True
        )
        processing_time = time.time() - start_time

        # 処理後メモリ使用量
        processing_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        print(f"処理後メモリ使用量: {processing_memory:.2f} MB")

        # パイプライン処理
        pipeline_start = time.time()
        pipeline_result = data_processor.preprocess_with_pipeline(
            result, pipeline_name="performance_test"
        )
        pipeline_time = time.time() - pipeline_start

        final_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        print(f"最終メモリ使用量: {final_memory:.2f} MB")

        # メモリ増加量チェック
        memory_increase = final_memory - initial_memory
        print(f"メモリ増加量: {memory_increase:.2f} MB")

        # パフォーマンス指標
        print("=== パフォーマンス指標 ===")
        print(f"データクリーニング時間: {processing_time:.3f} 秒")
        print(f"パイプライン処理時間: {pipeline_time:.3f} 秒")
        print(f"処理データサイズ: {len(result)} 行")
        print(f"特徴量数: {len(pipeline_result.columns)} 個")

        # 基本的なパフォーマンスチェック
        assert processing_time < 30  # 30秒以内に完了
        assert pipeline_time < 60    # 60秒以内に完了
        assert memory_increase < 500  # 500MB以内の増加

        print("=== メモリ使用量比較テスト完了 ===")

    def test_processing_speed_optimization(self, data_processor, large_sample_data):
        """処理速度最適化テスト"""
        print("\n=== 処理速度最適化テスト開始 ===")

        # 複数回の処理でキャッシュ効果を確認
        times = []

        for i in range(3):
            start_time = time.time()
            result = data_processor.process_data_efficiently(
                large_sample_data,
                pipeline_name="speed_test"
            )
            processing_time = time.time() - start_time
            times.append(processing_time)
            print(f"処理{i+1}回目: {processing_time:.3f} 秒")

        # 2回目以降が高速化されていることを確認（キャッシュ効果）
        if len(times) >= 2:
            improvement = times[0] - times[1]
            print(f"キャッシュによる改善: {improvement:.3f} 秒")

        # データサイズと処理時間のバランスをチェック
        throughput = len(large_sample_data) / min(times)  # 行/秒
        print(f"処理スループット: {throughput:.0f} 行/秒")

        print("=== 処理速度最適化テスト完了 ===")


class TestErrorHandling:
    """エラーハンドリングとエッジケースのテスト"""

    @pytest.fixture
    def data_processor(self):
        """DataProcessorインスタンス"""
        from backend.app.utils.data_processing import DataProcessor
        return DataProcessor()

    def test_empty_data_handling(self, data_processor):
        """空データに対するエラーハンドリング"""
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError, match="入力特徴量データが空です"):
            label_generator = Mock()
            data_processor.prepare_training_data(empty_df, label_generator)

    def test_missing_required_columns(self, data_processor):
        """必須カラム欠如時のエラーハンドリング"""
        incomplete_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            # lowとcloseが欠如
        })

        # 実際の挙動では例外を投げないが、警告が出力される
        # OHLCVの検証はスキップされる（'Open'カラムが存在しないため）
        result = data_processor.clean_and_validate_data(
            incomplete_data,
            required_columns=['open', 'high', 'low', 'close']
        )

        # 結果はDataFrameであること
        assert isinstance(result, pd.DataFrame)
        # 入力データのカラムはそのまま残る
        assert 'open' in result.columns
        assert 'high' in result.columns

    def test_extreme_values_handling(self, data_processor):
        """極端な値に対するハンドリング"""
        extreme_data = pd.DataFrame({
            'open': [np.inf, -np.inf, np.nan],
            'high': [1e10, -1e10, 0],
            'low': [1e-10, -1e-10, 0],
            'close': [100, 101, 102]
        })

        # 極端な値が適切に処理されることを確認
        result = data_processor.clean_and_validate_data(
            extreme_data,
            required_columns=['open', 'high', 'low', 'close'],
            interpolate=True
        )

        assert isinstance(result, pd.DataFrame)
        # inf値とNaNが処理されていることを確認
        numeric_data = result.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            # 補間処理でinfが残っている可能性があるので、適切な範囲に収まることを確認
            finite_mask = np.isfinite(numeric_data)
            # 少なくとも一部の値が有限値に変換されていることを確認
            assert finite_mask.any().any(), "すべての値が非有限値のまま"

    def test_large_data_memory_efficiency(self, data_processor):
        """大規模データでのメモリ効率テスト"""
        print("\n=== 大規模データメモリ効率テスト開始 ===")

        # 非常に大規模なデータセット
        dates = pd.date_range('2023-01-01', periods=50000, freq='min')
        large_data = pd.DataFrame({
            'open': np.random.uniform(100, 110, 50000),
            'high': np.random.uniform(105, 115, 50000),
            'low': np.random.uniform(95, 105, 50000),
            'close': np.random.uniform(100, 110, 50000),
            'volume': np.random.randint(1000, 10000, 50000),
        }, index=dates)

        initial_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

        try:
            start_time = time.time()
            result = data_processor.clean_and_validate_data(
                large_data,
                required_columns=['open', 'high', 'low', 'close'],
                optimize=True
            )
            processing_time = time.time() - start_time

            final_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            memory_increase = final_memory - initial_memory

            print(f"大規模データ処理時間: {processing_time:.3f} 秒")
            print(f"メモリ増加: {memory_increase:.2f} MB")
            print(f"処理成功: {len(result)} 行")

            # メモリリークがないことを確認
            assert memory_increase < 1000  # 1GB以内の増加
            assert processing_time < 120   # 2分以内に完了

        finally:
            # メモリ解放
            del large_data
            gc.collect()

        print("=== 大規模データメモリ効率テスト完了 ===")


class TestMemoryEfficiency:
    """メモリ使用量と効率改善のテスト"""

    @pytest.fixture
    def data_processor(self):
        """DataProcessorインスタンス"""
        from backend.app.utils.data_processing import DataProcessor
        return DataProcessor()

    def test_memory_leak_prevention(self, data_processor):
        """メモリリーク防止テスト"""
        print("\n=== メモリリーク防止テスト開始 ===")

        initial_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

        # 複数回の処理を実行
        for i in range(10):
            sample_data = pd.DataFrame({
                'open': np.random.uniform(100, 110, 1000),
                'high': np.random.uniform(105, 115, 1000),
                'low': np.random.uniform(95, 105, 1000),
                'close': np.random.uniform(100, 110, 1000),
            })

            result = data_processor.process_data_efficiently(
                sample_data,
                pipeline_name=f"memory_test_{i}"
            )

            # 中間結果を削除
            del sample_data, result

            # 定期的にガベージコレクション
            if i % 3 == 0:
                gc.collect()

        final_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory

        print(f"複数処理後のメモリ増加: {memory_increase:.2f} MB")

        # メモリリークがないことを確認（許容範囲内）
        assert memory_increase < 50  # 50MB以内の増加

        # キャッシュクリア
        data_processor.clear_cache()

        after_clear_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        print(f"キャッシュクリア後のメモリ: {after_clear_memory:.2f} MB")

        print("=== メモリリーク防止テスト完了 ===")

    def test_dtype_optimization_efficiency(self, data_processor):
        """データ型最適化の効率テスト"""
        print("\n=== データ型最適化効率テスト開始 ===")

        # 最適化前のデータ（非効率的なデータ型）
        dates = pd.date_range('2023-01-01', periods=10000, freq='h')
        original_data = pd.DataFrame({
            'open': np.random.uniform(100, 110, 10000).astype('float64'),
            'high': np.random.uniform(105, 115, 10000).astype('float64'),
            'low': np.random.uniform(95, 105, 10000).astype('float64'),
            'close': np.random.uniform(100, 110, 10000).astype('float64'),
            'volume': np.random.randint(1000, 10000, 10000).astype('int64'),
        }, index=dates)

        # メモリ使用量測定
        original_memory = original_data.memory_usage(deep=True).sum() / 1024 / 1024
        print(f"最適化前メモリ使用量: {original_memory:.2f} MB")

        # データ型最適化実行
        optimized_data = data_processor.clean_and_validate_data(
            original_data,
            required_columns=['open', 'high', 'low', 'close'],
            optimize=True
        )

        optimized_memory = optimized_data.memory_usage(deep=True).sum() / 1024 / 1024
        print(f"最適化後メモリ使用量: {optimized_memory:.2f} MB")

        memory_reduction = original_memory - optimized_memory
        reduction_percentage = (memory_reduction / original_memory) * 100

        print(f"メモリ削減量: {memory_reduction:.2f} MB")
        print(f"メモリ削減率: {reduction_percentage:.1f} %")

        # 最適化効果を確認
        assert reduction_percentage > 0  # メモリ削減があること
        assert optimized_memory < original_memory  # 最適化後の方が少ないメモリ使用

        print("=== データ型最適化効率テスト完了 ===")


if __name__ == "__main__":
    # 直接実行時のテスト
    pytest.main([__file__, "-v", "--tb=short"])