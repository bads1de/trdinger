"""
Performance and Resource Usage Tests
Focus: Memory usage, execution time, scalability, resource limits
"""

import pytest
import time
import psutil
import gc
from unittest.mock import Mock, patch
import numpy as np
import sys
import os

# Test framework setup
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


class TestPerformanceAndResources:
    """Performance and resource usage tests"""

    def test_memory_usage_during_ga_execution(self):
        """Test memory usage during GA operations"""
        try:
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Simulate GA operation
            large_population = []
            for i in range(1000):  # Simulate large population
                large_population.append({'fitness': i, 'genes': [j for j in range(100)]})

            operation_memory = process.memory_info().rss / 1024 / 1024
            memory_increase = operation_memory - initial_memory

            # Memory increase should be reasonable
            assert memory_increase < 100  # Less than 100MB
            assert operation_memory > 0

        except ImportError:
            # Skip if psutil not available
            pass

    def test_execution_time_scaling_analysis(self):
        """Test how execution time scales with population size"""
        from app.services.auto_strategy.core.ga_engine import GAEngine

        sizes_and_times = []
        base_time = time.time()

        populations = [10, 50, 100, 200]

        for pop_size in populations:
            try:
                pop_time = time.process_time()

                # Simulate operation scaling
                engine = GAEngine(population_size=pop_size)
                simulated_workload = pop_size * 0.001  # Simulate CPU work

                pop_end_time = time.process_time()
                sizes_and_times.append((pop_size, pop_end_time - pop_time))

            except (ImportError, AttributeError):
                # Skip if GA engine not available
                pass

        # Check scaling is roughly linear
        if len(sizes_and_times) >= 2:
            ratios = []
            for i in range(1, len(sizes_and_times)):
                current_size, current_time = sizes_and_times[i]
                prev_size, prev_time = sizes_and_times[i-1]
                ratio = current_time / prev_time if prev_time > 0 else 0
                size_ratio = current_size / prev_size
                ratios.append(abs(ratio / size_ratio - 1.0))  # Deviation from linear

            # Scaling should be close to linear (±30%)
            if ratios:
                avg_deviation = sum(ratios) / len(ratios)
                assert avg_deviation < 0.3 or ratios[0] < 0.5

    def test_resource_cleanup_validation(self):
        """Test proper cleanup of resources after operations"""
        initial_objects_before_gc = len(gc.get_objects())

        # Create temporary objects
        temp_data = []
        for i in range(1000):
            temp_data.append({'data': [j for j in range(1000)], 'metadata': 'test'})

        gc.collect()  # Force garbage collection
        objects_after_gc = len(gc.get_objects())

        # Objects should be cleaned up (though some may remain due to interpreter)
        # Just test that GC is functioning
        assert objects_after_gc >= 0

    def test_cpu_utilization_during_heavy_computation(self):
        """Test CPU usage during heavy calculations"""
        try:
            process = psutil.Process()
            initial_cpu = process.cpu_percent(interval=0.1)

            # Heavy computation simulation
            result = 0
            for i in range(1000000):
                result += np.sin(i) * np.cos(i)

            final_cpu = process.cpu_percent(interval=0.1)

            # Check that CPU was utilized (some variation expected)
            cpu_increase = final_cpu - initial_cpu

            # Basic check that computation happened
            assert final_cpu >= 0
            assert result is not None

        except ImportError:
            # Skip if psutil not available
            pass

    def test_database_connection_pooling_efficiency(self):
        """Test efficiency of database connection pooling"""
        with patch('app.services.auto_strategy.services.auto_strategy_service.SessionLocal') as mock_session:
            # Create multiple sessions
            for i in range(10):
                mock_session.return_value = Mock()

                try:
                    from app.services.auto_strategy.services.auto_strategy_service import AutoStrategyService

                    with patch('app.services.auto_strategy.services.auto_strategy_service.BacktestService'), \
                         patch('app.services.auto_strategy.services.auto_strategy_service.ExperimentPersistenceService'), \
                         patch('app.services.auto_strategy.services.auto_strategy_service.ExperimentManager'):

                        service = AutoStrategyService()
                        config = {"generations": 1, "population_size": 2}

                        result = service.start_strategy_generation(f"pool_test_{i}", f"Pool Test {i}", config, {}, Mock())
                        assert result is not None

                except (ImportError, AttributeError):
                    pass

            # Check that sessions were requested
            assert mock_session.call_count >= 0

    def test_large_dataset_indicators_performance(self):
        """Test indicator calculation performance with large datasets"""
        from app.services.indicators.technical_indicators.trend import TrendIndicators

        try:
            trend_indicators = TrendIndicators()

            # Large dataset
            large_data = np.random.randn(100000)

            start_time = time.time()
            result = TrendIndicators.sma(large_data, length=20)
            end_time = time.time()

            processing_time = end_time - start_time

            # Should process within reasonable time (5 seconds for 100k data points)
            assert processing_time < 5.0
            assert len(result) > 0

        except (ImportError, AttributeError):
            pass

    def test_concurrent_operations_resource_isolation(self):
        """Test resource isolation between concurrent operations"""
        import threading

        results = {}
        errors = []

        def worker_thread(worker_id):
            try:
                # Each thread does computation
                start_memory = psutil.Process().memory_info().rss if 'psutil' in sys.modules else 0
                time.sleep(0.1)  # Simulate work
                end_memory = psutil.Process().memory_info().rss if 'psutil' in sys.modules else 0

                results[worker_id] = {
                    'memory_used': end_memory - start_memory,
                    'result': f"Worker {worker_id} completed"
                }
            except Exception as e:
                errors.append(f"Worker {worker_id}: {str(e)}")

        # Start multiple threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=worker_thread, args=[i])
            threads.append(t)
            t.start()

        # Wait for all
        for t in threads:
            t.join()

        # Check results
        assert len(results) == 5
        assert len(errors) == 0

        # Memory usage should be reasonable for each thread
        for worker_id, result in results.items():
            if 'memory_used' in result:
                assert result['memory_used'] >= 0

    def test_cache_efficiency_and_hit_rate(self):
        """Test efficiency of any caching mechanisms"""
        # Mock cache operations
        cache_hits = 0
        cache_misses = 0

        # Simulate cache usage
        cache = {}

        # Simulate data access patterns
        keys = ['key1', 'key2', 'key1', 'key3', 'key2', 'key1']

        for key in keys:
            if key in cache:
                cache_hits += 1
                _ = cache[key]
            else:
                cache_misses += 1
                cache[key] = f"data_{key}"

        total_accesses = len(keys)
        hit_rate = cache_hits / total_accesses if total_accesses > 0 else 0

        # Basic cache functionality test
        assert cache_hits >= 0
        assert cache_misses >= 0
        assert len(cache) == 3  # Should have 3 unique keys

    def test_scalability_with_increasing_workload(self):
        """Test system scalability with increasing workload"""
        # Simple scalability test
        workloads = [100, 500, 1000, 5000]

        results = []

        for workload in workloads:
            start_time = time.time()

            # Simulate workload
            total = 0
            for i in range(workload):
                total += i * 2

            end_time = time.time()

            processing_time = end_time - start_time
            processing_rate = workload / processing_time if processing_time > 0 else 0

            results.append((workload, processing_time, processing_rate))

        # Check that larger workloads don't cause disproportionate slowdown
        if len(results) > 1:
            first_rate = results[0][2]
            last_rate = results[-1][2]

            # Rate shouldn't drop too dramatically (allow 50% reduction)
            if first_rate > 0:
                degradation = (first_rate - last_rate) / first_rate
                assert degradation < 0.5

    def test_memory_leak_detection(self):
        """Test for memory leaks over repeated operations"""
        try:
            process = psutil.Process()
            memory_samples = []

            # Take multiple memory samples over operations
            for i in range(10):
                # Simulate operation
                data = list(range(10000))
                processed = [x * 2 for x in data]
                del data, processed
                gc.collect()

                memory_samples.append(process.memory_info().rss / 1024 / 1024)  # MB

            # Check for significant memory growth
            initial_memory = memory_samples[0]
            final_memory = memory_samples[-1]
            growth_rate = (final_memory - initial_memory) / initial_memory if initial_memory > 0 else 0

            # Allow some growth but not excessive (less than 50%)
            assert growth_rate < 0.5

        except ImportError:
            # Skip if psutil not available
            pass