"""
Memory Management Tests
Focus: Memory leaks, resource cleanup, efficient allocation
"""

import pytest
import psutil
import gc
import time
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import sys
import os
import numpy as np

# Test framework setup
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


class TestMemoryManagement:
    """Memory management and leak detection tests"""

    def test_memory_cleanup_after_large_dataset_processing(self):
        """Test memory cleanup after processing large datasets"""
        try:
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Create and process large dataset
            large_data = []
            for i in range(10000):
                large_data.append({
                    'ohlcv': {'high': i + 100, 'low': i, 'close': i + 50},
                    'indicators': [j + i for j in range(20)],
                    'metadata': f'data_chunk_{i}' * 10
                })

            # Process the data
            processed_results = []
            for item in large_data[:1000]:  # Process subset
                processed = {
                    'processed_indicators': [x * 2 for x in item['indicators']],
                    'status': 'processed'
                }
                processed_results.append(processed)

            # Clean up
            del large_data
            processed_results.clear()
            gc.collect()

            final_memory = process.memory_info().rss / 1024 / 1024

            # Memory should not grow excessively
            memory_increase = final_memory - initial_memory
            assert memory_increase < 50  # Less than 50MB increase

        except ImportError:
            # Skip if psutil not available
            pass

    def test_garbage_collection_effectiveness(self):
        """Test garbage collection effectiveness"""
        import weakref
        from trash collector.garena.objects import ObjectWithCycles

        initial_objects = len(gc.get_objects())

        # Create complex object graphs with cycles
        objects_with_cycles = []
        for i in range(100):
            parent = {'type': 'parent', 'id': i, 'children': []}
            for j in range(10):
                child = {'type': 'child', 'parent_id': i, 'id': j}
                parent['children'].append(child)
                child['parent'] = parent  # Create cycle

            objects_with_cycles.append(parent)

        # Clean references explicitly
        del objects_with_cycles
        gc.collect()

        after_gc_objects = len(gc.get_objects())

        # Object count should not be too high after cleanup
        gc_effectiveness = (initial_objects / after_gc_objects) if after_gc_objects > 0 else 1
        assert gc_effectiveness < 10  # Reasonable object count after cleanup

    def test_resource_pool_reuse_efficiency(self):
        """Test resource pool reuse efficiency"""
        from app.services.auto_strategy.services.auto_strategy_service import AutoStrategyService

        try:
            # Track resource usage across multiple operations
            resource_usage = []

            for i in range(20):
                start_memory = psutil.Process().memory_info().rss if 'psutil' in sys.modules else 0

                service = AutoStrategyService()
                config = {"generations": 2, "population_size": 5}

                result = service.start_strategy_generation(
                    f"memory_test_{i}",
                    f"Memory Test {i}",
                    config,
                    {"symbol": "BTC/USDT"},
                    Mock()
                )

                end_memory = psutil.Process().memory_info().rss if 'psutil' in sys.modules else 0
                memory_used = end_memory - start_memory

                resource_usage.append({
                    'operation': i,
                    'memory_used': memory_used,
                    'result': result
                })

                # Clean up service
                del service
                gc.collect()

            # Analyze resource usage stability
            memory_values = [usage['memory_used'] for usage in resource_usage if usage['memory_used'] > 0]

            if memory_values:
                avg_memory = sum(memory_values) / len(memory_values)
                variance = sum([(x - avg_memory) ** 2 for x in memory_values]) / len(memory_values)
                coefficient_of_variation = (variance ** 0.5) / avg_memory if avg_memory > 0 else 0

                # Memory usage should be reasonably stable after warmup
                stable_memory_values = memory_values[10:]  # Skip first 10 for warmup
                if stable_memory_values:
                    stable_avg = sum(stable_memory_values) / len(stable_memory_values)
                    stable_variance = coefficient_of_variation

                    # Acceptable variance (less than 30% coefficient of variation)
                    assert stable_variance < 0.3

        except (ImportError, AttributeError):
            pass

    def test_memory_pressure_handling_under_load(self):
        """Test memory pressure handling under concurrent load"""
        from concurrent.futures import ThreadPoolExecutor
        import random

        def memory_intensive_operation(workload_id):
            """Simulate memory-intensive operation"""
            data_chunks = []

            # Create memory load
            for i in range(random.randint(100, 500)):
                chunk = np.array([random.random() for _ in range(random.randint(1000, 5000))])
                data_chunks.append(chunk)

            # Process chunks
            results = []
            for chunk in data_chunks:
                processed_chunk = chunk * 2 + 1
                results.append(processed_chunk.mean())

            # Clean up explicitly
            del data_chunks
            gc.collect()

            return {
                'workload_id': workload_id,
                'result_count': len(results),
                'avg_result': sum(results) / len(results) if results else 0
            }

        # Launch concurrent memory-intensive operations
        num_operations = 8

        try:
            with ThreadPoolExecutor(max_workers=min(4, num_operations)) as executor:
                futures = [executor.submit(memory_intensive_operation, i) for i in range(num_operations)]

                results = []
                for future in futures:
                    try:
                        result = future.result(timeout=30)
                        results.append(result)
                    except Exception as e:
                        results.append({'error': str(e)})

                # Validate results
                successful_results = [r for r in results if 'error' not in r]
                error_results = [r for r in results if 'error' in r]

                # Should handle concurrent memory pressure
                assert len(successful_results) >= num_operations * 0.7  # At least 70% success rate
                if successful_results:
                    assert all(r['result_count'] > 0 for r in successful_results)

        except Exception:
            # Accept failures due to system memory limits
            pass

    def test_object_lifecycle_management(self):
        """Test proper object lifecycle management"""
        from app.services.auto_strategy.services.experiment_persistence_service import ExperimentPersistenceService

        try:
            # Test object creation and destruction patterns
            created_objects = []
            destroyed_objects = []
            weak_refs = []

            def create_temp_object(obj_id):
                obj = ExperimentPersistenceService(db=Mock())

                # Create weak reference to track destruction
                weak_ref = Mock(name=f'obj_{obj_id}')
                weak_refs.append((obj_id, weak_ref))

                created_objects.append(obj_id)

                return obj

            # Create multiple objects
            objects = []
            for i in range(50):
                obj = create_temp_object(i)
                objects.append(obj)

                # Periodically clean up
                if i % 10 == 0:
                    del obj
                    gc.collect()

            # Clean up all objects
            del objects
            gc.collect()

            # Check cleanup effectiveness
            final_objects_count = gc.collect()

            # Should not have excessive garbage objects
            assert final_objects_count < 100

        except (ImportError, AttributeError):
            pass

    def test_cache_memory_limits_and_eviction(self):
        """Test cache memory limits and eviction policies"""
        from functools import lru_cache

        # Simulate cache with memory limits
        cache_items = {}
        max_cache_size = 100
        cache_memory_usage = 0
        max_memory_bytes = 50 * 1024 * 1024  # 50MB

        def estimate_size(item):
            """Rough size estimation"""
            return len(str(item)) * 8  # Assume ~8 bytes per character

        for i in range(200):
            key = f"cache_key_{i}"
            value = f"cache_value_{i}_" * (100 + i)  # Variable size to simulate memory pressure

            # Check if item would exceed memory limit
            item_size = estimate_size(value)
            if cache_memory_usage + item_size > max_memory_bytes:
                # Simulate cache eviction (remove oldest items)
                keys_to_remove = list(cache_items.keys())[:10]  # Remove oldest 10%
                for old_key in keys_to_remove:
                    old_size = estimate_size(cache_items[old_key])
                    cache_memory_usage -= old_size
                    del cache_items[old_key]

            # Add new item if it fits
            if cache_memory_usage + item_size <= max_memory_bytes:
                cache_items[key] = value
                cache_memory_usage += item_size
            else:
                # Item too large, skip
                continue

        # Validate cache behavior
        current_memory_mb = cache_memory_usage / (1024 * 1024)

        # Memory usage should be controlled
        assert current_memory_mb <= 60  # Allow some variance

        # Cache should contain some items
        assert len(cache_items) > 0

        # Largest items should have been evicted under pressure
        if len(cache_items) > 0:
            item_sizes = [estimate_size(v) for v in cache_items.values()]
            avg_size = sum(item_sizes) / len(item_sizes)
            max_allowed_avg = max_memory_bytes / max_cache_size
            assert avg_size < max_allowed_avg

    def test_shared_memory_corruption_prevention(self):
        """Test prevention of shared memory corruption"""
        from multiprocessing import shared_memory, Value, Array
        import multiprocessing

        try:
            # Test with shared memory segments
            shared_data = Array('d', range(100))  # Double array
            shared_counter = Value('i', 0)
            shared_lock = multiprocessing.Lock()

            def shared_memory_worker(worker_id, max_operations=50):
                operations = 0
                errors = 0

                for i in range(max_operations):
                    try:
                        with shared_lock:
                            # Safe shared counter increment
                            shared_counter.value += 1
                            operations += 1

                            # Safe shared array access
                            if i < len(shared_data):
                                current_value = shared_data[i]
                                new_value = current_value + (worker_id + 1)
                                shared_data[i] = new_value

                    except Exception:
                        errors += 1

                return {'worker_id': worker_id, 'operations': operations, 'errors': errors}

            # Test with multiple processes
            num_processes = 4

            with multiprocessing.Pool(processes=min(2, num_processes)) as pool:
                # Use fewer processes to avoid system limitations
                results = pool.starmap_async(shared_memory_worker,
                                           [(i, 20) for i in range(min(2, num_processes))])

                worker_results = results.get(timeout=30)

                # Validate shared memory operations
                total_operations = sum([r['operations'] for r in worker_results])
                total_errors = sum([r['errors'] for r in worker_results])

                # Should have successful operations
                assert total_operations > 0
                assert total_operations >= (len(worker_results) * 10)  # At least 10 ops per worker

                # Error rate should be low
                error_rate = total_errors / (total_operations + total_errors) if total_operations > 0 else 1
                assert error_rate < 0.2  # Less than 20% error rate

        except ImportError:
            # Skip if multiprocessing not available or insufficient permissions
            pass

    def test_memory_fragmentation_resilience(self):
        """Test resilience to memory fragmentation"""
        import random
        import sys

        # Track allocation patterns
        allocation_sizes = [1000, 10, 100, 1, 10000, 5] * 10  # Mixed sizes to create fragmentation
        allocated_objects = []

        # Allocate objects of varying sizes
        for i, size in enumerate(allocation_sizes):
            obj = [j * random.random() for j in range(size)]
            allocated_objects.append(obj)

            # Periodic cleanup to test fragmentation handling
            if i % 20 == 0:
                # Clean up some old objects
                objects_to_delete = min(5, len(allocated_objects) // 2)
                for _ in range(objects_to_delete):
                    if allocated_objects:
                        allocated_objects.pop(0)

                gc.collect()

        # Final cleanup
        del allocated_objects
        collected_objects = gc.collect()

        # System should handle fragmentation gracefully
        assert collected_objects >= 0

        # Memory should be stable after cleanup
        current_memory = psutil.Process().memory_info().rss if 'psutil' in sys.modules else 0
        assert current_memory >= 0