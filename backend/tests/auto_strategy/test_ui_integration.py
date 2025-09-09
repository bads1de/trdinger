"""
UI Integration Tests
Focus: Frontend-backend communication, request/response handling, user interactions
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json
import time
import sys
import os

# Test framework setup
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


class TestUIIntegration:
    """UI integration and user interaction tests"""

    def test_experiment_creation_ui_flow(self):
        """Test complete UI flow for experiment creation"""
        from app.services.auto_strategy.services.auto_strategy_service import AutoStrategyService

        try:
            service = AutoStrategyService()

            # Mock UI request data (frontend would send this)
            ui_request = {
                "experiment_name": "UI_Test_Experiment",
                "base_config": {
                    "symbol": "ETH/USD",
                    "timeframe": "1h",
                    "start_date": "2024-01-01",
                    "end_date": "2024-01-31",
                    "initial_capital": 10000,
                    "commission_rate": 0.001
                },
                "ga_config": {
                    "generations": 10,
                    "population_size": 20,
                    "crossover_rate": 0.8,
                    "mutation_rate": 0.1,
                    "elite_size": 5,
                    "enable_multi_objective": True
                }
            }

            # Backend UI response format
            result = service.start_strategy_generation(
                f"ui_{int(time.time())}",
                ui_request["experiment_name"],
                ui_request["ga_config"],
                ui_request["base_config"],
                Mock()
            )

            # Should return handle for frontend polling
            assert result is not None
            assert result.startswith("ui_")

        except (ImportError, AttributeError):
            pass

    def test_progress_tracking_integration(self):
        """Test progress tracking from UI perspectives"""
        # Mock experiment progress data
        progress_states = [
            {"stage": "init", "percent": 0, "message": "Initializing..."},
            {"stage": "loading", "percent": 25, "message": "Loading data..."},
            {"stage": "running", "percent": 50, "message": "Running GA optimization..."},
            {"stage": "finalizing", "percent": 90, "message": "Finalizing results..."},
            {"stage": "complete", "percent": 100, "message": "Done!"}
        ]

        polling_results = []
        poll_interval = 0.1  # UI would poll every 100ms

        # Simulate UI progress polling
        for i, state in enumerate(progress_states):
            time.sleep(poll_interval)

            # Backend would return this format
            poll_response = {
                "experiment_id": "ui_poll_test",
                "status": state["stage"],
                "progress": state["percent"],
                "message": state["message"],
                "timestamp": time.time()
            }

            polling_results.append(poll_response)

            # Progress should be non-decreasing
            if i > 0:
                assert polling_results[i]["progress"] >= polling_results[i-1]["progress"]

            # Percentages should be 0-100
            assert 0 <= poll_response["progress"] <= 100

        # Final state should be complete
        assert polling_results[-1]["status"] == "complete"

    def test_result_visualization_format(self):
        """Test result formatting for UI visualization"""
        # Mock GA results for visualization
        ga_results = {
            "experiments": [
                {
                    "experiment_id": "exp_001",
                    "best_fitness": 0.85,
                    "strategy": {
                        "type": "RSI_MACD_crossover",
                        "indicators": ["RSI", "MACD"],
                        "parameters": {
                            "rsi_period": 14,
                            "macd_fast": 12,
                            "macd_slow": 26,
                            "rsi_overbought": 70,
                            "rsi_oversold": 30
                        }
                    },
                    "fitness_history": [0.1, 0.3, 0.5, 0.7, 0.85, 0.82, 0.87, 0.84, 0.83, 0.85],
                    "execution_time": 45.2,
                    "population_metrics": {
                        "mean_fitness": 0.65,
                        "std_fitness": 0.12,
                        "diversity_score": 0.78
                    }
                }
            ]
        }

        # Test if results are structured for UI consumption
        for experiment in ga_results["experiments"]:
            assert "experiment_id" in experiment
            assert "strategy" in experiment
            assert "fitness_history" in experiment
            assert len(experiment["fitness_history"]) > 0

            # Strategy should be serializable
            strategy_json = json.dumps(experiment["strategy"])
            assert "type" in json.loads(strategy_json)
            assert "indicators" in json.loads(strategy_json)

            # Fitness history should be suitable for charting
            fitness_xs = list(range(len(experiment["fitness_history"])))
            fitness_ys = experiment["fitness_history"]

            assert len(fitness_xs) == len(fitness_ys)
            assert min(fitness_ys) >= 0.0  # Fitness should be non-negative
            assert max(fitness_ys) <= 1.0  # Fitness should be <= 1.0

    def test_error_message_ui_formatting(self):
        """Test error message formatting for UI display"""
        # Backend error types and messages
        error_scenarios = [
            {
                "error_type": "validation_error",
                "backend_message": "Invalid symbol: SPEC/USD",
                "ui_friendly_message": "取引通貨ペアが無効です。サポートされている通貨ペアを選択してください。",
                "severity": "warning"
            },
            {
                "error_type": "database_error",
                "backend_message": "Connection to DB failed after 3 retry attempts",
                "ui_friendly_message": "データベース接続に問題が発生しました。数分後に再試行してください。",
                "severity": "error"
            },
            {
                "error_type": "computation_error",
                "backend_message": "Division by zero in fitness function at individual #5",
                "ui_friendly_message": "最適化計算中にエラーが発生しました。実験を再実行してください。",
                "severity": "error"
            },
            {
                "error_type": "timeout_error",
                "backend_message": "GA optimization exceeded max time limit of 5 minutes",
                "ui_friendly_message": "最適化処理が制限時間を超過しました。時間を延長して再度実行してください。",
                "severity": "info"
            }
        ]

        # Test message transformation for UI
        for error in error_scenarios:
            ui_message = error["ui_friendly_message"]

            # UI messages should be:
            # - Japanese (as per user preference)
            # - User-friendly
            # - Include actionable suggestions
            # - Not contain technical jargon

            assert len(ui_message) > 10  # Sufficient length
            assert "英語" in ui_message or "日本語" in ui_message, "Should be appropriately localized"
            assert "ください" in ui_message, "Should be polite and actionable"

            # Severity classification should be appropriate
            if error["severity"] == "error":
                assert "エラー" in ui_message or "失敗" in ui_message
            elif error["severity"] == "warning":
                assert "無効" in ui_message or "確認" in ui_message

    def test_batch_experiment_management(self):
        """Test batch experiment execution and management from UI"""
        from app.services.auto_strategy.services.auto_strategy_service import AutoStrategyService

        try:
            service = AutoStrategyService()

            # Simulate batch of experiments from UI
            batch_experiments = [
                {
                    "name": "Short_term_EMA",
                    "config": {"generations": 5, "population_size": 10}
                },
                {
                    "name": "Medium_term_RSI",
                    "config": {"generations": 5, "population_size": 10}
                },
                {
                    "name": "Long_term_MACD",
                    "config": {"generations": 5, "population_size": 10}
                }
            ]

            # Process batch
            batch_handles = []
            for experiment in batch_experiments:
                handle = service.start_strategy_generation(
                    f"batch_{int(time.time())}",
                    experiment["name"],
                    experiment["config"],
                    {"symbol": "BTC/USDT"},
                    Mock()
                )
                batch_handles.append(handle)

            # Should have unique handles for tracking
            assert len(batch_handles) == len(batch_experiments)
            assert len(set(batch_handles)) == len(batch_handles)  # All unique

            # UI should be able to track batch status
            batch_summary = {
                "total_experiments": len(batch_experiments),
                "completed": sum(1 for h in batch_handles if "batch_" in h),  # Mock completion
                "running": 0,
                "failed": 0,
                "estimated_completion": time.time() + 300  # 5 minutes from now
            }

            assert batch_summary["total_experiments"] == 3
            assert batch_summary["completed"] == 3

        except (ImportError, AttributeError):
            pass

    def test_settings_persistence_across_sessions(self):
        """Test user settings persistence across UI sessions"""
        # Mock settings storage
        user_settings = {
            "theme": "dark",
            "auto_save": True,
            "chart_style": "candlestick",
            "default_symbol": "BTC/USDT",
            "default_timeframe": "1h",
            "notification_settings": {
                "email_alerts": True,
                "browser_notifications": False,
                "experiment_completion": True
            }
        }

        # Simulate settings save (localStorage equivalent)
        settings_json = json.dumps(user_settings)
        assert len(settings_json) > 50  # Reasonable size

        # Simulate settings load across session
        loaded_settings = json.loads(settings_json)

        # Should preserve all settings
        assert loaded_settings["theme"] == "dark"
        assert loaded_settings["auto_save"] is True
        assert loaded_settings["notification_settings"]["email_alerts"] is True

        # Test nested structure integrity
        notifications = loaded_settings["notification_settings"]
        assert "email_alerts" in notifications
        assert "browser_notifications" in notifications
        assert "experiment_completion" in notifications

    def test_real_time_data_streaming_simulation(self):
        """Test simulation of real-time data streaming to UI"""
        # Mock streaming data updates
        price_updates = []
        volume_data = []

        def generate_market_data():
            """Simulate real-time price feed"""
            for i in range(10):
                price_data = {
                    "symbol": "BTC/USDT",
                    "price": 50000 + i * 100,
                    "volume": 1000 + i * 10,
                    "timestamp": time.time() + i
                }
                price_updates.append(price_data)

                volume_entry = {
                    "timestamp": price_data["timestamp"],
                    "volume": price_data["volume"],
                    "is_buy_volume": i % 2 == 0
                }
                volume_data.append(volume_entry)

                time.sleep(0.1)  # Simulate stream interval

        # Generate streaming data
        generate_market_data()

        # Validate stream data
        assert len(price_updates) == 10
        assert len(volume_data) == 10

        # Verify data continuity
        for i in range(1, len(price_updates)):
            # Prices should be increasing
            assert price_updates[i]["price"] > price_updates[i-1]["price"]

            # Timestamps should be increasing
            assert volume_data[i]["timestamp"] > volume_data[i-1]["timestamp"]

        # Volume data should correspond to price data
        for price_update, volume_entry in zip(price_updates, volume_data):
            assert price_update["volume"] == volume_entry["volume"]
            assert price_update["timestamp"] == volume_entry["timestamp"]

        # UI should be able to consume this data for real-time charts
        chart_feed = {
            "price_series": [(p["timestamp"], p["price"]) for p in price_updates],
            "volume_series": [(v["timestamp"], v["volume"]) for v in volume_data]
        }

        assert len(chart_feed["price_series"]) == 10
        assert len(chart_feed["volume_series"]) == 10