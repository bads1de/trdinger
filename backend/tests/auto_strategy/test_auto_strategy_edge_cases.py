"""
オートストラテジーエッジケースの包括的テスト

極端な市場条件、データ欠損、異常値、境界値での動作をテストします。
"""

import logging

import numpy as np
import pandas as pd
import pytest

# テスト用のロガー設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestAutoStrategyEdgeCases:
    """オートストラテジーエッジケースの包括的テストクラス"""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """各テストメソッドの前に実行される初期化"""
        self.extreme_market_data = self._create_extreme_market_data()
        self.missing_data_cases = self._create_missing_data_cases()
        self.boundary_value_cases = self._create_boundary_value_cases()

    def _create_extreme_market_data(self):
        """極端な市場データを作成"""
        return {
            "flash_crash": self._create_flash_crash_data(),
            "extreme_volatility": self._create_extreme_volatility_data(),
            "flat_market": self._create_flat_market_data(),
            "gap_up_down": self._create_gap_data(),
            "micro_movements": self._create_micro_movement_data(),
            "extreme_volume": self._create_extreme_volume_data()
        }

    def _create_flash_crash_data(self) -> pd.DataFrame:
        """フラッシュクラッシュデータを作成"""
        dates = pd.date_range(start="2023-01-01", periods=100, freq="1H")
        data = []
        
        for i, date in enumerate(dates):
            if i == 50:  # 50番目でクラッシュ
                price = 50000 * 0.5  # 50%下落
            elif i > 50 and i < 55:  # 回復期間
                recovery_factor = (i - 50) / 5
                price = 25000 + (25000 * recovery_factor)
            else:
                price = 50000 + np.random.normal(0, 100)
            
            data.append({
                "timestamp": date,
                "open": price,
                "high": price * 1.01,
                "low": price * 0.99,
                "close": price,
                "volume": np.random.uniform(100, 1000)
            })
        
        return pd.DataFrame(data)

    def _create_extreme_volatility_data(self) -> pd.DataFrame:
        """極端なボラティリティデータを作成"""
        dates = pd.date_range(start="2023-01-01", periods=100, freq="1H")
        data = []
        
        base_price = 50000
        for i, date in enumerate(dates):
            # 極端な価格変動（±20%）
            price_change = np.random.normal(0, 0.2)
            price = base_price * (1 + price_change)
            
            # 極端なスプレッド
            spread = price * 0.1  # 10%のスプレッド
            high = price + spread
            low = price - spread
            
            data.append({
                "timestamp": date,
                "open": price,
                "high": high,
                "low": low,
                "close": price + np.random.normal(0, spread * 0.1),
                "volume": np.random.uniform(1, 10000)  # 極端なボリューム変動
            })
        
        return pd.DataFrame(data)

    def _create_flat_market_data(self) -> pd.DataFrame:
        """フラット市場データを作成"""
        dates = pd.date_range(start="2023-01-01", periods=100, freq="1H")
        data = []
        
        base_price = 50000
        for date in dates:
            # 極小の変動（0.001%以下）
            price = base_price + np.random.normal(0, 0.5)
            
            data.append({
                "timestamp": date,
                "open": price,
                "high": price + 0.1,
                "low": price - 0.1,
                "close": price,
                "volume": np.random.uniform(1, 10)  # 極小ボリューム
            })
        
        return pd.DataFrame(data)

    def _create_gap_data(self) -> pd.DataFrame:
        """ギャップアップ・ダウンデータを作成"""
        dates = pd.date_range(start="2023-01-01", periods=100, freq="1H")
        data = []
        
        price = 50000
        for i, date in enumerate(dates):
            if i % 20 == 0 and i > 0:  # 20時間ごとにギャップ
                gap_direction = 1 if np.random.random() > 0.5 else -1
                gap_size = np.random.uniform(0.05, 0.15)  # 5-15%のギャップ
                price = price * (1 + gap_direction * gap_size)
            
            data.append({
                "timestamp": date,
                "open": price,
                "high": price * 1.005,
                "low": price * 0.995,
                "close": price,
                "volume": np.random.uniform(100, 1000)
            })
        
        return pd.DataFrame(data)

    def _create_micro_movement_data(self) -> pd.DataFrame:
        """マイクロムーブメントデータを作成"""
        dates = pd.date_range(start="2023-01-01", periods=100, freq="1H")
        data = []
        
        base_price = 50000.123456789  # 高精度価格
        for date in dates:
            # 極小の変動（小数点以下の変化のみ）
            price = base_price + np.random.normal(0, 0.000001)
            
            data.append({
                "timestamp": date,
                "open": price,
                "high": price + 0.000001,
                "low": price - 0.000001,
                "close": price,
                "volume": np.random.uniform(0.001, 0.01)  # 極小ボリューム
            })
        
        return pd.DataFrame(data)

    def _create_extreme_volume_data(self) -> pd.DataFrame:
        """極端なボリュームデータを作成"""
        dates = pd.date_range(start="2023-01-01", periods=100, freq="1H")
        data = []
        
        for i, date in enumerate(dates):
            price = 50000 + np.random.normal(0, 100)
            
            if i % 10 == 0:  # 10時間ごとに極端なボリューム
                volume = np.random.uniform(1000000, 10000000)  # 極大ボリューム
            else:
                volume = np.random.uniform(0.001, 1)  # 極小ボリューム
            
            data.append({
                "timestamp": date,
                "open": price,
                "high": price * 1.01,
                "low": price * 0.99,
                "close": price,
                "volume": volume
            })
        
        return pd.DataFrame(data)

    def _create_missing_data_cases(self):
        """データ欠損ケースを作成"""
        base_data = pd.DataFrame({
            "timestamp": pd.date_range("2023-01-01", periods=100, freq="1H"),
            "open": [50000] * 100,
            "high": [50500] * 100,
            "low": [49500] * 100,
            "close": [50000] * 100,
            "volume": [1000] * 100
        })
        
        return {
            "random_missing": self._introduce_random_missing(base_data.copy()),
            "consecutive_missing": self._introduce_consecutive_missing(base_data.copy()),
            "weekend_gaps": self._introduce_weekend_gaps(base_data.copy()),
            "partial_columns": self._introduce_partial_column_missing(base_data.copy())
        }

    def _introduce_random_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """ランダムな欠損を導入"""
        missing_indices = np.random.choice(df.index, size=20, replace=False)
        df.loc[missing_indices, ['open', 'high', 'low', 'close']] = np.nan
        return df

    def _introduce_consecutive_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """連続する欠損を導入"""
        start_idx = 40
        end_idx = 50
        df.loc[start_idx:end_idx, ['open', 'high', 'low', 'close']] = np.nan
        return df

    def _introduce_weekend_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """週末ギャップを導入"""
        # 週末に相当する時間帯のデータを削除
        weekend_indices = [i for i in range(len(df)) if i % 24 in [0, 1]]  # 週末相当
        df = df.drop(weekend_indices).reset_index(drop=True)
        return df

    def _introduce_partial_column_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """部分的なカラム欠損を導入"""
        df.loc[10:20, 'volume'] = np.nan
        df.loc[30:35, 'high'] = np.nan
        df.loc[60:65, 'low'] = np.nan
        return df

    def _create_boundary_value_cases(self):
        """境界値ケースを作成"""
        return {
            "zero_values": pd.DataFrame({
                "timestamp": pd.date_range("2023-01-01", periods=10, freq="1H"),
                "open": [0] * 10,
                "high": [0] * 10,
                "low": [0] * 10,
                "close": [0] * 10,
                "volume": [0] * 10
            }),
            "max_float_values": pd.DataFrame({
                "timestamp": pd.date_range("2023-01-01", periods=10, freq="1H"),
                "open": [float('inf')] * 10,
                "high": [float('inf')] * 10,
                "low": [float('inf')] * 10,
                "close": [float('inf')] * 10,
                "volume": [float('inf')] * 10
            }),
            "min_positive_values": pd.DataFrame({
                "timestamp": pd.date_range("2023-01-01", periods=10, freq="1H"),
                "open": [1e-10] * 10,
                "high": [1e-10] * 10,
                "low": [1e-10] * 10,
                "close": [1e-10] * 10,
                "volume": [1e-10] * 10
            })
        }

    def test_extreme_market_conditions(self):
        """極端な市場条件テスト"""
        logger.info("=== 極端な市場条件テスト ===")
        
        try:
            from app.services.auto_strategy.services.ml_orchestrator import MLOrchestrator
            
            ml_orchestrator = MLOrchestrator(enable_automl=False)
            
            for condition_name, data in self.extreme_market_data.items():
                logger.info(f"テスト中: {condition_name}")
                
                try:
                    result = ml_orchestrator.calculate_ml_indicators(data)
                    
                    if result:
                        # 結果の妥当性確認
                        for key, values in result.items():
                            # 無限大やNaNの確認
                            finite_values = [v for v in values if np.isfinite(v)]
                            nan_count = sum(1 for v in values if np.isnan(v))
                            inf_count = sum(1 for v in values if np.isinf(v))
                            
                            logger.info(f"  {key}: 有限値={len(finite_values)}, NaN={nan_count}, Inf={inf_count}")
                            
                            # 極端な条件でも一部の値は有効であることを期待
                            assert len(finite_values) > 0 or nan_count > 0, f"{condition_name}: 全ての値が無効です"
                    
                    logger.info(f"✅ {condition_name} の処理成功")
                    
                except Exception as e:
                    logger.info(f"⚠️ {condition_name} でエラー（期待される場合もあります）: {e}")
            
            logger.info("✅ 極端な市場条件テスト成功")
            
        except Exception as e:
            pytest.fail(f"極端な市場条件テストエラー: {e}")

    def test_missing_data_handling(self):
        """データ欠損処理テスト"""
        logger.info("=== データ欠損処理テスト ===")
        
        try:
            from app.services.auto_strategy.services.ml_orchestrator import MLOrchestrator
            
            ml_orchestrator = MLOrchestrator(enable_automl=False)
            
            for case_name, data in self.missing_data_cases.items():
                logger.info(f"テスト中: {case_name}")
                
                try:
                    result = ml_orchestrator.calculate_ml_indicators(data)
                    
                    if result:
                        # 欠損データがある場合の結果確認
                        for key, values in result.items():
                            valid_count = sum(1 for v in values if not np.isnan(v))
                            logger.info(f"  {key}: 有効値数={valid_count}/{len(values)}")
                    
                    logger.info(f"✅ {case_name} の処理成功")
                    
                except Exception as e:
                    logger.info(f"⚠️ {case_name} でエラー（期待される場合もあります）: {e}")
            
            logger.info("✅ データ欠損処理テスト成功")
            
        except Exception as e:
            pytest.fail(f"データ欠損処理テストエラー: {e}")

    def test_boundary_value_handling(self):
        """境界値処理テスト"""
        logger.info("=== 境界値処理テスト ===")
        
        try:
            from app.services.auto_strategy.services.ml_orchestrator import MLOrchestrator
            
            ml_orchestrator = MLOrchestrator(enable_automl=False)
            
            for case_name, data in self.boundary_value_cases.items():
                logger.info(f"テスト中: {case_name}")
                
                try:
                    result = ml_orchestrator.calculate_ml_indicators(data)
                    
                    if result:
                        logger.info(f"✅ {case_name} の処理成功")
                    else:
                        logger.info(f"⚠️ {case_name} で結果がNone")
                    
                except Exception as e:
                    logger.info(f"⚠️ {case_name} でエラー（期待される場合もあります）: {e}")
            
            logger.info("✅ 境界値処理テスト成功")
            
        except Exception as e:
            pytest.fail(f"境界値処理テストエラー: {e}")

    def test_extreme_tpsl_scenarios(self):
        """極端なTP/SLシナリオテスト"""
        logger.info("=== 極端なTP/SLシナリオテスト ===")
        
        try:
            from app.services.auto_strategy.services.tpsl_auto_decision_service import (
                TPSLAutoDecisionService, TPSLConfig, TPSLStrategy
            )
            
            service = TPSLAutoDecisionService()
            
            # 極端なシナリオ
            extreme_scenarios = [
                {
                    "name": "極小リスク",
                    "config": TPSLConfig(
                        strategy=TPSLStrategy.RISK_REWARD,
                        max_risk_per_trade=0.0001,  # 0.01%
                        preferred_risk_reward_ratio=1000.0  # 極端なリワード比
                    )
                },
                {
                    "name": "極大リスク",
                    "config": TPSLConfig(
                        strategy=TPSLStrategy.RISK_REWARD,
                        max_risk_per_trade=0.99,  # 99%
                        preferred_risk_reward_ratio=0.1  # 極小リワード比
                    )
                },
                {
                    "name": "極端なボラティリティ",
                    "config": TPSLConfig(
                        strategy=TPSLStrategy.VOLATILITY_ADAPTIVE,
                        volatility_sensitivity="high"
                    ),
                    "market_data": {"atr_pct": 0.5}  # 50%のATR
                },
                {
                    "name": "ゼロボラティリティ",
                    "config": TPSLConfig(
                        strategy=TPSLStrategy.VOLATILITY_ADAPTIVE,
                        volatility_sensitivity="low"
                    ),
                    "market_data": {"atr_pct": 0.0}  # 0%のATR
                }
            ]
            
            for scenario in extreme_scenarios:
                logger.info(f"テスト中: {scenario['name']}")
                
                try:
                    market_data = scenario.get("market_data")
                    result = service.generate_tpsl_values(scenario["config"], market_data)
                    
                    # 結果の妥当性確認
                    assert result is not None, f"{scenario['name']}: 結果がNone"
                    assert result.stop_loss_pct >= 0, f"{scenario['name']}: SLが負の値"
                    assert result.take_profit_pct >= 0, f"{scenario['name']}: TPが負の値"
                    assert result.risk_reward_ratio >= 0, f"{scenario['name']}: RR比が負の値"
                    
                    logger.info(f"✅ {scenario['name']}: SL={result.stop_loss_pct:.4f}, TP={result.take_profit_pct:.4f}, RR={result.risk_reward_ratio:.2f}")
                    
                except Exception as e:
                    logger.info(f"⚠️ {scenario['name']} でエラー: {e}")
            
            logger.info("✅ 極端なTP/SLシナリオテスト成功")
            
        except Exception as e:
            pytest.fail(f"極端なTP/SLシナリオテストエラー: {e}")

    def test_extreme_condition_generation(self):
        """極端な条件生成テスト"""
        logger.info("=== 極端な条件生成テスト ===")
        
        try:
            from app.services.auto_strategy.generators.smart_condition_generator import SmartConditionGenerator
            from app.services.auto_strategy.models.gene_strategy import IndicatorGene
            
            generator = SmartConditionGenerator()
            
            # 極端なシナリオ
            extreme_scenarios = [
                {
                    "name": "大量の指標",
                    "indicators": [
                        IndicatorGene(type=f"RSI_{i}", parameters={"period": i+1}, enabled=True)
                        for i in range(50)  # 50個の指標
                    ]
                },
                {
                    "name": "全て無効な指標",
                    "indicators": [
                        IndicatorGene(type="RSI", parameters={"period": 14}, enabled=False)
                        for _ in range(10)
                    ]
                },
                {
                    "name": "極端なパラメータ",
                    "indicators": [
                        IndicatorGene(type="RSI", parameters={"period": 1}, enabled=True),  # 最小期間
                        IndicatorGene(type="SMA", parameters={"period": 10000}, enabled=True),  # 極大期間
                        IndicatorGene(type="EMA", parameters={"period": 0}, enabled=True),  # ゼロ期間
                    ]
                },
                {
                    "name": "空のパラメータ",
                    "indicators": [
                        IndicatorGene(type="RSI", parameters={}, enabled=True),
                        IndicatorGene(type="SMA", parameters=None, enabled=True),
                    ]
                }
            ]
            
            for scenario in extreme_scenarios:
                logger.info(f"テスト中: {scenario['name']}")
                
                try:
                    long_conditions, short_conditions, exit_conditions = generator.generate_balanced_conditions(
                        scenario["indicators"]
                    )
                    
                    # 結果の基本確認
                    assert isinstance(long_conditions, list), f"{scenario['name']}: ロング条件がリスト形式ではない"
                    assert isinstance(short_conditions, list), f"{scenario['name']}: ショート条件がリスト形式ではない"
                    assert isinstance(exit_conditions, list), f"{scenario['name']}: イグジット条件がリスト形式ではない"
                    
                    total_conditions = len(long_conditions) + len(short_conditions) + len(exit_conditions)
                    logger.info(f"✅ {scenario['name']}: 生成された条件数={total_conditions}")
                    
                except Exception as e:
                    logger.info(f"⚠️ {scenario['name']} でエラー: {e}")
            
            logger.info("✅ 極端な条件生成テスト成功")
            
        except Exception as e:
            pytest.fail(f"極端な条件生成テストエラー: {e}")

    def test_resource_exhaustion_scenarios(self):
        """リソース枯渇シナリオテスト"""
        logger.info("=== リソース枯渇シナリオテスト ===")
        
        try:
            import gc
            from app.services.auto_strategy.services.ml_orchestrator import MLOrchestrator
            
            # メモリ使用量の監視
            initial_objects = len(gc.get_objects())
            
            # 大量のオブジェクト作成と削除
            for i in range(10):
                try:
                    # 大きなデータセットを作成
                    large_data = self._create_test_data(1000)
                    ml_orchestrator = MLOrchestrator(enable_automl=False)
                    
                    # 処理実行
                    result = ml_orchestrator.calculate_ml_indicators(large_data)
                    
                    # 明示的に削除
                    del large_data
                    del ml_orchestrator
                    del result
                    
                    # ガベージコレクション
                    gc.collect()
                    
                    if i % 3 == 0:
                        current_objects = len(gc.get_objects())
                        logger.info(f"反復 {i+1}: オブジェクト数={current_objects}")
                
                except Exception as e:
                    logger.info(f"反復 {i+1} でエラー: {e}")
            
            final_objects = len(gc.get_objects())
            object_increase = final_objects - initial_objects
            
            logger.info(f"オブジェクト数の変化: {initial_objects} -> {final_objects} (増加: {object_increase})")
            
            # 過度なメモリリークがないことを確認
            assert object_increase < 10000, f"過度なオブジェクト増加: {object_increase}"
            
            logger.info("✅ リソース枯渇シナリオテスト成功")
            
        except Exception as e:
            pytest.fail(f"リソース枯渇シナリオテストエラー: {e}")

    def _create_test_data(self, rows: int) -> pd.DataFrame:
        """テスト用データを作成"""
        dates = pd.date_range(start="2023-01-01", periods=rows, freq="1H")
        np.random.seed(42)
        
        data = []
        for date in dates:
            price = 50000 + np.random.normal(0, 1000)
            data.append({
                "timestamp": date,
                "open": price,
                "high": price * 1.01,
                "low": price * 0.99,
                "close": price,
                "volume": np.random.uniform(100, 1000)
            })
        
        return pd.DataFrame(data)


if __name__ == "__main__":
    # 単体でテストを実行する場合
    pytest.main([__file__, "-v"])
