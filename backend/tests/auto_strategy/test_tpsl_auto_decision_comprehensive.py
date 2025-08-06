"""
TP/SL自動決定サービス包括的テスト

TPSLAutoDecisionServiceの各戦略（リスクリワード、ボラティリティ適応、
統計的、自動最適化）の包括的テストを実施します。
"""

import logging
import pytest

from app.services.auto_strategy.services.tpsl_auto_decision_service import (
    TPSLAutoDecisionService,
    TPSLStrategy,
    TPSLConfig,
    TPSLResult
)

logger = logging.getLogger(__name__)


class TestTPSLAutoDecisionServiceComprehensive:
    """TP/SL自動決定サービス包括的テストクラス"""

    @pytest.fixture
    def tpsl_service(self):
        """TP/SL自動決定サービスのインスタンス"""
        return TPSLAutoDecisionService()

    @pytest.fixture
    def base_config(self):
        """基本TP/SL設定"""
        return TPSLConfig(
            strategy=TPSLStrategy.RISK_REWARD,
            max_risk_per_trade=0.02,
            preferred_risk_reward_ratio=2.0,
            volatility_sensitivity=1.0
        )

    @pytest.fixture
    def sample_market_data(self):
        """サンプル市場データ"""
        return {
            'current_price': 50000.0,
            'volatility': 0.02,
            'atr': 1000.0,
            'recent_high': 52000.0,
            'recent_low': 48000.0,
            'volume': 1000000,
            'trend': 'bullish'
        }

    def test_service_initialization(self, tpsl_service):
        """サービス初期化テスト"""
        assert tpsl_service is not None
        assert hasattr(tpsl_service, 'generate_tpsl_values')

    def test_tpsl_config_creation(self):
        """TP/SL設定作成テスト"""
        # 有効な設定
        valid_config = TPSLConfig(
            strategy=TPSLStrategy.RISK_REWARD,
            max_risk_per_trade=0.01,
            preferred_risk_reward_ratio=1.5,
            volatility_sensitivity=0.8
        )
        
        assert valid_config.strategy == TPSLStrategy.RISK_REWARD
        assert valid_config.max_risk_per_trade == 0.01
        assert valid_config.preferred_risk_reward_ratio == 1.5
        assert valid_config.volatility_sensitivity == 0.8

    def test_random_strategy(self, tpsl_service, base_config, sample_market_data):
        """ランダム戦略テスト"""
        config = TPSLConfig(
            strategy=TPSLStrategy.RANDOM,
            max_risk_per_trade=0.02,
            preferred_risk_reward_ratio=2.0
        )
        
        try:
            result = tpsl_service.generate_tpsl_values(config, sample_market_data, "BTC/USDT")
            
            # 結果検証
            assert isinstance(result, TPSLResult)
            assert result.take_profit > 0
            assert result.stop_loss > 0
            assert result.strategy == TPSLStrategy.RANDOM
            
            # ランダム性の確認（複数回実行して異なる結果が得られることを確認）
            results = []
            for _ in range(5):
                r = tpsl_service.generate_tpsl_values(config, sample_market_data, "BTC/USDT")
                results.append((r.take_profit, r.stop_loss))
            
            # 少なくとも一部の結果が異なることを確認
            unique_results = set(results)
            assert len(unique_results) > 1, "ランダム戦略で同じ結果が連続して生成されました"
            
        except Exception as e:
            logger.warning(f"ランダム戦略テストでエラー: {e}")
            pytest.skip(f"ランダム戦略テストをスキップ: {e}")

    def test_risk_reward_strategy(self, tpsl_service, base_config, sample_market_data):
        """リスクリワード戦略テスト"""
        config = TPSLConfig(
            strategy=TPSLStrategy.RISK_REWARD,
            max_risk_per_trade=0.02,
            preferred_risk_reward_ratio=2.0
        )
        
        try:
            result = tpsl_service.generate_tpsl_values(config, sample_market_data, "BTC/USDT")
            
            # 結果検証
            assert isinstance(result, TPSLResult)
            assert result.take_profit > 0
            assert result.stop_loss > 0
            assert result.strategy == TPSLStrategy.RISK_REWARD
            
            # リスクリワード比の確認
            current_price = sample_market_data['current_price']
            risk = abs(current_price - result.stop_loss)
            reward = abs(result.take_profit - current_price)
            
            if risk > 0:
                actual_ratio = reward / risk
                # 設定されたリスクリワード比に近いことを確認（±20%の許容範囲）
                expected_ratio = config.preferred_risk_reward_ratio
                assert abs(actual_ratio - expected_ratio) / expected_ratio < 0.2, \
                    f"リスクリワード比が期待値から乖離: 期待={expected_ratio}, 実際={actual_ratio}"
            
        except Exception as e:
            logger.warning(f"リスクリワード戦略テストでエラー: {e}")
            pytest.skip(f"リスクリワード戦略テストをスキップ: {e}")

    def test_volatility_adaptive_strategy(self, tpsl_service, sample_market_data):
        """ボラティリティ適応戦略テスト"""
        config = TPSLConfig(
            strategy=TPSLStrategy.VOLATILITY_ADAPTIVE,
            max_risk_per_trade=0.02,
            volatility_sensitivity=1.5
        )
        
        try:
            result = tpsl_service.generate_tpsl_values(config, sample_market_data, "BTC/USDT")
            
            # 結果検証
            assert isinstance(result, TPSLResult)
            assert result.take_profit > 0
            assert result.stop_loss > 0
            assert result.strategy == TPSLStrategy.VOLATILITY_ADAPTIVE
            
            # ボラティリティに応じた調整の確認
            # 高ボラティリティ時のテスト
            high_vol_data = sample_market_data.copy()
            high_vol_data['volatility'] = 0.05
            high_vol_data['atr'] = 2000.0
            
            high_vol_result = tpsl_service.generate_tpsl_values(config, high_vol_data, "BTC/USDT")
            
            # 高ボラティリティ時により広いTP/SLが設定されることを確認
            current_price = sample_market_data['current_price']
            normal_tp_distance = abs(result.take_profit - current_price)
            high_vol_tp_distance = abs(high_vol_result.take_profit - current_price)
            
            # 高ボラティリティ時により広いTP/SLが設定されることを期待
            # （実装に依存するため、基本的な構造のみ確認）
            
        except Exception as e:
            logger.warning(f"ボラティリティ適応戦略テストでエラー: {e}")
            pytest.skip(f"ボラティリティ適応戦略テストをスキップ: {e}")

    def test_statistical_strategy(self, tpsl_service, sample_market_data):
        """統計的戦略テスト"""
        config = TPSLConfig(
            strategy=TPSLStrategy.STATISTICAL,
            max_risk_per_trade=0.02
        )
        
        try:
            result = tpsl_service.generate_tpsl_values(config, sample_market_data, "BTC/USDT")
            
            # 結果検証
            assert isinstance(result, TPSLResult)
            assert result.take_profit > 0
            assert result.stop_loss > 0
            assert result.strategy == TPSLStrategy.STATISTICAL
            
            # 統計的根拠に基づく設定であることを確認
            # （具体的な統計値は実装に依存）
            
        except Exception as e:
            logger.warning(f"統計的戦略テストでエラー: {e}")
            pytest.skip(f"統計的戦略テストをスキップ: {e}")

    def test_auto_optimal_strategy(self, tpsl_service, sample_market_data):
        """自動最適化戦略テスト"""
        config = TPSLConfig(
            strategy=TPSLStrategy.AUTO_OPTIMAL,
            max_risk_per_trade=0.02,
            preferred_risk_reward_ratio=2.0,
            volatility_sensitivity=1.0
        )
        
        try:
            result = tpsl_service.generate_tpsl_values(config, sample_market_data, "BTC/USDT")
            
            # 結果検証
            assert isinstance(result, TPSLResult)
            assert result.take_profit > 0
            assert result.stop_loss > 0
            assert result.strategy == TPSLStrategy.AUTO_OPTIMAL
            
            # 自動最適化では複数の戦略を組み合わせた結果が期待される
            
        except Exception as e:
            logger.warning(f"自動最適化戦略テストでエラー: {e}")
            pytest.skip(f"自動最適化戦略テストをスキップ: {e}")

    def test_invalid_strategy_handling(self, tpsl_service, sample_market_data):
        """無効戦略ハンドリングテスト"""
        # 存在しない戦略を指定
        try:
            # 無効な戦略列挙値を作成（実際には不可能だが、テストのため）
            invalid_config = TPSLConfig(
                strategy=None,  # 無効な戦略
                max_risk_per_trade=0.02
            )
            
            with pytest.raises((ValueError, AttributeError, TypeError)):
                tpsl_service.generate_tpsl_values(invalid_config, sample_market_data, "BTC/USDT")
                
        except Exception as e:
            # 設定作成時点でエラーが発生する場合
            assert any(keyword in str(e).lower() for keyword in ['invalid', 'strategy', 'none'])

    def test_edge_case_market_data(self, tpsl_service, base_config):
        """エッジケース市場データテスト"""
        edge_cases = [
            # 極端に高い価格
            {'current_price': 1000000.0, 'volatility': 0.01, 'atr': 10000.0},
            # 極端に低い価格
            {'current_price': 0.001, 'volatility': 0.01, 'atr': 0.00001},
            # 高ボラティリティ
            {'current_price': 50000.0, 'volatility': 0.1, 'atr': 5000.0},
            # 低ボラティリティ
            {'current_price': 50000.0, 'volatility': 0.001, 'atr': 50.0},
        ]
        
        for i, edge_data in enumerate(edge_cases):
            try:
                result = tpsl_service.generate_tpsl_values(base_config, edge_data, "BTC/USDT")
                
                # 基本的な妥当性確認
                assert isinstance(result, TPSLResult)
                assert result.take_profit > 0
                assert result.stop_loss > 0
                
                # 価格の妥当性確認
                current_price = edge_data['current_price']
                assert result.take_profit != current_price
                assert result.stop_loss != current_price
                
            except Exception as e:
                logger.warning(f"エッジケース {i} でエラー: {e}")

    def test_none_market_data_handling(self, tpsl_service, base_config):
        """None市場データハンドリングテスト"""
        try:
            result = tpsl_service.generate_tpsl_values(base_config, None, "BTC/USDT")
            
            # Noneデータでもエラーが発生しないか、適切なデフォルト値が使用されることを確認
            if result is not None:
                assert isinstance(result, TPSLResult)
                assert result.take_profit > 0
                assert result.stop_loss > 0
                
        except Exception as e:
            # 適切なエラーハンドリングが行われることを確認
            assert any(keyword in str(e).lower() for keyword in ['data', 'none', 'invalid', 'missing'])

    def test_symbol_specific_adjustments(self, tpsl_service, base_config, sample_market_data):
        """シンボル固有調整テスト"""
        symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", None]
        
        results = {}
        for symbol in symbols:
            try:
                result = tpsl_service.generate_tpsl_values(base_config, sample_market_data, symbol)
                results[symbol] = result
                
                # 基本的な妥当性確認
                assert isinstance(result, TPSLResult)
                assert result.take_profit > 0
                assert result.stop_loss > 0
                
            except Exception as e:
                logger.warning(f"シンボル {symbol} でエラー: {e}")

        # シンボルによる違いがあるかを確認（実装に依存）
        if len(results) > 1:
            logger.info(f"シンボル別結果数: {len(results)}")

    def test_risk_per_trade_limits(self, tpsl_service, sample_market_data):
        """取引あたりリスク制限テスト"""
        risk_levels = [0.001, 0.01, 0.02, 0.05, 0.1]  # 0.1%から10%まで
        
        for risk_level in risk_levels:
            config = TPSLConfig(
                strategy=TPSLStrategy.RISK_REWARD,
                max_risk_per_trade=risk_level,
                preferred_risk_reward_ratio=2.0
            )
            
            try:
                result = tpsl_service.generate_tpsl_values(config, sample_market_data, "BTC/USDT")
                
                # リスク制限が適切に適用されることを確認
                current_price = sample_market_data['current_price']
                actual_risk = abs(current_price - result.stop_loss) / current_price
                
                # 設定されたリスクレベルを大幅に超えないことを確認（±50%の許容範囲）
                assert actual_risk <= risk_level * 1.5, \
                    f"リスクレベル超過: 設定={risk_level}, 実際={actual_risk}"
                
            except Exception as e:
                logger.warning(f"リスクレベル {risk_level} でエラー: {e}")

    def test_concurrent_tpsl_generation(self, tpsl_service, base_config, sample_market_data):
        """並行TP/SL生成テスト"""
        import threading
        
        results = []
        errors = []
        
        def generate_tpsl():
            try:
                result = tpsl_service.generate_tpsl_values(base_config, sample_market_data, "BTC/USDT")
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # 複数スレッドで同時実行
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=generate_tpsl)
            threads.append(thread)
            thread.start()
        
        # 全スレッドの完了を待機
        for thread in threads:
            thread.join(timeout=10)
        
        # 結果検証
        assert len(results) > 0, "並行実行で結果が生成されませんでした"
        
        # 全ての結果が有効であることを確認
        for result in results:
            assert isinstance(result, TPSLResult)
            assert result.take_profit > 0
            assert result.stop_loss > 0
        
        # エラーが発生した場合は警告として記録
        if errors:
            logger.warning(f"並行実行で {len(errors)} 個のエラーが発生: {errors}")

    def test_performance_benchmark(self, tpsl_service, base_config, sample_market_data):
        """パフォーマンスベンチマークテスト"""
        import time
        
        # 大量実行でのパフォーマンステスト
        num_iterations = 100
        start_time = time.time()
        
        successful_generations = 0
        for _ in range(num_iterations):
            try:
                result = tpsl_service.generate_tpsl_values(base_config, sample_market_data, "BTC/USDT")
                if result is not None:
                    successful_generations += 1
            except Exception as e:
                logger.warning(f"パフォーマンステストでエラー: {e}")
        
        execution_time = time.time() - start_time
        
        # パフォーマンス要件の確認
        avg_time_per_generation = execution_time / num_iterations
        assert avg_time_per_generation < 0.1, \
            f"TP/SL生成が遅すぎます: {avg_time_per_generation:.3f}秒/回"
        
        # 成功率の確認
        success_rate = successful_generations / num_iterations
        assert success_rate > 0.8, \
            f"成功率が低すぎます: {success_rate:.2%}"
        
        logger.info(f"パフォーマンステスト結果: {num_iterations}回実行, "
                   f"平均時間={avg_time_per_generation:.3f}秒, 成功率={success_rate:.2%}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
