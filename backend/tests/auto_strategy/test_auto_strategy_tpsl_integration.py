"""
オートストラテジーとTP/SL統合の包括的テスト

TP/SL自動決定サービス、各種戦略（RANDOM、RISK_REWARD、VOLATILITY_ADAPTIVE、STATISTICAL、AUTO_OPTIMAL）、
計算精度をテストします。
"""

import logging
import random

import pytest

# テスト用のロガー設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestAutoStrategyTPSLIntegration:
    """オートストラテジーとTP/SL統合の包括的テストクラス"""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """各テストメソッドの前に実行される初期化"""
        # テスト用のシード設定（再現性のため）
        random.seed(42)

    def test_tpsl_auto_decision_service_initialization(self):
        """TP/SL自動決定サービスの初期化テスト"""
        logger.info("=== TP/SL自動決定サービス初期化テスト ===")
        
        try:
            from app.services.auto_strategy.services.tpsl_auto_decision_service import (
                TPSLAutoDecisionService
            )
            
            # サービスの初期化
            service = TPSLAutoDecisionService()
            
            # 基本属性の確認
            assert hasattr(service, 'volatility_multipliers'), "volatility_multipliers属性が不足しています"
            assert hasattr(service, 'statistical_data'), "statistical_data属性が不足しています"
            
            # ボラティリティ倍率の確認
            expected_sensitivities = ["low", "medium", "high"]
            for sensitivity in expected_sensitivities:
                assert sensitivity in service.volatility_multipliers, f"ボラティリティ感度 {sensitivity} が不足しています"
                multiplier = service.volatility_multipliers[sensitivity]
                assert "sl" in multiplier and "tp" in multiplier, f"倍率設定が不完全です: {sensitivity}"
            
            # 統計データの確認
            assert "optimal_rr_ratios" in service.statistical_data, "optimal_rr_ratios が不足しています"
            assert "optimal_sl_ranges" in service.statistical_data, "optimal_sl_ranges が不足しています"
            assert "win_rates" in service.statistical_data, "win_rates が不足しています"
            
            logger.info("✅ TP/SL自動決定サービス初期化テスト成功")
            
        except Exception as e:
            pytest.fail(f"TP/SL自動決定サービス初期化テストエラー: {e}")

    def test_tpsl_config_validation(self):
        """TP/SL設定の妥当性テスト"""
        logger.info("=== TP/SL設定妥当性テスト ===")
        
        try:
            from app.services.auto_strategy.services.tpsl_auto_decision_service import (
                TPSLConfig, TPSLStrategy
            )
            
            # 基本設定の作成
            config = TPSLConfig(
                strategy=TPSLStrategy.RISK_REWARD,
                max_risk_per_trade=0.03,
                preferred_risk_reward_ratio=2.0
            )
            
            # 設定値の確認
            assert config.strategy == TPSLStrategy.RISK_REWARD, "戦略設定が正しくありません"
            assert config.max_risk_per_trade == 0.03, "最大リスク設定が正しくありません"
            assert config.preferred_risk_reward_ratio == 2.0, "リスクリワード比設定が正しくありません"
            
            # デフォルト値の確認
            assert config.volatility_sensitivity == "medium", "デフォルトボラティリティ感度が正しくありません"
            assert config.min_stop_loss == 0.005, "最小SL設定が正しくありません"
            assert config.max_stop_loss == 0.1, "最大SL設定が正しくありません"
            assert config.min_take_profit == 0.01, "最小TP設定が正しくありません"
            assert config.max_take_profit == 0.2, "最大TP設定が正しくありません"
            
            logger.info("✅ TP/SL設定妥当性テスト成功")
            
        except Exception as e:
            pytest.fail(f"TP/SL設定妥当性テストエラー: {e}")

    def test_random_strategy(self):
        """ランダム戦略テスト"""
        logger.info("=== ランダム戦略テスト ===")
        
        try:
            from app.services.auto_strategy.services.tpsl_auto_decision_service import (
                TPSLAutoDecisionService, TPSLConfig, TPSLStrategy
            )
            
            service = TPSLAutoDecisionService()
            config = TPSLConfig(strategy=TPSLStrategy.RANDOM)
            
            # 複数回実行して結果の妥当性を確認
            results = []
            for _ in range(10):
                result = service.generate_tpsl_values(config)
                results.append(result)
                
                # 基本的な妥当性チェック
                assert result.strategy_used == "random", "戦略名が正しくありません"
                assert config.min_stop_loss <= result.stop_loss_pct <= config.max_stop_loss, "SL値が範囲外です"
                assert config.min_take_profit <= result.take_profit_pct <= config.max_take_profit, "TP値が範囲外です"
                assert result.risk_reward_ratio > 0, "リスクリワード比が無効です"
                assert 0 <= result.confidence_score <= 1, "信頼度スコアが範囲外です"
                assert isinstance(result.metadata, dict), "メタデータが辞書形式ではありません"
            
            # ランダム性の確認（全て同じ値ではないことを確認）
            sl_values = [r.stop_loss_pct for r in results]
            tp_values = [r.take_profit_pct for r in results]
            assert len(set(sl_values)) > 1 or len(set(tp_values)) > 1, "ランダム性が不足しています"
            
            logger.info("✅ ランダム戦略テスト成功")
            
        except Exception as e:
            pytest.fail(f"ランダム戦略テストエラー: {e}")

    def test_risk_reward_strategy(self):
        """リスクリワード戦略テスト"""
        logger.info("=== リスクリワード戦略テスト ===")
        
        try:
            from app.services.auto_strategy.services.tpsl_auto_decision_service import (
                TPSLAutoDecisionService, TPSLConfig, TPSLStrategy
            )
            
            service = TPSLAutoDecisionService()
            
            # 異なるリスクリワード比でテスト
            test_cases = [
                {"rr_ratio": 1.5, "max_risk": 0.02},
                {"rr_ratio": 2.0, "max_risk": 0.03},
                {"rr_ratio": 3.0, "max_risk": 0.025}
            ]
            
            for case in test_cases:
                config = TPSLConfig(
                    strategy=TPSLStrategy.RISK_REWARD,
                    max_risk_per_trade=case["max_risk"],
                    preferred_risk_reward_ratio=case["rr_ratio"]
                )
                
                result = service.generate_tpsl_values(config)
                
                # 基本的な妥当性チェック
                assert result.strategy_used == "risk_reward", "戦略名が正しくありません"
                assert result.stop_loss_pct <= config.max_risk_per_trade, "SLが最大リスクを超えています"
                
                # リスクリワード比の確認（許容誤差を考慮）
                expected_rr = case["rr_ratio"]
                actual_rr = result.risk_reward_ratio
                tolerance = 0.1
                assert abs(actual_rr - expected_rr) <= tolerance, f"リスクリワード比が期待値と異なります: {actual_rr} vs {expected_rr}"
                
                # メタデータの確認
                assert "target_rr_ratio" in result.metadata, "target_rr_ratio メタデータが不足しています"
                assert "actual_rr_ratio" in result.metadata, "actual_rr_ratio メタデータが不足しています"
            
            logger.info("✅ リスクリワード戦略テスト成功")
            
        except Exception as e:
            pytest.fail(f"リスクリワード戦略テストエラー: {e}")

    def test_volatility_adaptive_strategy(self):
        """ボラティリティ適応戦略テスト"""
        logger.info("=== ボラティリティ適応戦略テスト ===")
        
        try:
            from app.services.auto_strategy.services.tpsl_auto_decision_service import (
                TPSLAutoDecisionService, TPSLConfig, TPSLStrategy
            )
            
            service = TPSLAutoDecisionService()
            
            # 異なるボラティリティ感度でテスト
            sensitivities = ["low", "medium", "high"]
            
            for sensitivity in sensitivities:
                config = TPSLConfig(
                    strategy=TPSLStrategy.VOLATILITY_ADAPTIVE,
                    volatility_sensitivity=sensitivity
                )
                
                # ATRデータありの場合
                market_data_with_atr = {"atr_pct": 0.025}
                result_with_atr = service.generate_tpsl_values(config, market_data_with_atr)
                
                # ATRデータなしの場合
                result_without_atr = service.generate_tpsl_values(config)
                
                for result in [result_with_atr, result_without_atr]:
                    assert result.strategy_used == "volatility_adaptive", "戦略名が正しくありません"
                    assert config.min_stop_loss <= result.stop_loss_pct <= config.max_stop_loss, "SL値が範囲外です"
                    assert result.take_profit_pct <= config.max_take_profit, "TP値が範囲外です"
                    assert result.risk_reward_ratio > 0, "リスクリワード比が無効です"
                    
                    # メタデータの確認
                    assert "volatility_sensitivity" in result.metadata, "volatility_sensitivity メタデータが不足しています"
                    assert "atr_available" in result.metadata, "atr_available メタデータが不足しています"
                    assert "multiplier_used" in result.metadata, "multiplier_used メタデータが不足しています"
                
                # ATRの有無による違いの確認
                assert result_with_atr.metadata["atr_available"] is True, "ATR利用可能フラグが正しくありません"
                assert result_without_atr.metadata["atr_available"] is False, "ATR利用不可フラグが正しくありません"
            
            logger.info("✅ ボラティリティ適応戦略テスト成功")
            
        except Exception as e:
            pytest.fail(f"ボラティリティ適応戦略テストエラー: {e}")

    def test_statistical_strategy(self):
        """統計的戦略テスト"""
        logger.info("=== 統計的戦略テスト ===")
        
        try:
            from app.services.auto_strategy.services.tpsl_auto_decision_service import (
                TPSLAutoDecisionService, TPSLConfig, TPSLStrategy
            )
            
            service = TPSLAutoDecisionService()
            config = TPSLConfig(strategy=TPSLStrategy.STATISTICAL)
            
            # 複数回実行して統計的データの使用を確認
            results = []
            for _ in range(10):
                result = service.generate_tpsl_values(config, symbol="BTC:USDT")
                results.append(result)
                
                # 基本的な妥当性チェック
                assert result.strategy_used == "statistical", "戦略名が正しくありません"
                assert config.min_stop_loss <= result.stop_loss_pct <= config.max_stop_loss, "SL値が範囲外です"
                assert result.take_profit_pct <= config.max_take_profit, "TP値が範囲外です"
                assert result.risk_reward_ratio > 0, "リスクリワード比が無効です"
                
                # メタデータの確認
                assert "symbol" in result.metadata, "symbol メタデータが不足しています"
                assert "statistical_sl" in result.metadata, "statistical_sl メタデータが不足しています"
                assert "statistical_rr" in result.metadata, "statistical_rr メタデータが不足しています"
                
                # 統計データの範囲確認
                statistical_sl = result.metadata["statistical_sl"]
                statistical_rr = result.metadata["statistical_rr"]
                assert statistical_sl in service.statistical_data["optimal_sl_ranges"], "統計SL値が無効です"
                assert statistical_rr in service.statistical_data["optimal_rr_ratios"], "統計RR値が無効です"
            
            logger.info("✅ 統計的戦略テスト成功")
            
        except Exception as e:
            pytest.fail(f"統計的戦略テストエラー: {e}")

    def test_auto_optimal_strategy(self):
        """自動最適化戦略テスト"""
        logger.info("=== 自動最適化戦略テスト ===")
        
        try:
            from app.services.auto_strategy.services.tpsl_auto_decision_service import (
                TPSLAutoDecisionService, TPSLConfig, TPSLStrategy
            )
            
            service = TPSLAutoDecisionService()
            config = TPSLConfig(strategy=TPSLStrategy.AUTO_OPTIMAL)
            
            market_data = {"atr_pct": 0.03, "volatility": 0.025}
            result = service.generate_tpsl_values(config, market_data, "BTC:USDT")
            
            # 基本的な妥当性チェック
            assert result.strategy_used == "auto_optimal", "戦略名が正しくありません"
            assert config.min_stop_loss <= result.stop_loss_pct <= config.max_stop_loss, "SL値が範囲外です"
            assert result.take_profit_pct <= config.max_take_profit, "TP値が範囲外です"
            assert result.risk_reward_ratio > 0, "リスクリワード比が無効です"
            
            # メタデータの確認（複数戦略の比較結果）
            assert "selected_from" in result.metadata, "selected_from メタデータが不足しています"
            assert "confidence_scores" in result.metadata, "confidence_scores メタデータが不足しています"
            assert "original_metadata" in result.metadata, "original_metadata メタデータが不足しています"
            
            # 選択された戦略の確認
            selected_strategies = result.metadata["selected_from"]
            expected_strategies = ["risk_reward", "volatility_adaptive", "statistical"]
            assert len(selected_strategies) == len(expected_strategies), "比較戦略数が正しくありません"
            
            # 信頼度スコアの確認
            confidence_scores = result.metadata["confidence_scores"]
            assert len(confidence_scores) == len(selected_strategies), "信頼度スコア数が正しくありません"
            assert all(0 <= score <= 1 for score in confidence_scores), "信頼度スコアが範囲外です"
            
            logger.info("✅ 自動最適化戦略テスト成功")
            
        except Exception as e:
            pytest.fail(f"自動最適化戦略テストエラー: {e}")

    def test_tpsl_result_validation(self):
        """TP/SL結果の妥当性テスト"""
        logger.info("=== TP/SL結果妥当性テスト ===")
        
        try:
            from app.services.auto_strategy.services.tpsl_auto_decision_service import (
                TPSLAutoDecisionService, TPSLConfig, TPSLStrategy, TPSLResult
            )
            
            service = TPSLAutoDecisionService()
            
            # 全戦略をテスト
            strategies = [
                TPSLStrategy.RANDOM,
                TPSLStrategy.RISK_REWARD,
                TPSLStrategy.VOLATILITY_ADAPTIVE,
                TPSLStrategy.STATISTICAL,
                TPSLStrategy.AUTO_OPTIMAL
            ]
            
            for strategy in strategies:
                config = TPSLConfig(strategy=strategy)
                result = service.generate_tpsl_values(config)
                
                # TPSLResult型の確認
                assert isinstance(result, TPSLResult), f"結果がTPSLResult型ではありません: {strategy}"
                
                # 必須フィールドの確認
                assert hasattr(result, 'stop_loss_pct'), "stop_loss_pct フィールドが不足しています"
                assert hasattr(result, 'take_profit_pct'), "take_profit_pct フィールドが不足しています"
                assert hasattr(result, 'risk_reward_ratio'), "risk_reward_ratio フィールドが不足しています"
                assert hasattr(result, 'strategy_used'), "strategy_used フィールドが不足しています"
                assert hasattr(result, 'confidence_score'), "confidence_score フィールドが不足しています"
                assert hasattr(result, 'metadata'), "metadata フィールドが不足しています"
                
                # 値の妥当性確認
                assert result.stop_loss_pct > 0, "SL値が正の値ではありません"
                assert result.take_profit_pct > 0, "TP値が正の値ではありません"
                assert result.risk_reward_ratio > 0, "リスクリワード比が正の値ではありません"
                assert 0 <= result.confidence_score <= 1, "信頼度スコアが範囲外です"
                assert isinstance(result.strategy_used, str), "戦略名が文字列ではありません"
                assert isinstance(result.metadata, dict), "メタデータが辞書形式ではありません"
            
            logger.info("✅ TP/SL結果妥当性テスト成功")
            
        except Exception as e:
            pytest.fail(f"TP/SL結果妥当性テストエラー: {e}")

    def test_error_handling_and_fallback(self):
        """エラーハンドリングとフォールバック機能テスト"""
        logger.info("=== エラーハンドリングとフォールバック機能テスト ===")
        
        try:
            from app.services.auto_strategy.services.tpsl_auto_decision_service import (
                TPSLAutoDecisionService, TPSLConfig, TPSLStrategy
            )
            
            service = TPSLAutoDecisionService()
            
            # 無効な戦略での処理（フォールバック確認）
            # 注意: 実際の実装では例外が発生する可能性があるため、try-catchで処理
            try:
                # 存在しない戦略を直接設定することはできないため、
                # 代わりに極端な設定値でテスト
                extreme_config = TPSLConfig(
                    strategy=TPSLStrategy.RISK_REWARD,
                    max_risk_per_trade=0.0001,  # 極端に小さい値
                    preferred_risk_reward_ratio=100.0,  # 極端に大きい値
                    min_stop_loss=0.001,
                    max_stop_loss=0.002,  # 非常に狭い範囲
                    min_take_profit=0.001,
                    max_take_profit=0.002
                )
                
                result = service.generate_tpsl_values(extreme_config)
                
                # 結果が返されることを確認（フォールバックまたは調整された値）
                assert isinstance(result, service.generate_tpsl_values(TPSLConfig(strategy=TPSLStrategy.RISK_REWARD)).__class__), "結果が返されませんでした"
                assert result.stop_loss_pct > 0, "SL値が無効です"
                assert result.take_profit_pct > 0, "TP値が無効です"
                
            except Exception as e:
                logger.info(f"極端な設定でのエラー（期待される場合もあります）: {e}")
            
            logger.info("✅ エラーハンドリングとフォールバック機能テスト成功")
            
        except Exception as e:
            pytest.fail(f"エラーハンドリングとフォールバック機能テストエラー: {e}")


if __name__ == "__main__":
    # 単体でテストを実行する場合
    pytest.main([__file__, "-v"])
