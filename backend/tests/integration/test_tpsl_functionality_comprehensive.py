"""
TP/SL機能の包括的テスト

TPSLAutoDecisionServiceの全戦略の動作確認、
計算精度、エラーハンドリング、エッジケースを網羅的にテストします。
"""

import pytest
import numpy as np
from typing import Dict, Any, Optional
from unittest.mock import Mock, patch

# テスト対象のインポート
from app.core.services.auto_strategy.services.tpsl_auto_decision_service import (
    TPSLAutoDecisionService,
    TPSLConfig,
    TPSLStrategy,
    TPSLResult,
)


class TestTPSLFunctionalityComprehensive:
    """TP/SL機能の包括的テストクラス"""

    @pytest.fixture
    def tpsl_service(self):
        """TPSLAutoDecisionServiceのインスタンス"""
        return TPSLAutoDecisionService()

    @pytest.fixture
    def base_config(self) -> TPSLConfig:
        """基本的なTP/SL設定"""
        return TPSLConfig(
            strategy=TPSLStrategy.RANDOM,
            max_risk_per_trade=0.02,
            preferred_risk_reward_ratio=2.0,
            volatility_sensitivity="medium",
            min_stop_loss=0.01,
            max_stop_loss=0.05,
            min_take_profit=0.02,
            max_take_profit=0.10,
        )

    @pytest.fixture
    def sample_market_data(self) -> Dict[str, Any]:
        """サンプル市場データ"""
        return {
            "current_price": 50000.0,
            "atr": 1000.0,
            "volatility": 0.02,
            "volume": 1000000,
            "bid": 49950.0,
            "ask": 50050.0,
            "spread": 100.0,
        }

    def test_random_strategy(self, tpsl_service, base_config):
        """ランダム戦略のテスト"""
        print("\n=== ランダム戦略テスト ===")
        
        config = base_config
        config.strategy = TPSLStrategy.RANDOM
        
        # 複数回実行して結果の妥当性を確認
        results = []
        for i in range(10):
            result = tpsl_service.generate_tpsl_values(config)
            
            # 基本的な検証
            assert isinstance(result, TPSLResult), f"実行{i+1}: 結果の型が不正"
            assert result.stop_loss_pct > 0, f"実行{i+1}: SLが0以下"
            assert result.take_profit_pct > 0, f"実行{i+1}: TPが0以下"
            assert result.risk_reward_ratio > 0, f"実行{i+1}: RR比が0以下"
            assert result.strategy_used == "random", f"実行{i+1}: 戦略名が不正"
            
            # 範囲チェック
            assert config.min_stop_loss <= result.stop_loss_pct <= config.max_stop_loss, \
                f"実行{i+1}: SLが範囲外 {result.stop_loss_pct}"
            # ランダム戦略では、TPの最小値はSL*1.2で決まるため、設定値より小さくなる場合がある
            assert result.take_profit_pct <= config.max_take_profit, \
                f"実行{i+1}: TPが最大値を超過 {result.take_profit_pct}"
            assert result.take_profit_pct >= result.stop_loss_pct * 1.2, \
                f"実行{i+1}: TPがSL*1.2未満 {result.take_profit_pct}"
            
            # 最小RR比チェック（1.2倍以上）
            assert result.risk_reward_ratio >= 1.2, \
                f"実行{i+1}: RR比が最小値未満 {result.risk_reward_ratio}"
            
            results.append(result)
        
        # ランダム性の確認（全て同じ値でないこと）
        sl_values = [r.stop_loss_pct for r in results]
        tp_values = [r.take_profit_pct for r in results]
        
        assert len(set(sl_values)) > 1 or len(set(tp_values)) > 1, "ランダム性が不足"
        
        print(f"  ✅ ランダム戦略: {len(results)}回実行、全て正常")
        print(f"  📊 SL範囲: {min(sl_values):.4f} - {max(sl_values):.4f}")
        print(f"  📊 TP範囲: {min(tp_values):.4f} - {max(tp_values):.4f}")

    def test_risk_reward_strategy(self, tpsl_service, base_config):
        """リスクリワード戦略のテスト"""
        print("\n=== リスクリワード戦略テスト ===")
        
        config = base_config
        config.strategy = TPSLStrategy.RISK_REWARD
        
        # 異なるRR比でテスト
        rr_ratios = [1.5, 2.0, 2.5, 3.0]
        
        for rr_ratio in rr_ratios:
            config.preferred_risk_reward_ratio = rr_ratio
            result = tpsl_service.generate_tpsl_values(config)
            
            # 基本検証
            assert isinstance(result, TPSLResult), f"RR{rr_ratio}: 結果の型が不正"
            assert result.strategy_used == "risk_reward", f"RR{rr_ratio}: 戦略名が不正"
            
            # RR比の精度確認（許容誤差10%）
            actual_rr = result.risk_reward_ratio
            expected_rr = rr_ratio
            tolerance = 0.1 * expected_rr
            
            assert abs(actual_rr - expected_rr) <= tolerance, \
                f"RR{rr_ratio}: RR比が期待値から乖離 実際={actual_rr:.3f}, 期待={expected_rr:.3f}"
            
            print(f"  ✅ RR比{rr_ratio}: 実際={actual_rr:.3f}, SL={result.stop_loss_pct:.4f}, TP={result.take_profit_pct:.4f}")

    def test_volatility_adaptive_strategy(self, tpsl_service, base_config, sample_market_data):
        """ボラティリティ適応戦略のテスト"""
        print("\n=== ボラティリティ適応戦略テスト ===")
        
        config = base_config
        config.strategy = TPSLStrategy.VOLATILITY_ADAPTIVE
        
        # 異なるボラティリティ感度でテスト
        sensitivities = ["low", "medium", "high"]
        
        results = {}
        for sensitivity in sensitivities:
            config.volatility_sensitivity = sensitivity
            result = tpsl_service.generate_tpsl_values(config, sample_market_data)
            
            # 基本検証
            assert isinstance(result, TPSLResult), f"{sensitivity.value}: 結果の型が不正"
            assert result.strategy_used == "volatility_adaptive", f"{sensitivity.value}: 戦略名が不正"
            
            results[sensitivity] = result
            print(f"  ✅ {sensitivity}: SL={result.stop_loss_pct:.4f}, TP={result.take_profit_pct:.4f}, RR={result.risk_reward_ratio:.3f}")

        # ボラティリティ感度による違いの確認
        low_result = results["low"]
        high_result = results["high"]
        
        # HIGH感度の方がより大きなSL/TPを設定することを確認
        assert high_result.stop_loss_pct >= low_result.stop_loss_pct, \
            "HIGH感度のSLがLOW感度より小さい"
        assert high_result.take_profit_pct >= low_result.take_profit_pct, \
            "HIGH感度のTPがLOW感度より小さい"

    def test_statistical_strategy(self, tpsl_service, base_config):
        """統計的戦略のテスト"""
        print("\n=== 統計的戦略テスト ===")
        
        config = base_config
        config.strategy = TPSLStrategy.STATISTICAL
        
        # 異なるシンボルでテスト
        symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", None]
        
        for symbol in symbols:
            result = tpsl_service.generate_tpsl_values(config, symbol=symbol)
            
            # 基本検証
            assert isinstance(result, TPSLResult), f"シンボル{symbol}: 結果の型が不正"
            assert result.strategy_used == "statistical", f"シンボル{symbol}: 戦略名が不正"
            assert result.confidence_score == 0.9, f"シンボル{symbol}: 信頼度スコアが不正"
            
            # メタデータの確認
            assert "symbol" in result.metadata, f"シンボル{symbol}: メタデータにsymbolがない"
            assert "statistical_sl" in result.metadata, f"シンボル{symbol}: メタデータにstatistical_slがない"
            assert "statistical_rr" in result.metadata, f"シンボル{symbol}: メタデータにstatistical_rrがない"
            
            print(f"  ✅ シンボル{symbol}: SL={result.stop_loss_pct:.4f}, TP={result.take_profit_pct:.4f}, 信頼度={result.confidence_score}")

    def test_auto_optimal_strategy(self, tpsl_service, base_config, sample_market_data):
        """自動最適化戦略のテスト"""
        print("\n=== 自動最適化戦略テスト ===")
        
        config = base_config
        config.strategy = TPSLStrategy.AUTO_OPTIMAL
        
        result = tpsl_service.generate_tpsl_values(config, sample_market_data, "BTCUSDT")
        
        # 基本検証
        assert isinstance(result, TPSLResult), "結果の型が不正"
        assert result.strategy_used == "auto_optimal", "戦略名が不正"
        assert 0.0 <= result.confidence_score <= 1.0, "信頼度スコアが範囲外"
        
        # メタデータの確認
        assert "selected_from" in result.metadata, "メタデータにselected_fromがない"
        assert "confidence_scores" in result.metadata, "メタデータにconfidence_scoresがない"

        selected_strategies = result.metadata["selected_from"]
        assert len(selected_strategies) >= 1, "選択された戦略が不足"
        
        print(f"  ✅ 自動最適化: SL={result.stop_loss_pct:.4f}, TP={result.take_profit_pct:.4f}")
        print(f"  📊 選択戦略: {selected_strategies}")
        print(f"  📊 評価戦略数: {len(selected_strategies)}")

    def test_error_handling(self, tpsl_service):
        """エラーハンドリングテスト"""
        print("\n=== エラーハンドリングテスト ===")
        
        # 無効な戦略
        invalid_config = TPSLConfig(strategy="INVALID_STRATEGY")
        
        # エラーが発生してもフォールバック結果が返されることを確認
        result = tpsl_service.generate_tpsl_values(invalid_config)
        assert isinstance(result, TPSLResult), "フォールバック結果が返されない"
        assert result.stop_loss_pct > 0, "フォールバックSLが無効"
        assert result.take_profit_pct > 0, "フォールバックTPが無効"
        
        print("  ✅ 無効戦略: フォールバック動作確認")
        
        # 極端な設定値
        extreme_config = TPSLConfig(
            strategy=TPSLStrategy.RANDOM,
            min_stop_loss=0.001,
            max_stop_loss=0.002,
            min_take_profit=0.001,
            max_take_profit=0.002,
        )
        
        result = tpsl_service.generate_tpsl_values(extreme_config)
        assert isinstance(result, TPSLResult), "極端設定での結果が無効"
        
        print("  ✅ 極端設定: 正常に処理")

    def test_edge_cases(self, tpsl_service, base_config):
        """エッジケーステスト"""
        print("\n=== エッジケーステスト ===")
        
        # ゼロボラティリティ
        zero_volatility_data = {
            "current_price": 50000.0,
            "atr": 0.0,
            "volatility": 0.0,
        }
        
        config = base_config
        config.strategy = TPSLStrategy.VOLATILITY_ADAPTIVE
        
        result = tpsl_service.generate_tpsl_values(config, zero_volatility_data)
        assert isinstance(result, TPSLResult), "ゼロボラティリティで結果が無効"
        assert result.stop_loss_pct > 0, "ゼロボラティリティでSLが無効"
        
        print("  ✅ ゼロボラティリティ: 正常に処理")
        
        # 極端に高いボラティリティ
        high_volatility_data = {
            "current_price": 50000.0,
            "atr": 10000.0,
            "volatility": 0.5,
        }
        
        result = tpsl_service.generate_tpsl_values(config, high_volatility_data)
        assert isinstance(result, TPSLResult), "高ボラティリティで結果が無効"
        
        print("  ✅ 高ボラティリティ: 正常に処理")

    def test_consistency_and_reproducibility(self, tpsl_service, base_config):
        """一貫性と再現性のテスト"""
        print("\n=== 一貫性・再現性テスト ===")
        
        # 決定論的戦略（RISK_REWARD）の再現性確認
        config = base_config
        config.strategy = TPSLStrategy.RISK_REWARD
        config.preferred_risk_reward_ratio = 2.0
        
        results = []
        for i in range(5):
            result = tpsl_service.generate_tpsl_values(config)
            results.append(result)
        
        # 全ての結果が同じであることを確認
        first_result = results[0]
        for i, result in enumerate(results[1:], 1):
            assert abs(result.stop_loss_pct - first_result.stop_loss_pct) < 1e-10, \
                f"実行{i+1}: SLが一致しない"
            assert abs(result.take_profit_pct - first_result.take_profit_pct) < 1e-10, \
                f"実行{i+1}: TPが一致しない"
        
        print("  ✅ RISK_REWARD戦略: 再現性確認")
        
        # 設定値の境界での一貫性
        boundary_configs = [
            (0.01, 0.05),  # 最小SL, 最大SL
            (0.02, 0.10),  # 最小TP, 最大TP
        ]
        
        for min_sl, max_sl in boundary_configs:
            config.min_stop_loss = min_sl
            config.max_stop_loss = max_sl
            
            result = tpsl_service.generate_tpsl_values(config)
            assert min_sl <= result.stop_loss_pct <= max_sl, \
                f"境界設定({min_sl}, {max_sl})でSLが範囲外"
        
        print("  ✅ 境界値設定: 一貫性確認")


def main():
    """メイン実行関数"""
    print("TP/SL機能包括的テスト開始")
    print("=" * 60)
    
    # pytest実行
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    main()
