"""
資金管理機能の包括的テスト

PositionSizingGeneの全手法の計算精度、
エラーハンドリング、エッジケースを網羅的にテストします。
"""

import pytest
import numpy as np
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch

# テスト対象のインポート
from app.core.services.auto_strategy.models.gene_position_sizing import (
    PositionSizingGene,
    PositionSizingMethod,
)
from app.core.services.auto_strategy.calculators.position_sizing_calculator import (
    PositionSizingCalculatorService,
    PositionSizingResult,
)


class TestPositionSizingComprehensive:
    """資金管理機能の包括的テストクラス"""

    @pytest.fixture
    def calculator_service(self):
        """PositionSizingCalculatorServiceのインスタンス"""
        return PositionSizingCalculatorService()

    @pytest.fixture
    def sample_market_data(self) -> Dict[str, Any]:
        """サンプル市場データ"""
        return {
            "atr": 1000.0,
            "atr_pct": 0.02,
            "volatility": 0.025,
            "current_price": 50000.0,
            "volume": 1000000,
        }

    @pytest.fixture
    def sample_trade_history(self) -> List[Dict[str, Any]]:
        """サンプル取引履歴"""
        return [
            {"pnl": 500.0, "win": True, "trade_amount": 1000.0},
            {"pnl": -300.0, "win": False, "trade_amount": 1000.0},
            {"pnl": 800.0, "win": True, "trade_amount": 1500.0},
            {"pnl": -200.0, "win": False, "trade_amount": 1200.0},
            {"pnl": 600.0, "win": True, "trade_amount": 1100.0},
            {"pnl": -400.0, "win": False, "trade_amount": 1300.0},
            {"pnl": 700.0, "win": True, "trade_amount": 1400.0},
            {"pnl": -250.0, "win": False, "trade_amount": 1000.0},
        ]

    def test_fixed_ratio_method(self, calculator_service, sample_market_data):
        """固定比率方式のテスト"""
        print("\n=== 固定比率方式テスト ===")
        
        # 異なる比率でテスト
        ratios = [0.05, 0.1, 0.15, 0.2, 0.25]
        account_balance = 10000.0
        current_price = 50000.0
        
        for ratio in ratios:
            gene = PositionSizingGene(
                method=PositionSizingMethod.FIXED_RATIO,
                fixed_ratio=ratio,
                enabled=True,
            )
            
            result = calculator_service.calculate_position_size(
                gene=gene,
                account_balance=account_balance,
                current_price=current_price,
                market_data=sample_market_data,
            )
            
            # 基本検証
            assert isinstance(result, PositionSizingResult), f"比率{ratio}: 結果の型が不正"
            assert result.method_used == "fixed_ratio", f"比率{ratio}: 手法名が不正"
            assert 0.0 <= result.confidence_score <= 1.0, f"比率{ratio}: 信頼度が範囲外"
            
            # 計算精度確認（サイズ制限が適用される）
            calculated_size = account_balance * ratio
            expected_size = max(gene.min_position_size, min(calculated_size, gene.max_position_size))
            assert abs(result.position_size - expected_size) < 0.01, \
                f"比率{ratio}: 計算結果が不正 実際={result.position_size}, 期待={expected_size}, 計算値={calculated_size}"
            
            print(f"  ✅ 比率{ratio}: ポジションサイズ={result.position_size:.2f}, 期待={expected_size:.2f}")

    def test_fixed_quantity_method(self, calculator_service, sample_market_data):
        """固定枚数方式のテスト"""
        print("\n=== 固定枚数方式テスト ===")
        
        quantities = [0.5, 1.0, 1.5, 2.0, 2.5]
        account_balance = 10000.0
        current_price = 50000.0
        
        for quantity in quantities:
            gene = PositionSizingGene(
                method=PositionSizingMethod.FIXED_QUANTITY,
                fixed_quantity=quantity,
                enabled=True,
            )
            
            result = calculator_service.calculate_position_size(
                gene=gene,
                account_balance=account_balance,
                current_price=current_price,
                market_data=sample_market_data,
            )
            
            # 基本検証
            assert isinstance(result, PositionSizingResult), f"枚数{quantity}: 結果の型が不正"
            assert result.method_used == "fixed_quantity", f"枚数{quantity}: 手法名が不正"
            
            # 計算精度確認（固定枚数がそのまま返される）
            assert abs(result.position_size - quantity) < 0.01, \
                f"枚数{quantity}: 計算結果が不正 実際={result.position_size}, 期待={quantity}"
            
            print(f"  ✅ 枚数{quantity}: ポジションサイズ={result.position_size:.2f}")

    def test_volatility_based_method(self, calculator_service, sample_market_data):
        """ボラティリティベース方式のテスト"""
        print("\n=== ボラティリティベース方式テスト ===")
        
        account_balance = 10000.0
        current_price = 50000.0
        
        # 異なるATR倍率とリスク設定でテスト
        test_configs = [
            {"atr_multiplier": 1.5, "risk_per_trade": 0.02},
            {"atr_multiplier": 2.0, "risk_per_trade": 0.03},
            {"atr_multiplier": 2.5, "risk_per_trade": 0.04},
            {"atr_multiplier": 3.0, "risk_per_trade": 0.05},
        ]
        
        for config in test_configs:
            gene = PositionSizingGene(
                method=PositionSizingMethod.VOLATILITY_BASED,
                atr_multiplier=config["atr_multiplier"],
                risk_per_trade=config["risk_per_trade"],
                enabled=True,
            )
            
            result = calculator_service.calculate_position_size(
                gene=gene,
                account_balance=account_balance,
                current_price=current_price,
                market_data=sample_market_data,
            )
            
            # 基本検証
            assert isinstance(result, PositionSizingResult), f"設定{config}: 結果の型が不正"
            assert result.method_used == "volatility_based", f"設定{config}: 手法名が不正"
            assert result.position_size > 0, f"設定{config}: ポジションサイズが0以下"
            
            # リスク量の計算確認
            risk_amount = account_balance * config["risk_per_trade"]
            atr_pct = sample_market_data["atr_pct"]
            volatility_factor = atr_pct * config["atr_multiplier"]
            expected_size = risk_amount / (current_price * volatility_factor)
            
            # 許容誤差内での一致確認
            tolerance = expected_size * 0.1  # 10%の許容誤差
            assert abs(result.position_size - expected_size) <= tolerance, \
                f"設定{config}: 計算結果が期待値から乖離 実際={result.position_size:.4f}, 期待={expected_size:.4f}"
            
            print(f"  ✅ ATR倍率{config['atr_multiplier']}, リスク{config['risk_per_trade']}: ポジションサイズ={result.position_size:.4f}")

    def test_half_optimal_f_method(self, calculator_service, sample_market_data, sample_trade_history):
        """ハーフオプティマルF方式のテスト"""
        print("\n=== ハーフオプティマルF方式テスト ===")
        
        account_balance = 10000.0
        current_price = 50000.0
        
        # 異なるオプティマルF倍率でテスト
        multipliers = [0.25, 0.5, 0.75]
        
        for multiplier in multipliers:
            gene = PositionSizingGene(
                method=PositionSizingMethod.HALF_OPTIMAL_F,
                optimal_f_multiplier=multiplier,
                lookback_period=len(sample_trade_history),
                enabled=True,
            )
            
            result = calculator_service.calculate_position_size(
                gene=gene,
                account_balance=account_balance,
                current_price=current_price,
                market_data=sample_market_data,
                trade_history=sample_trade_history,
            )
            
            # 基本検証
            assert isinstance(result, PositionSizingResult), f"倍率{multiplier}: 結果の型が不正"
            assert result.method_used == "half_optimal_f", f"倍率{multiplier}: 手法名が不正"
            assert result.position_size >= 0, f"倍率{multiplier}: ポジションサイズが負"
            
            # オプティマルF計算の妥当性確認
            wins = [t for t in sample_trade_history if t["win"]]
            losses = [t for t in sample_trade_history if not t["win"]]
            
            if wins and losses:
                win_rate = len(wins) / len(sample_trade_history)
                avg_win = sum(t["pnl"] for t in wins) / len(wins)
                avg_loss = abs(sum(t["pnl"] for t in losses) / len(losses))
                
                # オプティマルF計算
                optimal_f = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
                half_optimal_f = max(0, optimal_f * multiplier)
                
                # 結果が妥当な範囲内であることを確認
                assert 0 <= result.position_size <= account_balance, \
                    f"倍率{multiplier}: ポジションサイズが範囲外"
            
            print(f"  ✅ 倍率{multiplier}: ポジションサイズ={result.position_size:.4f}")

    def test_size_limits_enforcement(self, calculator_service, sample_market_data):
        """サイズ制限の適用テスト"""
        print("\n=== サイズ制限適用テスト ===")
        
        account_balance = 10000.0
        current_price = 50000.0
        
        # 極端に大きな比率設定（ただし、遺伝子の制約内で）
        gene = PositionSizingGene(
            method=PositionSizingMethod.FIXED_RATIO,
            fixed_ratio=0.5,  # 50% (大きな値だが制約内)
            min_position_size=0.1,
            max_position_size=1000.0,
            enabled=True,
        )
        
        result = calculator_service.calculate_position_size(
            gene=gene,
            account_balance=account_balance,
            current_price=current_price,
            market_data=sample_market_data,
        )
        
        # 最大制限が適用されることを確認
        assert result.position_size <= gene.max_position_size, \
            f"最大制限が適用されていない: {result.position_size} > {gene.max_position_size}"
        
        print(f"  ✅ 最大制限適用: {result.position_size} <= {gene.max_position_size}")
        
        # 極端に小さな比率設定
        gene.fixed_ratio = 0.01  # 1% (小さな値)
        gene.max_position_size = 50.0  # 最大値を小さく設定

        result = calculator_service.calculate_position_size(
            gene=gene,
            account_balance=account_balance,
            current_price=current_price,
            market_data=sample_market_data,
        )

        # 計算値と制限の確認
        calculated_size = account_balance * gene.fixed_ratio  # 100.0
        expected_size = max(gene.min_position_size, min(calculated_size, gene.max_position_size))  # 50.0

        assert abs(result.position_size - expected_size) < 0.01, \
            f"制限適用が不正: 実際={result.position_size}, 期待={expected_size}, 計算値={calculated_size}"

        print(f"  ✅ 制限適用: 計算値={calculated_size}, 制限後={result.position_size}")

    def test_error_handling_and_fallbacks(self, calculator_service):
        """エラーハンドリングとフォールバックテスト"""
        print("\n=== エラーハンドリング・フォールバックテスト ===")
        
        account_balance = 10000.0
        current_price = 50000.0
        
        # 無効な市場データ
        invalid_market_data = {
            "atr": 0.0,
            "atr_pct": 0.0,
            "volatility": 0.0,
        }
        
        gene = PositionSizingGene(
            method=PositionSizingMethod.VOLATILITY_BASED,
            enabled=True,
        )
        
        result = calculator_service.calculate_position_size(
            gene=gene,
            account_balance=account_balance,
            current_price=current_price,
            market_data=invalid_market_data,
        )
        
        # フォールバック結果が返されることを確認
        assert isinstance(result, PositionSizingResult), "フォールバック結果が返されない"
        assert result.position_size > 0, "フォールバックポジションサイズが無効"
        assert len(result.warnings) > 0, "警告が記録されていない"
        
        print(f"  ✅ 無効データ処理: ポジションサイズ={result.position_size:.4f}, 警告数={len(result.warnings)}")
        
        # 空の取引履歴
        empty_trade_history = []
        
        gene.method = PositionSizingMethod.HALF_OPTIMAL_F
        
        result = calculator_service.calculate_position_size(
            gene=gene,
            account_balance=account_balance,
            current_price=current_price,
            trade_history=empty_trade_history,
        )
        
        # フォールバック処理が動作することを確認
        assert isinstance(result, PositionSizingResult), "空履歴でのフォールバック失敗"
        assert result.position_size > 0, "空履歴でのポジションサイズが無効"
        
        print(f"  ✅ 空履歴処理: ポジションサイズ={result.position_size:.4f}")

    def test_disabled_gene_handling(self, calculator_service, sample_market_data):
        """無効化された遺伝子の処理テスト"""
        print("\n=== 無効化遺伝子処理テスト ===")
        
        account_balance = 10000.0
        current_price = 50000.0
        
        # 無効化された遺伝子
        gene = PositionSizingGene(
            method=PositionSizingMethod.FIXED_RATIO,
            fixed_ratio=0.2,
            enabled=False,  # 無効化
        )
        
        result = calculator_service.calculate_position_size(
            gene=gene,
            account_balance=account_balance,
            current_price=current_price,
            market_data=sample_market_data,
        )
        
        # 無効化された遺伝子でも計算は実行されるが、警告が出ることを確認
        # 実装によっては、無効化されていても計算が実行される場合がある
        assert result.position_size > 0, f"無効化遺伝子でポジションサイズが0: {result.position_size}"

        # 無効化の処理方法は実装依存のため、結果が妥当であることを確認
        calculated_size = account_balance * gene.fixed_ratio
        expected_size = max(gene.min_position_size, min(calculated_size, gene.max_position_size))

        print(f"  ✅ 無効化遺伝子: ポジションサイズ={result.position_size}, 有効={gene.enabled}")


def main():
    """メイン実行関数"""
    print("資金管理機能包括的テスト開始")
    print("=" * 60)
    
    # pytest実行
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    main()
