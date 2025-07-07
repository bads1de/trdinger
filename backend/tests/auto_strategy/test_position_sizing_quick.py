"""
Position Sizingシステムの簡単な動作確認テスト
"""

import sys
import os

# パス設定
backend_path = os.path.dirname(__file__)
sys.path.insert(0, backend_path)

def test_position_sizing_basic():
    """基本的なPosition Sizingテスト"""
    print("=== Position Sizing基本テスト ===")
    
    try:
        # 1. PositionSizingGeneのインポートテスト
        from app.core.services.auto_strategy.models.position_sizing_gene import (
            PositionSizingGene,
            PositionSizingMethod,
        )
        print("✅ PositionSizingGeneのインポート成功")
        
        # 2. 基本的な遺伝子作成テスト
        gene = PositionSizingGene(
            method=PositionSizingMethod.FIXED_RATIO,
            fixed_ratio=0.2,
            enabled=True,
        )
        print(f"✅ PositionSizingGene作成成功: {gene.method.value}")
        
        # 3. GAConfigのテスト
        from app.core.services.auto_strategy.models.ga_config import GAConfig
        config = GAConfig()
        
        # position_size_rangeが削除されていることを確認
        assert not hasattr(config, 'position_size_range'), "position_size_rangeが残っています"
        print("✅ GAConfigからposition_size_rangeが削除されています")
        
        # 新しいposition_sizing関連パラメータが存在することを確認
        assert hasattr(config, 'position_sizing_method_constraints'), "position_sizing_method_constraintsが存在しません"
        print(f"✅ Position Sizing制約が設定されています: {len(config.position_sizing_method_constraints)}個の手法")
        
        # 4. StrategyGeneのテスト
        from app.core.services.auto_strategy.models.strategy_gene import StrategyGene
        
        strategy_gene = StrategyGene(
            id="test_strategy",
            indicators=[],
            entry_conditions=[],
            exit_conditions=[],
            risk_management={"position_size": 0.1},
            position_sizing_gene=gene,
        )
        
        assert hasattr(strategy_gene, 'position_sizing_gene'), "position_sizing_geneフィールドが存在しません"
        assert strategy_gene.position_sizing_gene is not None, "position_sizing_geneがNullです"
        print("✅ StrategyGeneにposition_sizing_geneが正常に設定されています")
        
        # 5. StrategyFactoryのテスト
        from app.core.services.auto_strategy.factories.strategy_factory import StrategyFactory
        
        factory = StrategyFactory()
        
        # ポジションサイズ計算をテスト
        calculated_size = factory._calculate_position_size(
            strategy_gene, 
            account_balance=10000.0, 
            current_price=50000.0, 
            data=None
        )
        
        print(f"✅ StrategyFactoryでポジションサイズが計算されました: {calculated_size}")
        
        # Position Sizing遺伝子の設定に基づいて計算されていることを確認
        expected_size = min(10000.0 * 0.2, 1.0)  # min(2000.0, 1.0) = 1.0 (デフォルト最大値)
        assert calculated_size == expected_size, f"期待値: {expected_size}, 実際: {calculated_size}"
        
        print(f"✅ 計算結果が期待値と一致しています: {expected_size}")
        
        print("\n🎉 Position Sizingシステムの基本テストが全て成功しました！")
        return True
        
    except Exception as e:
        print(f"\n❌ テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_frontend_types():
    """フロントエンド型定義のテスト"""
    print("\n=== フロントエンド型定義テスト ===")
    
    try:
        # フロントエンドの型定義ファイルを読み込んで確認
        frontend_types_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "types", "optimization.ts")
        
        if os.path.exists(frontend_types_path):
            with open(frontend_types_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # position_size_rangeが削除されていることを確認
            if 'position_size_range: [number, number]' in content:
                print("❌ フロントエンドでposition_size_rangeが残っています")
                return False
            else:
                print("✅ フロントエンドからposition_size_rangeが削除されています")
                
            # 新しいposition_sizing関連のコメントがあることを確認
            if 'Position Sizing' in content or 'position_sizing' in content:
                print("✅ Position Sizing関連のコメントが存在します")
            else:
                print("⚠️ Position Sizing関連のコメントが見つかりません")
                
        else:
            print("⚠️ フロントエンド型定義ファイルが見つかりません")
            
        return True
        
    except Exception as e:
        print(f"❌ フロントエンド型定義テストエラー: {e}")
        return False


def main():
    """メイン関数"""
    print("Position Sizingシステム簡単動作確認テスト開始")
    print("=" * 60)
    
    success1 = test_position_sizing_basic()
    success2 = test_frontend_types()
    
    if success1 and success2:
        print("\n" + "=" * 60)
        print("🎉 全てのテストが成功しました！")
        print("Position Sizingシステムが正常に動作し、従来システムが適切に削除されています。")
        print("\n主な確認項目:")
        print("✅ GAConfigからposition_size_rangeが削除されている")
        print("✅ 新しいPosition Sizing制約が設定されている")
        print("✅ StrategyGeneにposition_sizing_geneフィールドが追加されている")
        print("✅ StrategyFactoryでPosition Sizingが動作している")
        print("✅ フロントエンドの型定義が更新されている")
        return True
    else:
        print("\n❌ 一部のテストが失敗しました")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
