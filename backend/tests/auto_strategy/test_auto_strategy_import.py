#!/usr/bin/env python3
"""
自動戦略モジュールのインポートテスト
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

def test_imports():
    """インポートテスト"""
    print("🔍 自動戦略モジュールインポートテスト")
    print("=" * 50)
    
    try:
        print("1. 基本モジュールのインポート...")
        from app.core.services.auto_strategy.models.strategy_gene import StrategyGene
        print("  ✅ StrategyGene")
        
        from app.core.services.auto_strategy.models.ga_config import GAConfig
        print("  ✅ GAConfig")
        
        from app.core.services.auto_strategy.factories.strategy_factory import StrategyFactory
        print("  ✅ StrategyFactory")
        
        from app.core.services.auto_strategy.engines.ga_engine import GeneticAlgorithmEngine
        print("  ✅ GeneticAlgorithmEngine")
        
        from app.core.services.auto_strategy.services.auto_strategy_service import AutoStrategyService
        print("  ✅ AutoStrategyService")
        
        print("\n2. APIモジュールのインポート...")
        from app.api.auto_strategy import router
        print("  ✅ auto_strategy router")
        
        print("\n3. メインアプリのインポート...")
        from app.main import app
        print("  ✅ main app")
        
        print("\n4. ルーター確認...")
        routes = [route.path for route in app.routes]
        auto_strategy_routes = [route for route in routes if 'auto-strategy' in route]
        print(f"  自動戦略ルート数: {len(auto_strategy_routes)}")
        for route in auto_strategy_routes:
            print(f"    {route}")
        
        if len(auto_strategy_routes) == 0:
            print("  ⚠️ 自動戦略ルートが見つかりません")
            
            # ルーター詳細確認
            print("\n5. 詳細ルーター確認...")
            for route in app.routes:
                if hasattr(route, 'path'):
                    print(f"    {route.path}")
        
        print("\n✅ 全てのインポートが成功しました")
        return True
        
    except Exception as e:
        print(f"\n❌ インポートエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_imports()
