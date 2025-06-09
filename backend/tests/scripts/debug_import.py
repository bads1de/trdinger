#!/usr/bin/env python3
"""
インポートエラーのデバッグスクリプト
"""

import sys
import traceback

def test_imports():
    """段階的にインポートをテストして問題を特定"""
    
    print("=== インポートデバッグ開始 ===")
    
    # 基本的なインポート
    try:
        import deap
        print("✅ DEAP インポート成功")
    except Exception as e:
        print(f"❌ DEAP インポートエラー: {e}")
        traceback.print_exc()
        return False
    
    # 戦略遺伝子モデル
    try:
        from app.core.services.auto_strategy.models.strategy_gene import StrategyGene
        print("✅ StrategyGene インポート成功")
    except Exception as e:
        print(f"❌ StrategyGene インポートエラー: {e}")
        traceback.print_exc()
        return False
    
    # GA設定モデル
    try:
        from app.core.services.auto_strategy.models.ga_config import GAConfig
        print("✅ GAConfig インポート成功")
    except Exception as e:
        print(f"❌ GAConfig インポートエラー: {e}")
        traceback.print_exc()
        return False
    
    # 戦略ファクトリー
    try:
        from app.core.services.auto_strategy.factories.strategy_factory import StrategyFactory
        print("✅ StrategyFactory インポート成功")
    except Exception as e:
        print(f"❌ StrategyFactory インポートエラー: {e}")
        traceback.print_exc()
        return False
    
    # GAエンジン
    try:
        from app.core.services.auto_strategy.engines.ga_engine import GeneticAlgorithmEngine
        print("✅ GeneticAlgorithmEngine インポート成功")
    except Exception as e:
        print(f"❌ GeneticAlgorithmEngine インポートエラー: {e}")
        traceback.print_exc()
        return False
    
    # 自動戦略サービス
    try:
        from app.core.services.auto_strategy.services.auto_strategy_service import AutoStrategyService
        print("✅ AutoStrategyService インポート成功")
    except Exception as e:
        print(f"❌ AutoStrategyService インポートエラー: {e}")
        traceback.print_exc()
        return False
    
    # パッケージ全体
    try:
        from app.core.services.auto_strategy import AutoStrategyService as PackageService
        print("✅ パッケージ全体インポート成功")
    except Exception as e:
        print(f"❌ パッケージ全体インポートエラー: {e}")
        traceback.print_exc()
        return False
    
    # API モジュール
    try:
        from app.api.auto_strategy import router
        print("✅ API router インポート成功")
    except Exception as e:
        print(f"❌ API router インポートエラー: {e}")
        traceback.print_exc()
        return False
    
    # メインアプリ
    try:
        from app.main import app
        print("✅ メインアプリインポート成功")
    except Exception as e:
        print(f"❌ メインアプリインポートエラー: {e}")
        traceback.print_exc()
        return False
    
    print("=== 全てのインポートが成功しました ===")
    return True

if __name__ == "__main__":
    success = test_imports()
    if not success:
        sys.exit(1)
    print("✅ インポートテスト完了")
