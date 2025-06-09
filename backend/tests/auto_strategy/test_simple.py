"""
自動戦略生成機能の簡単なテスト

基本的なインポートと動作確認を行います。
"""

def test_imports():
    """基本的なインポートテスト"""
    try:
        # 戦略遺伝子のインポート
        from app.core.services.auto_strategy.models.strategy_gene import StrategyGene, IndicatorGene, Condition
        print("✅ 戦略遺伝子モデルのインポート成功")
        
        # GA設定のインポート
        from app.core.services.auto_strategy.models.ga_config import GAConfig
        print("✅ GA設定モデルのインポート成功")
        
        # 戦略ファクトリーのインポート
        from app.core.services.auto_strategy.factories.strategy_factory import StrategyFactory
        print("✅ 戦略ファクトリーのインポート成功")
        
        return True
        
    except ImportError as e:
        print(f"❌ インポートエラー: {e}")
        return False


def test_basic_creation():
    """基本的なオブジェクト作成テスト"""
    try:
        from app.core.services.auto_strategy.models.strategy_gene import StrategyGene, IndicatorGene, Condition
        from app.core.services.auto_strategy.models.ga_config import GAConfig
        from app.core.services.auto_strategy.factories.strategy_factory import StrategyFactory
        
        # 指標遺伝子の作成
        indicator = IndicatorGene(
            type="SMA",
            parameters={"period": 20},
            enabled=True
        )
        print("✅ 指標遺伝子の作成成功")
        
        # 条件の作成
        condition = Condition(
            left_operand="price",
            operator=">",
            right_operand=100
        )
        print("✅ 条件の作成成功")
        
        # 戦略遺伝子の作成
        gene = StrategyGene(
            indicators=[indicator],
            entry_conditions=[condition],
            exit_conditions=[condition]
        )
        print("✅ 戦略遺伝子の作成成功")
        
        # GA設定の作成
        config = GAConfig(population_size=10, generations=5)
        print("✅ GA設定の作成成功")
        
        # 戦略ファクトリーの作成
        factory = StrategyFactory()
        print("✅ 戦略ファクトリーの作成成功")
        
        return True
        
    except Exception as e:
        print(f"❌ オブジェクト作成エラー: {e}")
        return False


def test_deap_import():
    """DEAPライブラリのインポートテスト"""
    try:
        import deap
        from deap import base, creator, tools, algorithms
        print("✅ DEAPライブラリのインポート成功")
        print(f"   DEAP バージョン: {deap.__version__ if hasattr(deap, '__version__') else 'Unknown'}")
        return True
        
    except ImportError as e:
        print(f"❌ DEAPインポートエラー: {e}")
        return False


if __name__ == "__main__":
    print("=== 自動戦略生成機能 簡単テスト ===")
    
    # インポートテスト
    print("\n1. インポートテスト")
    import_success = test_imports()
    
    if import_success:
        # オブジェクト作成テスト
        print("\n2. オブジェクト作成テスト")
        creation_success = test_basic_creation()
        
        # DEAPテスト
        print("\n3. DEAPライブラリテスト")
        deap_success = test_deap_import()
        
        if creation_success and deap_success:
            print("\n🎉 全ての簡単テストが成功しました！")
        else:
            print("\n⚠️ 一部のテストが失敗しました")
    else:
        print("\n❌ インポートに失敗したため、他のテストをスキップします")
