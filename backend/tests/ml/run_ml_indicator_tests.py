#!/usr/bin/env python3
"""
ML指標テスト実行スクリプト

ML_UP_PROB、ML_DOWN_PROB、ML_RANGE_PROB指標の動作確認を行います。
"""

import sys
import os
import subprocess
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def run_ml_indicator_tests():
    """ML指標テストの実行"""
    print("🧪 ML指標の動作テストを実行...")

    try:
        # テストファイルを直接実行
        test_file = Path(__file__).parent / "test_ml_indicators.py"
        
        if not test_file.exists():
            print(f"❌ テストファイルが見つかりません: {test_file}")
            return False
        
        # Pythonでテストファイルを実行
        result = subprocess.run([
            sys.executable, str(test_file)
        ], capture_output=True, text=True, cwd=project_root)
        
        print("📊 テスト結果:")
        print(result.stdout)
        
        if result.stderr:
            print("⚠️ エラー出力:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("✅ ML指標テストが正常に完了しました")
            return True
        else:
            print("❌ ML指標テストが失敗しました")
            return False
            
    except Exception as e:
        print(f"❌ テスト実行中にエラーが発生: {e}")
        return False


def check_ml_components():
    """ML関連コンポーネントの存在確認"""
    print("📋 ML関連コンポーネントの確認...")
    
    components = [
        "app/core/services/feature_engineering/feature_engineering_service.py",
        "app/core/services/ml/signal_generator.py",
        "app/core/services/auto_strategy/services/ml_indicator_service.py",
        "app/core/services/auto_strategy/engines/fitness_sharing.py"
    ]
    
    missing_components = []
    
    for component in components:
        component_path = project_root / component
        if component_path.exists():
            print(f"✅ {component}")
        else:
            print(f"❌ {component}")
            missing_components.append(component)
    
    if missing_components:
        print(f"\n⚠️ 以下のコンポーネントが見つかりません:")
        for component in missing_components:
            print(f"   - {component}")
        return False
    
    print("✅ すべてのML関連コンポーネントが存在します")
    return True


def test_imports():
    """重要なインポートのテスト"""
    print("📦 重要なインポートのテスト...")
    
    import_tests = [
        ("FeatureEngineeringService", "from app.core.services.feature_engineering import FeatureEngineeringService"),
        ("MLSignalGenerator", "from app.core.services.ml import MLSignalGenerator"),
        ("MLIndicatorService", "from app.core.services.auto_strategy.services.ml_indicator_service import MLIndicatorService"),
        ("FitnessSharing", "from app.core.services.auto_strategy.engines.fitness_sharing import FitnessSharing")
    ]
    
    failed_imports = []
    
    for name, import_statement in import_tests:
        try:
            exec(import_statement)
            print(f"✅ {name}")
        except ImportError as e:
            print(f"❌ {name}: {e}")
            failed_imports.append(name)
        except Exception as e:
            print(f"⚠️ {name}: {e}")
            failed_imports.append(name)
    
    if failed_imports:
        print(f"\n⚠️ 以下のインポートが失敗しました:")
        for name in failed_imports:
            print(f"   - {name}")
        return False
    
    print("✅ すべての重要なインポートが成功しました")
    return True


def main():
    """メイン実行関数"""
    print("🚀 ML指標テストを開始...")
    print("=" * 60)
    
    # ML関連コンポーネントの確認
    if not check_ml_components():
        print("❌ ML関連コンポーネントの確認に失敗しました")
        return False
    
    print()
    
    # インポートテスト
    if not test_imports():
        print("❌ インポートテストに失敗しました")
        return False
    
    print()
    
    # ML指標テスト実行
    if not run_ml_indicator_tests():
        print("❌ ML指標テストに失敗しました")
        return False
    
    print()
    print("🎉 すべてのML指標テストが完了しました！")
    print("ML_UP_PROB、ML_DOWN_PROB、ML_RANGE_PROB指標は正常に動作しています。")
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️ ユーザーによって中断されました")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 予期しないエラーが発生しました: {e}")
        sys.exit(1)