#!/usr/bin/env python3
"""
新ライブラリテスト実行スクリプト

オートストラテジー強化で追加したライブラリの動作確認を行います。
"""

import sys
import os
import subprocess
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def check_requirements():
    """requirements.txtの内容確認"""
    print("📋 requirements.txtの確認...")
    
    requirements_path = project_root / "requirements.txt"
    if not requirements_path.exists():
        print("❌ requirements.txtが見つかりません")
        return False
    
    with open(requirements_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    required_libs = ['scikit-learn', 'lightgbm', 'joblib']
    missing_libs = []
    
    for lib in required_libs:
        if lib not in content:
            missing_libs.append(lib)
    
    if missing_libs:
        print(f"❌ 以下のライブラリがrequirements.txtに見つかりません: {missing_libs}")
        return False
    
    print("✅ requirements.txtに必要なライブラリが含まれています")
    return True


def install_requirements():
    """requirements.txtからライブラリをインストール"""
    print("📦 ライブラリのインストール...")
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], capture_output=True, text=True, cwd=project_root)
        
        if result.returncode == 0:
            print("✅ ライブラリのインストールが完了しました")
            return True
        else:
            print(f"❌ インストールエラー: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ インストール中にエラーが発生: {e}")
        return False


def run_tests():
    """テストの実行"""
    print("🧪 新ライブラリの動作テストを実行...")
    
    try:
        # テストファイルを直接実行
        test_file = project_root / "tests" / "test_new_libraries.py"
        
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
            print("✅ すべてのテストが正常に完了しました")
            return True
        else:
            print("❌ テストが失敗しました")
            return False
            
    except Exception as e:
        print(f"❌ テスト実行中にエラーが発生: {e}")
        return False


def check_python_version():
    """Pythonバージョンの確認"""
    print(f"🐍 Python version: {sys.version}")
    
    version_info = sys.version_info
    if version_info.major < 3 or (version_info.major == 3 and version_info.minor < 8):
        print("⚠️ Python 3.8以上を推奨します")
        return False
    
    print("✅ Pythonバージョンは適切です")
    return True


def main():
    """メイン実行関数"""
    print("🚀 オートストラテジー強化ライブラリテストを開始...")
    print("=" * 60)
    
    # Pythonバージョン確認
    if not check_python_version():
        print("❌ Pythonバージョンが不適切です")
        return False
    
    print()
    
    # requirements.txt確認
    if not check_requirements():
        print("❌ requirements.txtの確認に失敗しました")
        return False
    
    print()
    
    # ライブラリインストール
    print("ライブラリをインストールしますか？ (y/n): ", end="")
    response = input().lower().strip()
    
    if response in ['y', 'yes', 'はい']:
        if not install_requirements():
            print("❌ ライブラリのインストールに失敗しました")
            return False
    else:
        print("ℹ️ ライブラリのインストールをスキップしました")
    
    print()
    
    # テスト実行
    if not run_tests():
        print("❌ テストに失敗しました")
        return False
    
    print()
    print("🎉 すべての確認が完了しました！")
    print("オートストラテジー強化システムの新ライブラリは正常に動作しています。")
    
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