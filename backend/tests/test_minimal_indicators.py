"""
最小限の指標設定テスト
"""

import sys
import os

def test_basic_imports():
    """基本的なインポートテスト"""
    try:
        # Pythonパスにバックエンドを追加
        current_dir = os.getcwd()
        print(f"現在のディレクトリ: {current_dir}")

        # 既にbackendにいる場合
        if 'backend' in current_dir.split(os.sep):
            backend_path = current_dir
        else:
            backend_path = os.path.join(current_dir, 'backend')
        print(f"バックエンドパス: {backend_path}")

        sys.path.insert(0, backend_path)

        print("=== 基本インポートテスト ===")

        # 最小限のインポートテスト
        print("Thresholdsクラステスト...")

        # 直接ファイルレベルでimport
        import importlib.util

        # indicator_policies.pyを直接インポート
        indicator_file = os.path.join(backend_path, 'app', 'services', 'auto_strategy', 'core', 'indicator_policies.py')
        print(f"indicator_policies.py ファイルパス: {indicator_file}")
        print(f"ファイルが存在する: {os.path.exists(indicator_file)}")

        if not os.path.exists(indicator_file):
            print("ファイルが見つかりません")
            return False

        spec = importlib.util.spec_from_file_location(
            "indicator_policies",
            indicator_file
        )
        indicator_policies = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(indicator_policies)

        # Thresholdsクラスを取得
        Thresholds = indicator_policies.Thresholds
        print(f"Thresholdsクラス取得成功")

        # フィールドの確認
        import inspect
        fields = [field.name for field in inspect.signature(Thresholds.__init__).parameters.values()]
        print(f"Thresholdsフィールド: {fields}")

        # 新しいフィールドが含まれているか確認
        new_fields = ['roc_long_lt', 'roc_short_gt', 'mom_long_lt', 'mom_short_gt']
        for field in new_fields:
            if field in fields:
                print(f"✓ {field} フィールドが存在します")
            else:
                print(f"✗ {field} フィールドがありません")

        # オペランドグループのテスト
        print("\nオペランドグループテスト...")
        spec = importlib.util.spec_from_file_location(
            "operand_grouping",
            os.path.join(backend_path, 'app', 'services', 'auto_strategy', 'utils', 'operand_grouping.py')
        )
        operand_grouping = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(operand_grouping)

        operand_grouping_system = operand_grouping.operand_grouping_system
        test_indicators = ["CMO", "TRIX", "APO", "ROCP"]
        for indicator in test_indicators:
            group = operand_grouping_system.get_operand_group(indicator)
            print(f"  {indicator}: {group.value}")

        print("\n=== テスト完了 ===")
        return True

    except Exception as e:
        print(f"テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_imports()
    print(f"\nテスト結果: {'成功' if success else '失敗'}")