"""
シンプルな新しい指標テスト
"""

import sys
import os

def test_indicator_imports():
    """指標系のシンプルテスト"""
    try:
        # Pythonパスにバックエンドを追加
        backend_path = os.path.join(os.getcwd(), 'backend')
        sys.path.insert(0, backend_path)

        print("=== 指標設定テスト ===")

        # unified_configを直接テスト
        print("unified_configテスト...")
        from app.config.unified_config import unified_config
        print(f"Comprehensibility unified_config設定取得成功")

        # ThresholdPolicyテスト
        print("ThresholdPolicyテスト...")
        from app.services.auto_strategy.core.indicator_policies import ThresholdPolicy
        normal_thresholds = ThresholdPolicy.get(profile="normal")
        print(f"正常プロファイル閾値取得成功")

        # 新しい指標のチェック
        new_fields = ['roc_long_lt', 'roc_short_gt', 'mom_long_lt', 'mom_short_gt',
                     'stoch_long_lt', 'stoch_short_gt', 'cmo_long_lt', 'cmo_short_gt',
                     'trix_long_lt', 'trix_short_gt', 'bop_long_gt', 'bop_short_lt',
                     'apo_long_gt', 'apo_short_lt']

        for field in new_fields:
            value = getattr(normal_thresholds, field, None)
            if value is not None:
                print(f"  ✓ {field}: {value}")
            else:
                print(f"  ✗ {field}: なし")

        # オペランドグループテスト
        print("\nオペランドグループテスト...")
        from app.services.auto_strategy.utils.operand_grouping import operand_grouping_system

        test_indicators = ["CMO", "TRIX", "ULTOSC", "BOP", "APO", "PPO", "ROCP", "ROCR", "ROCR100", "STOCHRSI", "SMI", "PVO"]
        for indicator in test_indicators:
            group = operand_grouping_system.get_operand_group(indicator)
            print(f"  {indicator}: {group.value}")

        print("\n=== テスト完了 ===")
        return True

    except ImportError as e:
        print(f"インポートエラー: {e}")
        return False
    except Exception as e:
        print(f"その他のエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_indicator_imports()
    print(f"\nテスト結果: {'成功' if success else '失敗'}")