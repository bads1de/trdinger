"""
新しい指標の統合テスト
"""

import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'backend'))

def test_new_indicators():
    """新しい指標のテスト"""
    try:
        # オペランドグループングシステムのテスト
        from app.services.auto_strategy.core.operand_grouping import operand_grouping_system
        print("=== オペランドグループングシステムテスト ===")

        # 新しい指標のグループマッピングテスト
        new_indicators = [
            "CMO", "TRIX", "ULTOSC", "BOP", "APO", "PPO", "ROCP", "ROCR", "ROCR100",
            "STOCHRSI", "SMI", "PVO", "QQE", "TSI", "KST", "STC", "COPPOCK", "ER", "ERI",
            "INERTIA", "PGO", "PSL", "RSX", "SQUEEZE", "SQUEEZE_PRO", "BIAS", "BRAR",
            "CG", "FISHER", "PVOL", "PVR", "EOM", "KVO", "PVT", "CMF", "NVI", "PVI",
            "AOBV", "EFI", "RVI"
        ]

        print("新しい指標のグループ分類:")
        for indicator in new_indicators:
            group = operand_grouping_system.get_operand_group(indicator)
            print(f"  {indicator}: {group.value}")

        # 閾値ポリシーのテスト
        print("\n=== 閾値ポリシーテスト ===")
        from app.services.auto_strategy.core.indicator_policies import ThresholdPolicy

        normal_thresholds = ThresholdPolicy.get(profile="normal")
        print("正常プロファイルの新しい閾値:")
        print(f"  ROC: {getattr(normal_thresholds, 'roc_long_lt', 'N/A')}/ {getattr(normal_thresholds, 'roc_short_gt', 'N/A')}")
        print(f"  MOM: {getattr(normal_thresholds, 'mom_long_lt', 'N/A')}/ {getattr(normal_thresholds, 'mom_short_gt', 'N/A')}")
        print(f"  STOCH: {getattr(normal_thresholds, 'stoch_long_lt', 'N/A')}/ {getattr(normal_thresholds, 'stoch_short_gt', 'N/A')}")
        print(f"  CMO: {getattr(normal_thresholds, 'cmo_long_lt', 'N/A')}/ {getattr(normal_thresholds, 'cmo_short_gt', 'N/A')}")
        print(f"  TRIX: {getattr(normal_thresholds, 'trix_long_lt', 'N/A')}/ {getattr(normal_thresholds, 'trix_short_gt', 'N/A')}")
        print(f"  BOP: {getattr(normal_thresholds, 'bop_long_gt', 'N/A')}/ {getattr(normal_thresholds, 'bop_short_lt', 'N/A')}")
        print(f"  APO: {getattr(normal_thresholds, 'apo_long_gt', 'N/A')}/ {getattr(normal_thresholds, 'apo_short_lt', 'N/A')}")

        # レジストリのテスト
        print("\n=== 指標レジストリーテスト ===")
        from app.services.indicators.config import indicator_registry

        # いくつ目の指標を取得
        momentum_indicators = [cfg for cfg in indicator_registry._configs.values() if cfg.category == "momentum"]
        print(f"登録されたモメンタム系指標数: {len(momentum_indicators)}")

        cmopear最新指標を確認
        cmo_config = indicator_registry.get_indicator_config("CMO")
        if cmo_config:
            print(f"CMO設定取得成功: {cmo_config.indicator_name}")
        else:
            print("CMO設定が見つかりません")

        print("\n=== テスト完了 ===")
        return True

    except Exception as e:
        print(f"テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_new_indicators()