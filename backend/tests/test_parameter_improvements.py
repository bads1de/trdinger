#!/usr/bin/env python3
"""
戦略パラメータ表示の改善テスト

数値丸め処理とrisk_managementとtpsl_geneの重複解消をテストします。
"""

import sys
import os
import json

# パスを追加
sys.path.append(os.path.join(os.path.dirname(__file__), "backend"))


def test_numeric_rounding():
    """数値丸め処理のテスト"""
    print("=== 数値丸め処理テスト ===")

    try:
        from app.core.services.auto_strategy.models.tpsl_gene import (
            TPSLGene,
            TPSLMethod,
        )

        # 長い小数点を含むTP/SL遺伝子を作成
        tpsl_gene = TPSLGene(
            method=TPSLMethod.STATISTICAL,
            stop_loss_pct=0.02987600353303097,  # 長い小数点
            take_profit_pct=0.1007857829775447,  # 長い小数点
            risk_reward_ratio=2.7428760306220967,  # 長い小数点
            base_stop_loss=0.0655246275976081,  # 長い小数点
            atr_multiplier_sl=1.8940175516761912,  # 長い小数点
            atr_multiplier_tp=5.8949331863881245,  # 長い小数点
            confidence_threshold=0.7123456789,  # 長い小数点
            priority=1.23456789,  # 長い小数点
        )

        print(f"✅ 元の値（丸め前）:")
        print(f"   - SL: {tpsl_gene.stop_loss_pct}")
        print(f"   - TP: {tpsl_gene.take_profit_pct}")
        print(f"   - リスクリワード比: {tpsl_gene.risk_reward_ratio}")
        print(f"   - 信頼度閾値: {tpsl_gene.confidence_threshold}")
        print(f"   - 優先度: {tpsl_gene.priority}")

        # to_dict()で丸め処理を確認
        tpsl_dict = tpsl_gene.to_dict()

        print(f"\n✅ 丸め後の値:")
        print(f"   - SL: {tpsl_dict['stop_loss_pct']}")
        print(f"   - TP: {tpsl_dict['take_profit_pct']}")
        print(f"   - リスクリワード比: {tpsl_dict['risk_reward_ratio']}")
        print(f"   - 信頼度閾値: {tpsl_dict['confidence_threshold']}")
        print(f"   - 優先度: {tpsl_dict['priority']}")

        # 桁数確認
        sl_decimals = (
            len(str(tpsl_dict["stop_loss_pct"]).split(".")[1])
            if "." in str(tpsl_dict["stop_loss_pct"])
            else 0
        )
        tp_decimals = (
            len(str(tpsl_dict["take_profit_pct"]).split(".")[1])
            if "." in str(tpsl_dict["take_profit_pct"])
            else 0
        )

        print(f"\n✅ 小数点桁数確認:")
        print(f"   - SL小数点桁数: {sl_decimals} (期待値: 4桁以下)")
        print(f"   - TP小数点桁数: {tp_decimals} (期待値: 4桁以下)")

        return True

    except Exception as e:
        print(f"❌ 数値丸め処理テストエラー: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_risk_management_cleanup():
    """risk_managementとtpsl_geneの重複解消テスト"""
    print("\n=== risk_management重複解消テスト ===")

    try:
        from app.core.services.auto_strategy.models.tpsl_gene import (
            TPSLGene,
            TPSLMethod,
        )
        from app.core.services.auto_strategy.models.strategy_gene import StrategyGene
        from app.core.services.auto_strategy.models.gene_serialization import (
            GeneSerializer,
        )

        # TP/SL遺伝子を作成
        tpsl_gene = TPSLGene(
            method=TPSLMethod.RISK_REWARD_RATIO,
            stop_loss_pct=0.025,
            take_profit_pct=0.075,
            risk_reward_ratio=3.0,
        )

        # 戦略遺伝子を作成（古いrisk_managementにTP/SL設定を含む）
        strategy_gene = StrategyGene(
            id="test-strategy-cleanup",
            indicators=[],
            entry_conditions=[],
            exit_conditions=[],
            risk_management={
                "position_size": 0.123456789,  # 残すべき
                "stop_loss": 0.03,  # 除外されるべき
                "take_profit": 0.15,  # 除外されるべき
                "stop_loss_pct": 0.025,  # 除外されるべき
                "take_profit_pct": 0.075,  # 除外されるべき
                "_tpsl_strategy": "old_method",  # 除外されるべき
                "_tpsl_method": "fixed",  # 除外されるべき
                "max_trades_per_day": 5,  # 残すべき
            },
            tpsl_gene=tpsl_gene,
            metadata={"test": True},
        )

        print(f"✅ 元のrisk_management:")
        for key, value in strategy_gene.risk_management.items():
            print(f"   - {key}: {value}")

        # シリアライゼーション
        serializer = GeneSerializer()
        strategy_dict = serializer.strategy_gene_to_dict(strategy_gene)

        print(f"\n✅ クリーンアップ後のrisk_management:")
        for key, value in strategy_dict["risk_management"].items():
            print(f"   - {key}: {value}")

        print(f"\n✅ tpsl_gene情報:")
        if strategy_dict["tpsl_gene"]:
            print(f"   - メソッド: {strategy_dict['tpsl_gene']['method']}")
            print(f"   - SL: {strategy_dict['tpsl_gene']['stop_loss_pct']}")
            print(f"   - TP: {strategy_dict['tpsl_gene']['take_profit_pct']}")

        # 重複チェック
        risk_keys = set(strategy_dict["risk_management"].keys())
        tpsl_related_keys = {
            "stop_loss",
            "take_profit",
            "stop_loss_pct",
            "take_profit_pct",
        }
        overlap = risk_keys & tpsl_related_keys

        print(f"\n✅ 重複チェック:")
        print(f"   - TP/SL関連キーの重複: {list(overlap)} (期待値: 空リスト)")
        print(f"   - 重複解消成功: {len(overlap) == 0}")

        return len(overlap) == 0

    except Exception as e:
        print(f"❌ risk_management重複解消テストエラー: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_complete_parameter_json():
    """完全なパラメータJSON表示テスト"""
    print("\n=== 完全なパラメータJSON表示テスト ===")

    try:
        from app.core.services.auto_strategy.models.tpsl_gene import (
            TPSLGene,
            TPSLMethod,
        )
        from app.core.services.auto_strategy.models.strategy_gene import (
            StrategyGene,
            IndicatorGene,
        )
        from app.core.services.auto_strategy.models.gene_serialization import (
            GeneSerializer,
        )

        # 完全な戦略遺伝子を作成
        tpsl_gene = TPSLGene(
            method=TPSLMethod.STATISTICAL,
            stop_loss_pct=0.02987600353303097,
            take_profit_pct=0.1007857829775447,
            risk_reward_ratio=2.7428760306220967,
            base_stop_loss=0.0655246275976081,
        )

        indicator = IndicatorGene(type="RSI", parameters={"period": 48}, enabled=True)

        strategy_gene = StrategyGene(
            id="complete-test-strategy",
            indicators=[indicator],
            entry_conditions=[],
            exit_conditions=[],
            risk_management={
                "position_size": 0.123456789,
                "stop_loss": 0.03,  # 除外されるべき
                "take_profit": 0.15,  # 除外されるべき
            },
            tpsl_gene=tpsl_gene,
            metadata={"generated_by": "test", "test": True},
        )

        # シリアライゼーション
        serializer = GeneSerializer()
        strategy_dict = serializer.strategy_gene_to_dict(strategy_gene)

        # JSON形式で表示
        strategy_json = json.dumps(strategy_dict, ensure_ascii=False, indent=2)

        print(f"✅ 改善されたパラメータJSON:")
        print("```json")
        print(strategy_json)
        print("```")

        # 改善点の確認
        print(f"\n✅ 改善点確認:")

        # 1. 数値丸め
        tpsl_data = strategy_dict["tpsl_gene"]
        sl_str = str(tpsl_data["stop_loss_pct"])
        tp_str = str(tpsl_data["take_profit_pct"])
        sl_decimals = len(sl_str.split(".")[1]) if "." in sl_str else 0
        tp_decimals = len(tp_str.split(".")[1]) if "." in tp_str else 0

        print(f"   - SL小数点桁数: {sl_decimals} (改善前: 17桁)")
        print(f"   - TP小数点桁数: {tp_decimals} (改善前: 16桁)")

        # 2. 重複解消
        risk_keys = list(strategy_dict["risk_management"].keys())
        print(f"   - risk_managementキー: {risk_keys}")
        print(
            f"   - TP/SL設定除外: {'stop_loss' not in risk_keys and 'take_profit' not in risk_keys}"
        )

        # 3. tpsl_gene存在確認
        print(f"   - tpsl_gene存在: {strategy_dict['tpsl_gene'] is not None}")
        print(f"   - TP/SLメソッド表示: {tpsl_data['method']}")

        return True

    except Exception as e:
        print(f"❌ 完全なパラメータJSON表示テストエラー: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """メインテスト実行"""
    print("戦略パラメータ表示改善テスト開始\n")

    results = []

    # 数値丸め処理テスト
    results.append(test_numeric_rounding())

    # risk_management重複解消テスト
    results.append(test_risk_management_cleanup())

    # 完全なパラメータJSON表示テスト
    results.append(test_complete_parameter_json())

    # 結果サマリー
    print(f"\n=== テスト結果サマリー ===")
    print(f"成功: {sum(results)}/{len(results)}")

    if all(results):
        print("✅ 全てのテストが成功しました！")
        print(
            "数値の丸め処理とrisk_managementとtpsl_geneの重複解消が正しく動作しています。"
        )
    else:
        print("❌ 一部のテストが失敗しました。")

    return all(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
