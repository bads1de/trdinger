#!/usr/bin/env python3
"""
RSI順張り戦略とoperand_grouping完全統合テスト

以下のテストを統合:
- RSI順張りとoperand_groupingの完全統合テスト
- エラー処理・条件生成・互換性の包括的検証
- 複数タイムフレームでの条件生成テスト
- operand_groupingとの互換性スコア確認
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from app.services.auto_strategy.generators.condition_generator import ConditionGenerator
    from app.services.auto_strategy.core.operand_grouping import operand_grouping_system, OperandGroup
    from app.services.auto_strategy.models.strategy_models import Condition, IndicatorGene
    from app.services.auto_strategy.config.constants import IndicatorType

    print("=== Import successful ===")

except Exception as e:
    print(f"Import error: {e}")
    sys.exit(1)

def test_rsi_operand_grouping_complete():
    """RSIとoperand_groupingの完全統合テスト"""
    print("=== RSIとoperand_grouping完全統合テスト ===")

    # 1. RSIがPERCENTAGE_0_100に正しく分類されているか
    rsi_group = operand_grouping_system.get_operand_group("RSI")
    print(f"RSIグループ: {rsi_group.value}")
    assert rsi_group == OperandGroup.PERCENTAGE_0_100, "RSIがPERCENTAGE_0_100グループに分類されているべき"

    # 2. 互換性スコアを確認（同じグループ内）
    compatibility_score = operand_grouping_system.get_compatibility_score("RSI", "RSI")
    print(f"RSI vs RSI互換性スコア: {compatibility_score}")
    assert compatibility_score == 1.0, "同じグループ内は完全に互換であるべき"

    # 3. 他のオシレーター指標との互換性
    other_oscillators = ["STOCH", "ADX", "MFI"]
    for indicator in other_oscillators:
        score = operand_grouping_system.get_compatibility_score("RSI", indicator)
        print(f"RSI vs {indicator}互換性スコア: {score}")
        assert score >= 0.8, f"RSIと{indicator}の互換性スコアが0.8以上であるべき"

    # 4. 異なるスケールの指標との低い互換性
    low_compatibility_indicators = ["SMA", "close", "ATR"]
    for indicator in low_compatibility_indicators:
        score = operand_grouping_system.get_compatibility_score("RSI", indicator)
        print(f"RSI vs {indicator}互換性スコア: {score}")
        assert score <= 0.3, f"RSIと{indicator}の互換性スコアが0.3以下であるべき"

    # 5. 境界値とエッジケース
    empty_score = operand_grouping_system.get_compatibility_score("", "")
    invalid_score = operand_grouping_system.get_compatibility_score("NONEXISTENT", "RSI")

    print(f"空オペランドスコア: {empty_score}")
    print(f"無効オペランドスコア: {invalid_score}")

    # None値の処理テスト
    try:
        none_score = operand_grouping_system.get_compatibility_score(None, None)
        print(f"Noneスコア: {none_score}")
    except Exception as e:
        print(f"None値処理: 例外が発生 - {e}")

    print("* operand_grouping完全統合テスト全通過")

    return True

def test_rsi_condition_generation_comprehensive():
    """RSI条件生成とoperand_grouping統合の包括的テスト"""
    print("\n=== RSI条件生成包括的テスト ===")

    generator = ConditionGenerator()
    generator.set_context(threshold_profile="normal", timeframe="1h")

    # RSIインジケータの作成
    rsi_indicator = IndicatorGene(
        enabled=True,
        type="RSI",
        parameters={"period": 14}
    )

    # ロング条件生成
    long_conditions = generator._create_momentum_long_conditions(rsi_indicator)
    print(f"ロング条件数: {len(long_conditions)}")

    if long_conditions:
        condition = long_conditions[0]
        print(f"ロング条件: {condition.left_operand} {condition.operator} {condition.right_operand}")

        # 順張りチェック: RSI > 75
        assert condition.left_operand == "RSI", "左オペランドがRSIであるべき"
        assert condition.operator == ">", "オペレーターが>であるべき"
        assert condition.right_operand == 75, "右オペランドが75であるべき"

        # operand_groupingとの互換性検証
        left_group = operand_grouping_system.get_operand_group(condition.left_operand)
        assert left_group == OperandGroup.PERCENTAGE_0_100

    # ショート条件生成
    short_conditions = generator._create_momentum_short_conditions(rsi_indicator)
    print(f"ショート条件数: {len(short_conditions)}")

    if short_conditions:
        condition = short_conditions[0]
        print(f"ショート条件: {condition.left_operand} {condition.operator} {condition.right_operand}")

        # 順張りチェック: RSI < 25
        assert condition.left_operand == "RSI", "左オペランドがRSIであるべき"
        assert condition.operator == "<", "オペレーターが<であるべき"
        assert condition.right_operand == 25, "右オペランドが25であるべき"

    # 条件妥当性検証
    long_condition = Condition(left_operand="RSI", operator=">", right_operand=75)
    short_condition = Condition(left_operand="RSI", operator="<", right_operand=25)

    long_valid, long_reason = operand_grouping_system.validate_condition("RSI", 75)
    print(f"ロング条件妥当性: {long_valid} - {long_reason}")

    short_valid, short_reason = operand_grouping_system.validate_condition("RSI", 25)
    print(f"ショート条件妥当性: {short_valid} - {short_reason}")

    assert long_valid and short_valid, "RSI条件が妥当であるべき"

    print("* RSI条件生成包括的テスト全通過")

    return True

def test_multi_timeframe_condition_generation():
    """複数タイムフレームでの条件生成テスト"""
    print("\n=== 複数タイムフレーム条件生成テスト ===")

    generator = ConditionGenerator()
    rsi_indicator = IndicatorGene(
        enabled=True,
        type="RSI",
        parameters={"period": 14}
    )

    # テストするタイムフレームとプロファイル
    timeframes = ["15m", "1h", "4h", "1d"]
    profiles = ["aggressive", "normal", "conservative"]

    total_tests = 0
    passed_tests = 0

    for tf in timeframes:
        for profile in profiles:
            generator.set_context(timeframe=tf, threshold_profile=profile)

            # ロング条件生成
            long_conditions = generator._create_momentum_long_conditions(rsi_indicator)
            short_conditions = generator._create_momentum_short_conditions(rsi_indicator)

            # 条件検証
            long_valid = (len(long_conditions) > 0 and
                         hasattr(long_conditions[0], 'right_operand') and
                         long_conditions[0].right_operand == 75 and
                         hasattr(long_conditions[0], 'operator') and
                         long_conditions[0].operator == '>')
            short_valid = (len(short_conditions) > 0 and
                          hasattr(short_conditions[0], 'right_operand') and
                          short_conditions[0].right_operand == 25 and
                          hasattr(short_conditions[0], 'operator') and
                          short_conditions[0].operator == '<')

            if long_valid and short_valid:
                print(f"* {tf}/{profile}: 正常 - Long: {len(long_conditions)}, Short: {len(short_conditions)}")
                passed_tests += 1
            else:
                print(f"* {tf}/{profile}: 異常 - Long: {'OK' if long_valid else 'NG'}, Short: {'OK' if short_valid else 'NG'}")

            total_tests += 1

    print(f"タイムフレームテスト結果: {passed_tests}/{total_tests} 通過")

    # operand_groupingタイムフレーム境界テスト
    timeframes_extended = ["1m", "5m", "15m", "1h", "4h", "1d", "1w"]
    for tf in timeframes_extended:
        generator.set_context(timeframe=tf, threshold_profile="normal")
        long_conditions = generator._create_momentum_long_conditions(rsi_indicator)
        assert len(long_conditions) > 0, f"{tf}で条件生成に失敗"

    assert passed_tests == total_tests, "全てのタイムフレーム/プロファイルでテストが通過すべき"
    print("* 複数タイムフレーム条件生成テスト全通過")

    return True

def test_error_handling_and_validation():
    """エラー処理と堅牢性テスト"""
    print("\n=== エラー処理と堅牢性テスト ===")

    try:
        generator = ConditionGenerator()

        # 正常な初期化
        generator.set_context(timeframe="15m", threshold_profile="normal")
        print("* ConditionGenerator初期化成功")

        # 境界値テスト
        timeframes_test = ["1m", "5m", "15m", "1h", "4h", "1d", "1w"]
        for tf in timeframes_test:
            generator.set_context(timeframe=tf, threshold_profile="normal")
            rsi_ind = IndicatorGene(enabled=True, type="RSI", parameters={"period": 14})
            long_conds = generator._create_momentum_long_conditions(rsi_ind)
            short_conds = generator._create_momentum_short_conditions(rsi_ind)
            assert len(long_conds) > 0 and len(short_conds) > 0, f"{tf}で条件生成失敗"

        print("* 境界値テスト通過")

        # 無効な入力処理
        try:
            generator.set_context(timeframe="invalid_tf", threshold_profile="normal")
            print("* 不正タイムフレーム対応")
        except Exception:
            print("* 不正タイムフレームエラーハンドリング")

        # operand_grouping境界値テスト
        score_all = operand_grouping_system.get_compatibility_score("RSI", "RSI")
        score_none = operand_grouping_system.get_compatibility_score("INVALID", "INVALID")

        print(f"* 有効スコア: {score_all}")
        print(f"* 無効スコア: {score_none}")

        print("* エラー処理と堅牢性テスト全通過")

        return True

    except Exception as e:
        print(f"* エラー処理テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_compatibility_score_verification():
    """operand_groupingとの互換性スコア検証テスト"""
    print("\n=== operand_grouping互換性スコア検証テスト ===")

    # RSIと類似オシレーターの高い互換性確認
    high_compat_indicators = ["RSI", "STOCH", "MFI", "ADX"]
    for ind1 in high_compat_indicators:
        for ind2 in high_compat_indicators:
            score = operand_grouping_system.get_compatibility_score(ind1, ind2)
            print(f"{ind1} vs {ind2}: {score}")
            if ind1 == ind2:
                assert score == 1.0, f"同じインジケータ{ind1}のスコアが1.0であるべき"
            else:
                assert score >= 0.7, f"{ind1}と{ind2}のスコアが0.7以上であるべき"

    # RSIとトレンドフォロー指標の低い互換性確認
    low_compat_indicators = ["SMA", "EMA", "close", "ATR"]
    for ind in low_compat_indicators:
        score = operand_grouping_system.get_compatibility_score("RSI", ind)
        print(f"RSI vs {ind}: {score}")
        assert score <= 0.4, f"RSIと{ind}のスコアが0.4以下であるべき"

    print("* operand_grouping互換性スコア検証テスト全通過")

    return True

def main():
    """メイン実行"""
    try:
        print("=== RSI順張り戦略とoperand_grouping完全統合テスト開始 ===\n")

        results = []

        # テスト1: RSIとoperand_grouping完全統合
        results.append(("RSIおよびoperand_grouping完全統合", test_rsi_operand_grouping_complete()))

        # テスト2: RSI条件生成包括的テスト
        results.append(("RSI条件生成包括的", test_rsi_condition_generation_comprehensive()))

        # テスト3: 複数タイムフレーム条件生成
        results.append(("複数タイムフレーム条件生成", test_multi_timeframe_condition_generation()))

        # テスト4: エラー処理と堅牢性
        results.append(("エラー処理と堅牢性", test_error_handling_and_validation()))

        # テスト5: 互換性スコア検証
        results.append(("operand_grouping互換性スコア", test_compatibility_score_verification()))

        # 結果集計
        passed = sum(1 for _, result in results if result)
        total = len(results)

        print("\n=== 最終結果集計 ===")
        for name, result in results:
            status = "PASS" if result else "FAIL"
            print(f"* {name}: {status}")

        print(f"\n* 全体結果: {passed}/{total} 通過")

        if passed == total:
            print("* [SUCCESS] 完全統合テスト完了 - RSI順張りとoperand_groupingの統合が正常に動作しています")
            print("* [SUCCESS] エラー処理・条件生成・互換性の包括的検証完了")
            print("* [SUCCESS] 複数タイムフレームでの条件生成テスト完了")
            print("* [SUCCESS] operand_groupingとの互換性スコア確認完了")
            return True
        else:
            print("* [FAILED] 統合テスト失敗 - 修正が必要")
            return False

    except Exception as e:
        print(f"\n=== 完全統合テスト失敗: {e} ===")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)