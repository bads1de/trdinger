import sys
import os
import yaml
from pathlib import Path

# バックエンドパスを追加
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_yaml_syntax_validation():
    """YAML構文検証テスト"""
    print("=== YAML構文検証テスト ===")

    yaml_path = Path(__file__).parent.parent / "app/services/auto_strategy/config/technical_indicators_config.yaml"

    if not yaml_path.exists():
        print(f"❌ ERROR: YAMLファイルが見つかりません: {yaml_path}")
        return False

    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        print("✅ YAML構文: OK")

        # 基本構造検証
        if "indicators" not in data:
            print("❌ ERROR: 'indicators'セクションが見つかりません")
            return False

        indicators = data.get("indicators", {})
        print(f"📊 指標数: {len(indicators)}")

        # required fieldsチェック
        required_fields = ["type", "scale_type", "thresholds", "conditions"]
        missing_any_fields = False

        for indicator_name, indicator_config in indicators.items():
            missing_fields = []
            for field in required_fields:
                if field not in indicator_config:
                    missing_fields.append(field)

            if missing_fields:
                print(f"⚠️  WARNING: {indicator_name}: 必須フィールド不足 {missing_fields}")
                missing_any_fields = True

        if not missing_any_fields:
            print("✅ 基本構造検証: PASSED")

        return True

    except yaml.YAMLError as e:
        print(f"❌ ERROR: YAML解析エラー: {e}")
        return False
    except Exception as e:
        print(f"❌ ERROR: 予期しないエラー: {e}")
        return False

def test_new_indicators_verification():
    """新規指標確認テスト"""
    print("\n=== 新規指標確認テスト ===")

    yaml_path = Path(__file__).parent.parent / "app/services/auto_strategy/config/technical_indicators_config.yaml"

    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        indicators = data.get("indicators", {})

        # 新規指標リスト（AO, ATR, ICHIMOKU, SUPERTRENDなどの主要新規指標）
        new_indicators = [
            "AO", "ATR", "ICHIMOKU", "SUPERTREND",
            "TSI", "RMI", "KELTNER", "DONCHIAN",
            "MASSI", "VIC", "CHANE", "RSX",
            "REX", "RVGI"
        ]

        all_present = True
        verified_new_indicators = []

        for indicator in new_indicators:
            if indicator in indicators:
                config = indicators[indicator]
                print(f"✅ {indicator}: 存在します")

                # thresholdsの構造確認
                thresholds = config.get("thresholds", {})
                if thresholds:
                    if isinstance(thresholds, dict):
                        risk_levels = ['aggressive', 'normal', 'conservative']
                        risk_level_ok = True

                        for risk_level in risk_levels:
                            if risk_level not in thresholds:
                                risk_level_ok = False
                                print(f"⚠️  WARNING: {indicator}: {risk_level} risk level 不足")

                        if risk_level_ok:
                            print(f"   ✅ Risk levels: OK")
                    else:
                        print(f"   ✅ Thresholds structure: OK")
                else:
                    print(f"   ⚠️  WARNING: No thresholds configured")

                verified_new_indicators.append(indicator)
            else:
                print(f"❌ {indicator}: 見つかりません")
                all_present = False

        print(f"📈 検証済み新規指標: {len(verified_new_indicators)}/{len(new_indicators)}")

        return all_present

    except Exception as e:
        print(f"❌ ERROR: 検証エラー: {e}")
        return False

def test_generate_characteristics_function():
    """generate_characteristics_from_yaml関数テスト"""
    print("\n=== generate_characteristics_from_yaml関数テスト ===")

    yaml_path = Path(__file__).parent.parent / "app/services/auto_strategy/config/technical_indicators_config.yaml"

    try:
        # 関数のダイナミックインポート
        from app.services.auto_strategy.utils.common_utils import YamlIndicatorUtils

        # 関数存在確認
        if not hasattr(YamlIndicatorUtils, 'generate_characteristics_from_yaml'):
            print("❌ ERROR: generate_characteristics_from_yaml関数が見つかりません")
            return False

        print("✅ 関数存在確認: OK")

        # 関数実行
        result = YamlIndicatorUtils.generate_characteristics_from_yaml(str(yaml_path))

        if not result:
            print("❌ ERROR: 空の結果が返されました")
            return False

        print(f"✅ 生成された特性数: {len(result)}")

        # 主要指標の確認
        key_indicators = ["AO", "ATR", "ICHIMOKU", "SUPERTREND", "RSI", "MACD", "BBANDS"]

        indicators_found = 0
        for indicator in key_indicators:
            if indicator in result:
                config = result[indicator]
                print(f"✅ {indicator}: {config.get('type', 'unknown')} タイプ")
                indicators_found += 1
            else:
                print(f"⚠️  WARNING: {indicator} が見つかりません")

        print(f"📊 主要指標検索結果: {indicators_found}/{len(key_indicators)}")

        # 構造サンプル表示
        if result:
            sample_key = next(iter(result.keys()))
            sample_config = result[sample_key]
            print("\n📋 サンプル特性構造:")
            print(f"  キー: {sample_key}")
            print(f"  タイプ: {sample_config.get('type', '不明')}")
            print(f"  スケールタイプ: {sample_config.get('scale_type', '不明')}")
            print(f"  フィールド数: {len(sample_config)}")

        return len(result) > 0

    except ImportError as e:
        print(f"❌ ERROR: Importエラー: {e}")
        return False
    except Exception as e:
        print(f"❌ ERROR: 実行エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_structure_validation():
    """構造解析・検証テスト"""
    print("\n=== 構造解析・検証テスト ===")

    yaml_path = Path(__file__).parent.parent / "app/services/auto_strategy/config/technical_indicators_config.yaml"

    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        indicators = data.get("indicators", {})

        # 指標タイプ別統計
        indicator_types = {}
        scale_types = {}

        validation_issues = []

        for name, config in indicators.items():
            # タイプ統計
            ind_type = config.get("type", "unknown")
            indicator_types[ind_type] = indicator_types.get(ind_type, 0) + 1

            # スケールタイプ統計
            scale_type = config.get("scale_type", "unknown")
            scale_types[scale_type] = scale_types.get(scale_type, 0) + 1

            # 構造検証
            if not isinstance(config, dict):
                validation_issues.append(f"{name}: 辞書形式である必要があります")
                continue

            # 必須フィールド存在チェック
            required_fields = ["type", "scale_type"]
            for field in required_fields:
                if field not in config:
                    validation_issues.append(f"{name}: {field}フィールドが必須です")

        print(f"📊 指標タイプ分布:")
        for typ, count in indicator_types.items():
            print(f"   {typ}: {count}")

        print(f"\n📊 スケールタイプ分布:")
        for scale, count in scale_types.items():
            print(f"   {scale}: {count}")

        if validation_issues:
            print("\n⚠️  構造検証警告:")
            for issue in validation_issues[:10]:  # 最初の10件のみ表示
                print(f"  - {issue}")
            if len(validation_issues) > 10:
                print(f"  - 他 {len(validation_issues) - 10} 件...")
        else:
            print("\n✅ 構造検証: PASSED")

        return len(validation_issues) == 0

    except Exception as e:
        print(f"❌ ERROR: 構造検証エラー: {e}")
        return False

def test_error_handling():
    """エラーハンドリングテスト"""
    print("\n=== エラーハンドリングテスト ===")

    from app.services.auto_strategy.utils.common_utils import YamlIndicatorUtils

    error_tests_passed = 0
    total_tests = 4

    # 1. 存在しないファイルパス
    try:
        result = YamlIndicatorUtils.generate_characteristics_from_yaml("/nonexistent/path/config.yaml")
        if result == {}:
            print("✅ 非存在ファイルテスト: PASSED")
            error_tests_passed += 1
        else:
            print("❌ 非存在ファイルテスト: FAILED - 空の辞書が返されるべき")
    except Exception as e:
        print(f"⚠️  非存在ファイルテスト: 例外発生 {e}")

    # 2. 無効なYAMLファイル
    try:
        invalid_yaml_path = Path(__file__).parent / "invalid_config.yaml"

        # 一時的な無効YAMLファイル作成
        with open(invalid_yaml_path, 'w', encoding='utf-8') as f:
            f.write("invalid: yaml: content: [\n")

        result = YamlIndicatorUtils.generate_characteristics_from_yaml(str(invalid_yaml_path))
        if result == {}:
            print("✅ 無効YAMLテスト: PASSED")
            error_tests_passed += 1
        else:
            print("❌ 無効YAMLテスト: FAILED")

        # クリーンアップ
        invalid_yaml_path.unlink(missing_ok=True)

    except Exception as e:
        print(f"⚠️  無効YAMLテスト: 例外発生 {e}")

    # 3. 空ファイルテスト
    try:
        empty_yaml_path = Path(__file__).parent / "empty_config.yaml"

        with open(empty_yaml_path, 'w', encoding='utf-8') as f:
            f.write("")

        result = YamlIndicatorUtils.generate_characteristics_from_yaml(str(empty_yaml_path))
        if result == {}:
            print("✅ 空ファイルテスト: PASSED")
            error_tests_passed += 1
        else:
            print("❌ 空ファイルテスト: FAILED")

        # クリーンアップ
        empty_yaml_path.unlink(missing_ok=True)

    except Exception as e:
        print(f"⚠️  空ファイルテスト: 例外発生 {e}")

    # 4. 無効なindicatorsセクション
    try:
        invalid_section_path = Path(__file__).parent / "invalid_section_config.yaml"

        with open(invalid_section_path, 'w', encoding='utf-8') as f:
            f.write("indicators: not_a_dict\n")

        result = YamlIndicatorUtils.generate_characteristics_from_yaml(str(invalid_section_path))
        if result == {}:
            print("✅ 無効セクションテスト: PASSED")
            error_tests_passed += 1
        else:
            print("❌ 無効セクションテスト: FAILED")

        # クリーンアップ
        invalid_section_path.unlink(missing_ok=True)

    except Exception as e:
        print(f"⚠️  無効セクションテスト: 例外発生 {e}")

    print(f"📊 エラーハンドリングテスト結果: {error_tests_passed}/{total_tests} PASSED")

    return error_tests_passed == total_tests

def test_compatibility_verification():
    """既存指標との互換性確認テスト"""
    print("\n=== 既存指標との互換性確認テスト ===")

    yaml_path = Path(__file__).parent.parent / "app/services/auto_strategy/config/technical_indicators_config.yaml"

    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        indicators = data.get("indicators", {})

        # 既存の主要指標リスト
        existing_indicators = [
            "RSI", "MACD", "STOCH", "STOCHRSI", "TSF",
            "BBANDS", "EMA", "SMA", "WMA", "LINEARREG",
            "ADX", "CCI", "ULTOSC", "TRIX", "ROC",
            "MFI", "OBV", "AD", "SAR", "CMF"
        ]

        compatibility_issues = []
        compatible_indicators = 0

        for indicator in existing_indicators:
            if indicator in indicators:
                config = indicators[indicator]

                # 基本互換性チェック
                has_basic_structure = (
                    isinstance(config, dict) and
                    "type" in config and
                    "scale_type" in config and
                    "thresholds" in config and
                    "conditions" in config
                )

                if has_basic_structure:
                    print(f"✅ {indicator}: 互換性 OK")
                    compatible_indicators += 1
                else:
                    compatibility_issues.append(f"{indicator}: 構造不備")
                    print(f"❌ {indicator}: 互換性 NG - 構造不備")

            else:
                compatibility_issues.append(f"{indicator}: 指標が見つからない")
                print(f"⚠️  WARNING: {indicator}: 見つかりません")

        print(f"\n📊 互換性結果: {compatible_indicators}/{len(existing_indicators)} 互換")

        if compatibility_issues:
            print(f"\n⚠️  互換性警告 ({len(compatibility_issues)} 件):")
            for issue in compatibility_issues[:10]:  # 最初の10件
                print(f"  - {issue}")
            if len(compatibility_issues) > 10:
                print(f"  - 他 {len(compatibility_issues) - 10} 件...")

        # リスクレベル互換性チェック
        print(f"\n🔍 リスクレベル互換性詳細チェック:")

        well_configured_risks = 0
        for name, config in indicators.items():
            if name in existing_indicators:
                thresholds = config.get("thresholds", {})
                if isinstance(thresholds, dict):
                    risk_levels = ['aggressive', 'normal', 'conservative']
                    has_all_risks = all(level in thresholds for level in risk_levels)
                    if has_all_risks:
                        well_configured_risks += 1

        print(f"   適切設定指標: {well_configured_risks}/{len([i for i in existing_indicators if i in indicators])}")

        return len(compatibility_issues) == 0

    except Exception as e:
        print(f"❌ ERROR: 互換性テストエラー: {e}")
        return False

def main():
    """メイン実行関数"""
    print("YAML設定包括的テストスイート開始")
    print("=" * 60)

    # テスト実行情報表示
    print(f"作業ディレクトリ: {Path(__file__).parent}")
    print(f"テストファイル: {Path(__file__).name}")
    print(f"実行日時: {Path(__file__).parent.stat().st_mtime if Path(__file__).exists() else 'N/A'}")

    # テストスイート実行
    test_results = []

    print("\n" + "=" * 60)

    # 各テスト実行
    test_results.append(("YAML構文検証", test_yaml_syntax_validation()))
    test_results.append(("新規指標確認", test_new_indicators_verification()))
    test_results.append(("generate_characteristics_from_yaml関数", test_generate_characteristics_function()))
    test_results.append(("構造解析・検証", test_structure_validation()))
    test_results.append(("エラーハンドリング", test_error_handling()))
    test_results.append(("既存指標との互換性確認", test_compatibility_verification()))

    print("\n" + "=" * 60)
    print("テスト実行結果サマリー")
    print("=" * 60)

    passed_tests = 0
    for test_name, result in test_results:
        status = "PASSED" if result else "FAILED"
        print(f"{status}: {test_name}")
        if result:
            passed_tests += 1

    print(f"\n総テスト数: {len(test_results)}")
    print(f"成功数: {passed_tests}")
    print(f"失敗数: {len(test_results) - passed_tests}")
    print(f"成功率: {passed_tests / len(test_results) * 100:.1f}%")

    # 結果判定
    if passed_tests == len(test_results):
        print("\n全テストが成功しました！YAML設定は正常に動作しています。")
        return 0
    elif passed_tests >= len(test_results) * 0.7:  # 70%以上成功
        print(f"\nほとんどのテストが成功しました（{passed_tests}/{len(test_results)}）")
        print("   一部のマイナーな問題がありますが、YAML設定はおおむね正常です。")
        return 1
    else:
        print(f"\n重大な問題があります（{passed_tests}/{len(test_results)}）")
        print("   YAML設定の修正を強く推奨します。")
        return 2

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)