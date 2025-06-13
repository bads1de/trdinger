#!/usr/bin/env python3
"""
重複コード検出スクリプト
"""

import re
import os
from collections import defaultdict
from pathlib import Path


def extract_indicators_from_file(file_path):
    """ファイルから指標名を抽出"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 指標名を抽出（クォートで囲まれた文字列）
        patterns = [
            r'"([A-Z_]+)"',  # ダブルクォート
            r"'([A-Z_]+)'",  # シングルクォート
        ]

        indicators = set()
        for pattern in patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                # 指標名らしいもの（大文字のアルファベットとアンダースコア）
                if re.match(r"^[A-Z][A-Z_]*$", match) and len(match) >= 2:
                    indicators.add(match)

        return list(indicators)
    except Exception as e:
        print(f"❌ ファイル読み込みエラー {file_path}: {e}")
        return []


def detect_duplicate_indicators():
    """指標リストの重複を検出"""
    print("🔍 指標リスト重複検出開始")
    print("=" * 50)

    # 検索対象ファイル
    files_to_check = [
        "backend/app/core/services/auto_strategy/generators/random_gene_generator.py",
        "backend/app/core/services/auto_strategy/models/ga_config.py",
        "frontend/components/backtest/GAConfigForm.tsx",
    ]

    # 各ファイルから指標リストを抽出
    file_indicators = {}
    for file_path in files_to_check:
        if os.path.exists(file_path):
            indicators = extract_indicators_from_file(file_path)
            file_indicators[file_path] = indicators
            print(f"📁 {os.path.basename(file_path)}: {len(indicators)}個の指標")
        else:
            print(f"❌ ファイルが見つかりません: {file_path}")

    # 重複指標の検出
    all_indicators = []
    for indicators in file_indicators.values():
        all_indicators.extend(indicators)

    indicator_count = defaultdict(int)
    for indicator in all_indicators:
        indicator_count[indicator] += 1

    print(f"\n📊 重複指標分析:")
    duplicates = {k: v for k, v in indicator_count.items() if v > 1}
    if duplicates:
        for indicator, count in sorted(duplicates.items()):
            print(f"  🔄 {indicator}: {count}箇所で定義")
    else:
        print("  ✅ 重複指標なし")

    print(f"\n📈 統計:")
    print(f"  - 総指標数: {len(set(all_indicators))}")
    print(f"  - 重複指標数: {len(duplicates)}")
    if len(set(all_indicators)) > 0:
        print(f"  - 重複率: {len(duplicates)/len(set(all_indicators))*100:.1f}%")

    return file_indicators, duplicates


def detect_duplicate_imports():
    """重複インポートを検出"""
    print("\n🔍 重複インポート検出開始")
    print("=" * 50)

    # 検索対象ディレクトリ
    search_dirs = [
        "backend/app/core/services/indicators/",
        "backend/app/core/services/auto_strategy/",
    ]

    import_count = defaultdict(int)
    file_imports = {}

    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            for root, dirs, files in os.walk(search_dir):
                for file in files:
                    if file.endswith(".py"):
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, "r", encoding="utf-8") as f:
                                content = f.read()

                            # import文を抽出
                            import_lines = re.findall(
                                r"^(import .*|from .* import .*)$",
                                content,
                                re.MULTILINE,
                            )
                            file_imports[file_path] = import_lines

                            for imp in import_lines:
                                import_count[imp] += 1

                        except Exception as e:
                            print(f"❌ ファイル読み込みエラー {file_path}: {e}")

    # 重複インポートの検出
    duplicate_imports = {k: v for k, v in import_count.items() if v > 1}

    if duplicate_imports:
        print(f"🔄 重複インポート ({len(duplicate_imports)}個):")
        for imp, count in sorted(duplicate_imports.items()):
            print(f"  - {imp} ({count}箇所)")
    else:
        print("✅ 重複インポートなし")

    return duplicate_imports


def detect_parameter_logic_duplicates():
    """パラメータ生成ロジックの重複を検出"""
    print("\n🔍 パラメータ生成ロジック重複検出開始")
    print("=" * 50)

    file_path = (
        "backend/app/core/services/auto_strategy/generators/random_gene_generator.py"
    )

    if not os.path.exists(file_path):
        print(f"❌ ファイルが見つかりません: {file_path}")
        return {}

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # elif indicator_type == パターンを抽出
        elif_patterns = re.findall(
            r'elif indicator_type == ["\']([^"\']+)["\']:(.*?)(?=elif|else:|return)',
            content,
            re.DOTALL,
        )

        # パラメータ生成パターンを分析
        parameter_patterns = defaultdict(list)
        for indicator, logic in elif_patterns:
            # パラメータ生成の種類を分析
            if "random.randint" in logic:
                if "period" in logic:
                    parameter_patterns["period_randint"].append(indicator)
                if "fast_period" in logic and "slow_period" in logic:
                    parameter_patterns["fast_slow_periods"].append(indicator)
            if "random.uniform" in logic:
                parameter_patterns["uniform_params"].append(indicator)
            if "random.choice" in logic:
                parameter_patterns["choice_params"].append(indicator)

        print("📊 パラメータ生成パターン分析:")
        for pattern, indicators in parameter_patterns.items():
            if len(indicators) > 1:
                print(f"  🔄 {pattern}: {len(indicators)}個の指標")
                for indicator in indicators[:5]:  # 最初の5個を表示
                    print(f"    - {indicator}")
                if len(indicators) > 5:
                    print(f"    ... 他{len(indicators)-5}個")

        return parameter_patterns

    except Exception as e:
        print(f"❌ ファイル読み込みエラー: {e}")
        return {}


def detect_threshold_logic_duplicates():
    """閾値生成ロジックの重複を検出"""
    print("\n🔍 閾値生成ロジック重複検出開始")
    print("=" * 50)

    file_path = (
        "backend/app/core/services/auto_strategy/generators/random_gene_generator.py"
    )

    if not os.path.exists(file_path):
        print(f"❌ ファイルが見つかりません: {file_path}")
        return {}

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 閾値生成関数内のelif文を抽出
        threshold_function = re.search(
            r"def _generate_threshold_value.*?(?=def|\Z)", content, re.DOTALL
        )
        if not threshold_function:
            print("❌ 閾値生成関数が見つかりません")
            return {}

        threshold_content = threshold_function.group(0)
        elif_patterns = re.findall(
            r'elif ["\']([^"\']+)["\'] in operand:(.*?)(?=elif|else:|return)',
            threshold_content,
            re.DOTALL,
        )

        # 閾値生成パターンを分析
        threshold_patterns = defaultdict(list)
        for indicator, logic in elif_patterns:
            if "random.uniform(0, 100)" in logic or "random.uniform(20, 80)" in logic:
                threshold_patterns["percentage_0_100"].append(indicator)
            elif "random.uniform(-100, 100)" in logic:
                threshold_patterns["percentage_neg100_100"].append(indicator)
            elif (
                "random.uniform(0.9, 1.1)" in logic
                or "random.uniform(0.95, 1.05)" in logic
            ):
                threshold_patterns["price_ratio"].append(indicator)
            elif "random.uniform(-" in logic and ", 0)" in logic:
                threshold_patterns["negative_range"].append(indicator)

        print("📊 閾値生成パターン分析:")
        for pattern, indicators in threshold_patterns.items():
            if len(indicators) > 1:
                print(f"  🔄 {pattern}: {len(indicators)}個の指標")
                for indicator in indicators:
                    print(f"    - {indicator}")

        return threshold_patterns

    except Exception as e:
        print(f"❌ ファイル読み込みエラー: {e}")
        return {}


if __name__ == "__main__":
    # 指標リストの重複検出
    file_indicators, duplicate_indicators = detect_duplicate_indicators()

    # インポートの重複検出
    duplicate_imports = detect_duplicate_imports()

    # パラメータ生成ロジックの重複検出
    parameter_duplicates = detect_parameter_logic_duplicates()

    # 閾値生成ロジックの重複検出
    threshold_duplicates = detect_threshold_logic_duplicates()

    print(f"\n🎯 検出結果サマリー:")
    print(f"  - 重複指標: {len(duplicate_indicators)}個")
    print(f"  - 重複インポート: {len(duplicate_imports)}個")
    print(f"  - パラメータ生成パターン: {len(parameter_duplicates)}種類")
    print(f"  - 閾値生成パターン: {len(threshold_duplicates)}種類")
