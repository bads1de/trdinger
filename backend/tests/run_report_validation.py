"""
機械学習包括的検証レポートの品質検証スクリプト

pytestの代わりに直接実行してレポートの品質を検証します。
"""

import os
import re
import sys
from pathlib import Path


def test_report_quality():
    """レポートの品質を検証"""
    
    # レポートファイルのパス
    report_path = Path("tests/reports/ml_comprehensive_validation_report.md")
    
    if not report_path.exists():
        print("❌ レポートファイルが存在しません")
        return False
    
    # レポートの内容を読み込み
    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("🔍 機械学習包括的検証レポートの品質検証を開始...")
    print("=" * 60)
    
    # テスト結果を格納
    test_results = []
    
    # 1. エグゼクティブサマリーの確認
    print("📋 1. エグゼクティブサマリーの確認...")
    executive_elements = ["Executive Summary", "主要成果", "推奨事項"]
    for element in executive_elements:
        if element in content:
            print(f"   ✅ {element} - 存在")
            test_results.append(True)
        else:
            print(f"   ❌ {element} - 不足")
            test_results.append(False)
    
    # 2. 業界標準メトリクスの確認
    print("\n📊 2. 業界標準メトリクスの確認...")
    required_metrics = [
        "Precision", "Recall", "F1スコア", "ROC-AUC", "PR-AUC",
        "Sharpe Ratio", "Sortino Ratio", "Calmar Ratio",
        "最大ドローダウン", "勝率", "利益因子"
    ]
    
    for metric in required_metrics:
        if metric in content:
            print(f"   ✅ {metric} - 存在")
            test_results.append(True)
        else:
            print(f"   ❌ {metric} - 不足")
            test_results.append(False)
    
    # 3. 統計的検証の確認
    print("\n📈 3. 統計的検証の確認...")
    statistical_tests = [
        "統計的検証", "統計的有意性テスト", "信頼区間",
        "クロスバリデーション", "ロバストネス検証"
    ]
    
    for test in statistical_tests:
        if test in content:
            print(f"   ✅ {test} - 存在")
            test_results.append(True)
        else:
            print(f"   ❌ {test} - 不足")
            test_results.append(False)
    
    # 4. リスク評価の確認
    print("\n⚠️ 4. リスク評価の確認...")
    risk_elements = [
        "リスク評価", "モデルリスク", "運用リスク", "セキュリティリスク",
        "過学習リスク", "データドリフト", "概念ドリフト"
    ]
    
    for element in risk_elements:
        if element in content:
            print(f"   ✅ {element} - 存在")
            test_results.append(True)
        else:
            print(f"   ❌ {element} - 不足")
            test_results.append(False)
    
    # 5. バイアス・公平性分析の確認
    print("\n🔍 5. バイアス・公平性分析の確認...")
    bias_elements = [
        "バイアス・公平性分析", "予測バイアス", "特徴量重要度",
        "公平性メトリクス", "時間バイアス"
    ]
    
    for element in bias_elements:
        if element in content:
            print(f"   ✅ {element} - 存在")
            test_results.append(True)
        else:
            print(f"   ❌ {element} - 不足")
            test_results.append(False)
    
    # 6. ビジネス影響分析の確認
    print("\n💼 6. ビジネス影響分析の確認...")
    business_elements = [
        "ビジネス影響分析", "収益性評価", "コスト効果分析",
        "ROI", "年間収益率", "回収期間"
    ]
    
    for element in business_elements:
        if element in content:
            print(f"   ✅ {element} - 存在")
            test_results.append(True)
        else:
            print(f"   ❌ {element} - 不足")
            test_results.append(False)
    
    # 7. コンプライアンス・ガバナンスの確認
    print("\n🏛️ 7. コンプライアンス・ガバナンスの確認...")
    compliance_elements = [
        "コンプライアンス・ガバナンス", "規制準拠", "モデルガバナンス",
        "データガバナンス", "IEEE", "NIST"
    ]
    
    for element in compliance_elements:
        if element in content:
            print(f"   ✅ {element} - 存在")
            test_results.append(True)
        else:
            print(f"   ❌ {element} - 不足")
            test_results.append(False)
    
    # 8. 再現性・文書化の確認
    print("\n🔄 8. 再現性・文書化の確認...")
    repro_elements = [
        "再現性・文書化", "技術文書", "環境再現性", "実験管理",
        "Docker", "バージョン管理"
    ]
    
    for element in repro_elements:
        if element in content:
            print(f"   ✅ {element} - 存在")
            test_results.append(True)
        else:
            print(f"   ❌ {element} - 不足")
            test_results.append(False)
    
    # 9. 数値データの確認
    print("\n📊 9. 数値データの確認...")
    
    # パーセンテージの確認
    percentage_pattern = r'\d+\.\d+%'
    percentages = re.findall(percentage_pattern, content)
    if len(percentages) >= 20:
        print(f"   ✅ パーセンテージデータ - {len(percentages)}個存在")
        test_results.append(True)
    else:
        print(f"   ❌ パーセンテージデータ - {len(percentages)}個（20個以上必要）")
        test_results.append(False)
    
    # 比率・スコアの確認
    ratio_pattern = r'\d+\.\d+'
    ratios = re.findall(ratio_pattern, content)
    if len(ratios) >= 30:
        print(f"   ✅ 比率・スコアデータ - {len(ratios)}個存在")
        test_results.append(True)
    else:
        print(f"   ❌ 比率・スコアデータ - {len(ratios)}個（30個以上必要）")
        test_results.append(False)
    
    # 10. テーブル形式の確認
    print("\n📋 10. テーブル形式の確認...")
    table_pattern = r'\|.*\|.*\|'
    tables = re.findall(table_pattern, content)
    if len(tables) >= 10:
        print(f"   ✅ テーブル形式データ - {len(tables)}行存在")
        test_results.append(True)
    else:
        print(f"   ❌ テーブル形式データ - {len(tables)}行（10行以上必要）")
        test_results.append(False)
    
    # 11. セクション階層の確認
    print("\n📑 11. セクション階層の確認...")
    
    # H2ヘッダーの確認
    h2_pattern = r'^## .+'
    h2_headers = re.findall(h2_pattern, content, re.MULTILINE)
    if len(h2_headers) >= 10:
        print(f"   ✅ H2ヘッダー - {len(h2_headers)}個存在")
        test_results.append(True)
    else:
        print(f"   ❌ H2ヘッダー - {len(h2_headers)}個（10個以上必要）")
        test_results.append(False)
    
    # H3ヘッダーの確認
    h3_pattern = r'^### .+'
    h3_headers = re.findall(h3_pattern, content, re.MULTILINE)
    if len(h3_headers) >= 20:
        print(f"   ✅ H3ヘッダー - {len(h3_headers)}個存在")
        test_results.append(True)
    else:
        print(f"   ❌ H3ヘッダー - {len(h3_headers)}個（20個以上必要）")
        test_results.append(False)
    
    # 12. 視覚的要素の確認
    print("\n🎨 12. 視覚的要素の確認...")
    emoji_pattern = r'[📊📈📉🎯🔧🚨⚠️✅🟡🔴🎉🚀💼💰🔍🏆📝🔄📋🛡️]'
    emojis = re.findall(emoji_pattern, content)
    if len(emojis) >= 50:
        print(f"   ✅ 絵文字 - {len(emojis)}個存在")
        test_results.append(True)
    else:
        print(f"   ❌ 絵文字 - {len(emojis)}個（50個以上必要）")
        test_results.append(False)
    
    # 結果の集計
    print("\n" + "=" * 60)
    print("📊 検証結果サマリー")
    print("=" * 60)
    
    total_tests = len(test_results)
    passed_tests = sum(test_results)
    success_rate = (passed_tests / total_tests) * 100
    
    print(f"総テスト数: {total_tests}")
    print(f"成功: {passed_tests}")
    print(f"失敗: {total_tests - passed_tests}")
    print(f"成功率: {success_rate:.1f}%")
    
    if success_rate >= 90:
        print("🎉 優秀 - レポートは業界標準に準拠しています")
        return True
    elif success_rate >= 80:
        print("✅ 良好 - レポートは概ね適切ですが、改善の余地があります")
        return True
    elif success_rate >= 70:
        print("🟡 要改善 - レポートには重要な要素が不足しています")
        return False
    else:
        print("❌ 不適切 - レポートは大幅な改善が必要です")
        return False


if __name__ == "__main__":
    success = test_report_quality()
    sys.exit(0 if success else 1)
