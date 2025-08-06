"""
機械学習包括的検証レポートの品質検証テスト

このテストは改善されたMLレポートが業界標準に準拠し、
必要な要素を含んでいることを検証します。
"""

import re
import pytest
from pathlib import Path


class TestMLValidationReport:
    """ML検証レポートの品質テスト"""
    
    @pytest.fixture
    def report_path(self):
        """レポートファイルのパス"""
        return Path("backend/tests/reports/ml_comprehensive_validation_report.md")
    
    @pytest.fixture
    def report_content(self, report_path):
        """レポートの内容を読み込み"""
        with open(report_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def test_report_file_exists(self, report_path):
        """レポートファイルが存在することを確認"""
        assert report_path.exists(), "ML検証レポートファイルが存在しません"
    
    def test_executive_summary_exists(self, report_content):
        """エグゼクティブサマリーが存在することを確認"""
        assert "Executive Summary" in report_content, "エグゼクティブサマリーが不足しています"
        assert "主要成果" in report_content, "主要成果セクションが不足しています"
        assert "推奨事項" in report_content, "推奨事項が不足しています"
    
    def test_industry_standard_metrics(self, report_content):
        """業界標準メトリクスが含まれていることを確認"""
        required_metrics = [
            "Precision", "Recall", "F1スコア", "ROC-AUC", "PR-AUC",
            "Sharpe Ratio", "Sortino Ratio", "Calmar Ratio",
            "最大ドローダウン", "勝率", "利益因子"
        ]
        
        for metric in required_metrics:
            assert metric in report_content, f"必須メトリクス '{metric}' が不足しています"
    
    def test_statistical_validation(self, report_content):
        """統計的検証セクションが存在することを確認"""
        statistical_tests = [
            "統計的検証", "統計的有意性テスト", "信頼区間",
            "クロスバリデーション", "ロバストネス検証"
        ]
        
        for test in statistical_tests:
            assert test in report_content, f"統計的検証項目 '{test}' が不足しています"
    
    def test_risk_assessment(self, report_content):
        """リスク評価セクションが存在することを確認"""
        risk_elements = [
            "リスク評価", "モデルリスク", "運用リスク", "セキュリティリスク",
            "過学習リスク", "データドリフト", "概念ドリフト"
        ]
        
        for element in risk_elements:
            assert element in report_content, f"リスク評価項目 '{element}' が不足しています"
    
    def test_bias_fairness_analysis(self, report_content):
        """バイアス・公平性分析が存在することを確認"""
        bias_elements = [
            "バイアス・公平性分析", "予測バイアス", "特徴量重要度",
            "公平性メトリクス", "時間バイアス"
        ]
        
        for element in bias_elements:
            assert element in report_content, f"バイアス分析項目 '{element}' が不足しています"
    
    def test_business_impact_analysis(self, report_content):
        """ビジネス影響分析が存在することを確認"""
        business_elements = [
            "ビジネス影響分析", "収益性評価", "コスト効果分析",
            "ROI", "年間収益率", "回収期間"
        ]
        
        for element in business_elements:
            assert element in report_content, f"ビジネス分析項目 '{element}' が不足しています"
    
    def test_compliance_governance(self, report_content):
        """コンプライアンス・ガバナンスセクションが存在することを確認"""
        compliance_elements = [
            "コンプライアンス・ガバナンス", "規制準拠", "モデルガバナンス",
            "データガバナンス", "IEEE", "NIST"
        ]
        
        for element in compliance_elements:
            assert element in report_content, f"コンプライアンス項目 '{element}' が不足しています"
    
    def test_reproducibility_documentation(self, report_content):
        """再現性・文書化セクションが存在することを確認"""
        repro_elements = [
            "再現性・文書化", "技術文書", "環境再現性", "実験管理",
            "Docker", "バージョン管理"
        ]
        
        for element in repro_elements:
            assert element in report_content, f"再現性項目 '{element}' が不足しています"
    
    def test_financial_specific_metrics(self, report_content):
        """金融特化メトリクスが含まれていることを確認"""
        financial_metrics = [
            "Sharpe Ratio", "Sortino Ratio", "Calmar Ratio",
            "Information Ratio", "VaR", "CVaR", "Kelly Criterion"
        ]
        
        for metric in financial_metrics:
            assert metric in report_content, f"金融メトリクス '{metric}' が不足しています"
    
    def test_structured_recommendations(self, report_content):
        """構造化された推奨事項が存在することを確認"""
        recommendation_levels = [
            "即座に対応すべき項目", "中期的な改善計画", "長期的な発展方向"
        ]
        
        for level in recommendation_levels:
            assert level in report_content, f"推奨事項レベル '{level}' が不足しています"
    
    def test_quality_score_table(self, report_content):
        """品質スコア評価テーブルが存在することを確認"""
        quality_aspects = ["信頼性", "性能", "機能性", "保守性", "総合"]
        
        for aspect in quality_aspects:
            assert aspect in report_content, f"品質評価項目 '{aspect}' が不足しています"
    
    def test_conclusion_completeness(self, report_content):
        """結論セクションが完全であることを確認"""
        conclusion_elements = [
            "結論", "主要成果", "戦略的価値", "最終評価",
            "本番環境投入準備完了"
        ]
        
        for element in conclusion_elements:
            assert element in report_content, f"結論要素 '{element}' が不足しています"
    
    def test_numerical_data_presence(self, report_content):
        """数値データが適切に含まれていることを確認"""
        # パーセンテージの存在確認
        percentage_pattern = r'\d+\.\d+%'
        percentages = re.findall(percentage_pattern, report_content)
        assert len(percentages) >= 20, "十分な数値データ（パーセンテージ）が不足しています"
        
        # 比率・スコアの存在確認
        ratio_pattern = r'\d+\.\d+'
        ratios = re.findall(ratio_pattern, report_content)
        assert len(ratios) >= 30, "十分な数値データ（比率・スコア）が不足しています"
    
    def test_table_formatting(self, report_content):
        """テーブル形式が適切であることを確認"""
        # Markdownテーブルの存在確認
        table_pattern = r'\|.*\|.*\|'
        tables = re.findall(table_pattern, report_content)
        assert len(tables) >= 10, "十分なテーブル形式データが不足しています"
    
    def test_section_hierarchy(self, report_content):
        """セクション階層が適切であることを確認"""
        # H2ヘッダーの存在確認
        h2_pattern = r'^## .+'
        h2_headers = re.findall(h2_pattern, report_content, re.MULTILINE)
        assert len(h2_headers) >= 10, "十分なセクション構造が不足しています"
        
        # H3ヘッダーの存在確認
        h3_pattern = r'^### .+'
        h3_headers = re.findall(h3_pattern, report_content, re.MULTILINE)
        assert len(h3_headers) >= 20, "十分なサブセクション構造が不足しています"
    
    def test_emoji_visual_elements(self, report_content):
        """視覚的要素（絵文字）が適切に使用されていることを確認"""
        emoji_pattern = r'[📊📈📉🎯🔧🚨⚠️✅🟡🔴🎉🚀💼💰🔍🏆📝🔄📋🛡️]'
        emojis = re.findall(emoji_pattern, report_content)
        assert len(emojis) >= 50, "視覚的要素（絵文字）が不足しています"


if __name__ == "__main__":
    # テスト実行
    pytest.main([__file__, "-v"])
