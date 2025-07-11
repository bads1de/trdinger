#!/usr/bin/env python3
"""
新しいインジケータ統合の詳細分析レポート生成

実証検証結果を詳細に分析し、統合の成功度を評価します。
"""

import json
import sys
import os
from typing import Dict, List
from collections import Counter, defaultdict

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


class VerificationResultAnalyzer:
    """検証結果の詳細分析クラス"""
    
    def __init__(self, result_file: str = "new_indicator_verification_result.json"):
        """初期化"""
        self.result_file = result_file
        self.data = self._load_results()
        
    def _load_results(self) -> Dict:
        """結果ファイルを読み込み"""
        try:
            with open(self.result_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"結果ファイル {self.result_file} が見つかりません。")
            return {}
    
    def analyze_indicator_coverage(self) -> Dict:
        """インジケータカバレッジ分析"""
        print("=== インジケータカバレッジ分析 ===")
        
        service_data = self.data.get("service_verification", {})
        total_supported = service_data.get("total_supported", 0)
        new_indicators = service_data.get("new_indicators_found", {})
        
        coverage_analysis = {
            "total_indicators": total_supported,
            "new_indicators_by_category": {},
            "coverage_summary": {}
        }
        
        total_new_indicators = 0
        for category, indicators in new_indicators.items():
            count = len(indicators)
            total_new_indicators += count
            coverage_analysis["new_indicators_by_category"][category] = {
                "count": count,
                "indicators": indicators,
                "percentage": (count / total_supported * 100) if total_supported > 0 else 0
            }
            
            print(f"{category.upper()}:")
            print(f"  インジケータ数: {count}")
            print(f"  全体に占める割合: {count / total_supported * 100:.1f}%")
            print(f"  代表例: {', '.join(indicators[:5])}")
            print()
        
        coverage_analysis["coverage_summary"] = {
            "total_new_indicators": total_new_indicators,
            "new_indicator_percentage": (total_new_indicators / total_supported * 100) if total_supported > 0 else 0,
            "traditional_indicators": total_supported - total_new_indicators
        }
        
        print(f"新しいインジケータ総数: {total_new_indicators}")
        print(f"従来のインジケータ数: {total_supported - total_new_indicators}")
        print(f"新しいインジケータの割合: {total_new_indicators / total_supported * 100:.1f}%")
        
        return coverage_analysis
    
    def analyze_strategy_quality(self) -> Dict:
        """戦略品質分析"""
        print("\n=== 戦略品質分析 ===")
        
        strategies = self.data.get("strategies", [])
        detailed_report = self.data.get("detailed_report", {})
        
        quality_metrics = {
            "total_strategies": len(strategies),
            "valid_strategies": 0,
            "strategies_with_new_indicators": 0,
            "indicator_distribution": defaultdict(int),
            "condition_distribution": defaultdict(int),
            "category_effectiveness": defaultdict(list),
            "complexity_analysis": {}
        }
        
        indicator_counts = []
        condition_counts = []
        
        for strategy in strategies:
            # 基本品質チェック
            if (len(strategy.get("indicators", [])) > 0 and 
                len(strategy.get("long_conditions", [])) > 0):
                quality_metrics["valid_strategies"] += 1
            
            # 新しいインジケータ使用チェック
            if strategy.get("has_new_indicators", False):
                quality_metrics["strategies_with_new_indicators"] += 1
            
            # インジケータ分布
            indicator_count = len(strategy.get("indicators", []))
            indicator_counts.append(indicator_count)
            quality_metrics["indicator_distribution"][indicator_count] += 1
            
            # 条件分布
            total_conditions = (len(strategy.get("long_conditions", [])) + 
                              len(strategy.get("short_conditions", [])))
            condition_counts.append(total_conditions)
            quality_metrics["condition_distribution"][total_conditions] += 1
            
            # カテゴリ効果分析
            for category in strategy.get("categories_used", []):
                quality_metrics["category_effectiveness"][category].append({
                    "strategy_id": strategy.get("strategy_id"),
                    "indicator_count": indicator_count,
                    "condition_count": total_conditions
                })
        
        # 複雑度分析
        if indicator_counts and condition_counts:
            quality_metrics["complexity_analysis"] = {
                "avg_indicators": sum(indicator_counts) / len(indicator_counts),
                "avg_conditions": sum(condition_counts) / len(condition_counts),
                "max_indicators": max(indicator_counts),
                "min_indicators": min(indicator_counts),
                "max_conditions": max(condition_counts),
                "min_conditions": min(condition_counts)
            }
        
        # 結果出力
        total = quality_metrics["total_strategies"]
        valid = quality_metrics["valid_strategies"]
        with_new = quality_metrics["strategies_with_new_indicators"]
        
        print(f"生成戦略総数: {total}")
        print(f"有効な戦略: {valid} ({valid/total*100:.1f}%)")
        print(f"新しいインジケータを含む戦略: {with_new} ({with_new/total*100:.1f}%)")
        
        complexity = quality_metrics["complexity_analysis"]
        if complexity:
            print(f"\n戦略複雑度:")
            print(f"  平均インジケータ数: {complexity['avg_indicators']:.1f}")
            print(f"  平均条件数: {complexity['avg_conditions']:.1f}")
            print(f"  インジケータ数範囲: {complexity['min_indicators']}-{complexity['max_indicators']}")
            print(f"  条件数範囲: {complexity['min_conditions']}-{complexity['max_conditions']}")
        
        return quality_metrics
    
    def analyze_category_effectiveness(self) -> Dict:
        """カテゴリ別効果分析"""
        print("\n=== カテゴリ別効果分析 ===")
        
        detailed_report = self.data.get("detailed_report", {})
        category_usage = detailed_report.get("category_usage", {})
        total_strategies = detailed_report.get("summary", {}).get("total_strategies", 0)
        
        effectiveness_analysis = {
            "category_usage_rates": {},
            "category_rankings": [],
            "usage_patterns": {}
        }
        
        # 使用率計算
        for category, count in category_usage.items():
            usage_rate = (count / total_strategies * 100) if total_strategies > 0 else 0
            effectiveness_analysis["category_usage_rates"][category] = {
                "usage_count": count,
                "usage_rate": usage_rate
            }
        
        # ランキング作成
        sorted_categories = sorted(category_usage.items(), key=lambda x: x[1], reverse=True)
        effectiveness_analysis["category_rankings"] = sorted_categories
        
        print("カテゴリ別使用状況 (使用率順):")
        for i, (category, count) in enumerate(sorted_categories, 1):
            usage_rate = (count / total_strategies * 100) if total_strategies > 0 else 0
            print(f"  {i}. {category}: {count}回 ({usage_rate:.1f}%)")
        
        # 使用パターン分析
        strategies = self.data.get("strategies", [])
        category_combinations = defaultdict(int)
        
        for strategy in strategies:
            categories = sorted(strategy.get("categories_used", []))
            if categories:
                combo_key = " + ".join(categories)
                category_combinations[combo_key] += 1
        
        effectiveness_analysis["usage_patterns"] = dict(category_combinations)
        
        print(f"\nカテゴリ組み合わせパターン:")
        for combo, count in sorted(category_combinations.items(), key=lambda x: x[1], reverse=True):
            if count > 1:  # 2回以上使用されたパターンのみ表示
                print(f"  {combo}: {count}回")
        
        return effectiveness_analysis
    
    def analyze_indicator_popularity(self) -> Dict:
        """インジケータ人気度分析"""
        print("\n=== インジケータ人気度分析 ===")
        
        detailed_report = self.data.get("detailed_report", {})
        top_indicators = detailed_report.get("top_indicators", [])
        
        popularity_analysis = {
            "top_indicators": top_indicators,
            "popularity_distribution": {},
            "category_leaders": {}
        }
        
        print("最も使用された新しいインジケータ:")
        for i, (indicator, count) in enumerate(top_indicators[:10], 1):
            total_strategies = detailed_report.get("summary", {}).get("total_strategies", 0)
            usage_rate = (count / total_strategies * 100) if total_strategies > 0 else 0
            print(f"  {i}. {indicator}: {count}回 ({usage_rate:.1f}%)")
        
        # カテゴリ別リーダー分析
        service_data = self.data.get("service_verification", {})
        new_indicators = service_data.get("new_indicators_found", {})
        
        for category, indicators in new_indicators.items():
            category_usage = {}
            for indicator, count in top_indicators:
                if indicator in indicators:
                    category_usage[indicator] = count
            
            if category_usage:
                leader = max(category_usage.items(), key=lambda x: x[1])
                popularity_analysis["category_leaders"][category] = {
                    "leader": leader[0],
                    "usage_count": leader[1],
                    "total_in_category": len([ind for ind, _ in top_indicators if ind in indicators])
                }
        
        print(f"\nカテゴリ別人気リーダー:")
        for category, data in popularity_analysis["category_leaders"].items():
            print(f"  {category}: {data['leader']} ({data['usage_count']}回)")
        
        return popularity_analysis
    
    def generate_comprehensive_report(self) -> Dict:
        """包括的レポート生成"""
        print("\n" + "="*60)
        print("新しいインジケータ統合 - 包括的分析レポート")
        print("="*60)
        
        # 各分析を実行
        coverage = self.analyze_indicator_coverage()
        quality = self.analyze_strategy_quality()
        effectiveness = self.analyze_category_effectiveness()
        popularity = self.analyze_indicator_popularity()
        
        # 総合評価
        print("\n=== 総合評価 ===")
        
        verification_status = self.data.get("verification_status", "UNKNOWN")
        new_indicator_usage_rate = self.data.get("detailed_report", {}).get("summary", {}).get("new_indicator_usage_rate", 0)
        
        evaluation = {
            "overall_status": verification_status,
            "integration_success_rate": new_indicator_usage_rate,
            "key_achievements": [],
            "recommendations": []
        }
        
        # 主要成果
        if new_indicator_usage_rate >= 60:
            evaluation["key_achievements"].append("新しいインジケータの高い使用率を達成")
        if quality["valid_strategies"] / quality["total_strategies"] >= 0.9:
            evaluation["key_achievements"].append("高品質な戦略生成を維持")
        if len(effectiveness["category_rankings"]) >= 4:
            evaluation["key_achievements"].append("多様なカテゴリの活用を実現")
        
        # 推奨事項
        if new_indicator_usage_rate < 70:
            evaluation["recommendations"].append("新しいインジケータの選択重みを調整")
        
        low_usage_categories = [cat for cat, count in effectiveness["category_rankings"] if count < 2]
        if low_usage_categories:
            evaluation["recommendations"].append(f"使用率の低いカテゴリ({', '.join(low_usage_categories)})の条件生成ロジックを改善")
        
        print(f"統合ステータス: {verification_status}")
        print(f"新しいインジケータ使用率: {new_indicator_usage_rate:.1f}%")
        print(f"主要成果: {len(evaluation['key_achievements'])}項目")
        print(f"改善推奨: {len(evaluation['recommendations'])}項目")
        
        comprehensive_report = {
            "coverage_analysis": coverage,
            "quality_analysis": quality,
            "effectiveness_analysis": effectiveness,
            "popularity_analysis": popularity,
            "evaluation": evaluation
        }
        
        return comprehensive_report


def main():
    """メイン実行関数"""
    analyzer = VerificationResultAnalyzer()
    
    if not analyzer.data:
        print("分析対象のデータがありません。先に verify_new_indicators.py を実行してください。")
        return
    
    # 包括的分析実行
    comprehensive_report = analyzer.generate_comprehensive_report()
    
    # 詳細レポートをファイルに保存
    output_file = "comprehensive_analysis_report.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(comprehensive_report, f, ensure_ascii=False, indent=2)
    
    print(f"\n包括的分析レポートを {output_file} に保存しました。")


if __name__ == "__main__":
    main()
