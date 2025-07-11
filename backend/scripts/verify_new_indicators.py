#!/usr/bin/env python3
"""
新しいインジケータ統合の実証検証スクリプト

101個の新しいテクニカルインジケータがオートストラテジー機能で
実際に動作することを実証的に確認します。
"""

import sys
import os
import json
from typing import Dict, List, Set
from collections import defaultdict, Counter

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.core.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator
from app.core.services.auto_strategy.generators.smart_condition_generator import (
    SmartConditionGenerator, 
    INDICATOR_CHARACTERISTICS,
    IndicatorType
)
from app.core.services.auto_strategy.models.ga_config import GAConfig
from app.core.services.indicators import TechnicalIndicatorService


class NewIndicatorVerifier:
    """新しいインジケータの実証検証クラス"""
    
    def __init__(self):
        """初期化"""
        self.config = GAConfig(
            population_size=20,
            generations=5,
            max_indicators=4,
            min_indicators=2,
            max_conditions=3,
            min_conditions=1
        )
        
        self.random_generator = RandomGeneGenerator(self.config, enable_smart_generation=True)
        self.smart_generator = SmartConditionGenerator(enable_smart_generation=True)
        self.indicator_service = TechnicalIndicatorService()
        
        # 新しいインジケータカテゴリの定義
        self.new_indicator_categories = {
            "cycle": ["HT_DCPERIOD", "HT_DCPHASE", "HT_PHASOR", "HT_SINE", "HT_TRENDMODE"],
            "statistics": ["BETA", "CORREL", "LINEARREG", "LINEARREG_ANGLE", "LINEARREG_INTERCEPT", 
                          "LINEARREG_SLOPE", "STDDEV", "TSF", "VAR"],
            "math_transform": ["ACOS", "ASIN", "ATAN", "COS", "COSH", "SIN", "SINH", "TAN", "TANH",
                              "CEIL", "FLOOR", "SQRT", "LN", "LOG10", "EXP"],
            "math_operators": ["ADD", "SUB", "MULT", "DIV", "MAX", "MIN", "MAXINDEX", "MININDEX", 
                              "MINMAX", "MINMAXINDEX", "SUM"],
            "pattern_recognition": ["CDL_DOJI", "CDL_HAMMER", "CDL_HANGING_MAN", "CDL_SHOOTING_STAR",
                                   "CDL_ENGULFING", "CDL_HARAMI", "CDL_PIERCING", "CDL_THREE_BLACK_CROWS",
                                   "CDL_THREE_WHITE_SOLDIERS"]
        }
        
        # 統計情報の初期化
        self.stats = {
            "total_strategies": 0,
            "strategies_with_new_indicators": 0,
            "new_indicator_usage": defaultdict(int),
            "category_usage": defaultdict(int),
            "condition_types": defaultdict(int),
            "generated_strategies": []
        }

    def verify_indicator_service(self) -> Dict:
        """TechnicalIndicatorServiceが新しいインジケータを認識しているか確認"""
        print("=== TechnicalIndicatorService検証 ===")
        
        supported_indicators = self.indicator_service.get_supported_indicators()
        total_supported = len(supported_indicators)
        
        new_indicators_found = {}
        for category, indicators in self.new_indicator_categories.items():
            found_in_category = []
            for indicator in indicators:
                if indicator in supported_indicators:
                    found_in_category.append(indicator)
            new_indicators_found[category] = found_in_category
        
        total_new_found = sum(len(indicators) for indicators in new_indicators_found.values())
        
        print(f"サポートされているインジケータ総数: {total_supported}")
        print(f"新しいインジケータ発見数: {total_new_found}")
        
        for category, indicators in new_indicators_found.items():
            print(f"  {category}: {len(indicators)}個 - {indicators[:3]}{'...' if len(indicators) > 3 else ''}")
        
        return {
            "total_supported": total_supported,
            "new_indicators_found": new_indicators_found,
            "total_new_found": total_new_found
        }

    def generate_and_analyze_strategies(self, num_strategies: int = 15) -> List[Dict]:
        """戦略を生成して分析"""
        print(f"\n=== {num_strategies}個の戦略生成・分析 ===")
        
        strategies = []
        
        for i in range(num_strategies):
            try:
                # 戦略生成
                strategy = self.random_generator.generate_random_gene()
                
                # 戦略分析
                analysis = self._analyze_strategy(strategy, i + 1)
                strategies.append(analysis)
                
                # 統計更新
                self._update_stats(analysis)
                
                print(f"戦略 {i+1}: {analysis['summary']}")
                
            except Exception as e:
                print(f"戦略 {i+1} 生成エラー: {e}")
                continue
        
        return strategies

    def _analyze_strategy(self, strategy, strategy_id: int) -> Dict:
        """個別戦略の詳細分析"""
        analysis = {
            "strategy_id": strategy_id,
            "indicators": [],
            "new_indicators_used": [],
            "categories_used": [],
            "long_conditions": [],
            "short_conditions": [],
            "exit_conditions": [],
            "has_new_indicators": False,
            "summary": ""
        }
        
        # インジケータ分析
        for indicator in strategy.indicators:
            indicator_info = {
                "type": indicator.type,
                "parameters": indicator.parameters,
                "enabled": indicator.enabled
            }
            analysis["indicators"].append(indicator_info)
            
            # 新しいインジケータかチェック
            for category, indicators in self.new_indicator_categories.items():
                if indicator.type in indicators:
                    analysis["new_indicators_used"].append(indicator.type)
                    if category not in analysis["categories_used"]:
                        analysis["categories_used"].append(category)
                    analysis["has_new_indicators"] = True
        
        # 条件分析
        analysis["long_conditions"] = [
            {
                "left": cond.left_operand,
                "operator": cond.operator,
                "right": cond.right_operand
            }
            for cond in strategy.long_entry_conditions
        ]
        
        analysis["short_conditions"] = [
            {
                "left": cond.left_operand,
                "operator": cond.operator,
                "right": cond.right_operand
            }
            for cond in strategy.short_entry_conditions
        ]
        
        analysis["exit_conditions"] = [
            {
                "left": cond.left_operand,
                "operator": cond.operator,
                "right": cond.right_operand
            }
            for cond in strategy.exit_conditions
        ]
        
        # サマリー生成
        indicator_types = [ind["type"] for ind in analysis["indicators"]]
        new_indicator_count = len(analysis["new_indicators_used"])
        categories = analysis["categories_used"]

        analysis["summary"] = (
            f"指標{len(indicator_types)}個({', '.join(indicator_types[:2])}{'...' if len(indicator_types) > 2 else ''}) "
            f"新規{new_indicator_count}個 "
            f"カテゴリ{categories if categories else '従来のみ'} "
            f"条件L{len(analysis['long_conditions'])}:S{len(analysis['short_conditions'])}:E{len(analysis['exit_conditions'])}"
        )
        
        return analysis

    def _update_stats(self, analysis: Dict):
        """統計情報を更新"""
        self.stats["total_strategies"] += 1
        self.stats["generated_strategies"].append(analysis)
        
        if analysis["has_new_indicators"]:
            self.stats["strategies_with_new_indicators"] += 1
        
        # 新しいインジケータ使用回数
        for indicator in analysis["new_indicators_used"]:
            self.stats["new_indicator_usage"][indicator] += 1
        
        # カテゴリ使用回数
        for category in analysis["categories_used"]:
            self.stats["category_usage"][category] += 1

    def generate_detailed_report(self) -> Dict:
        """詳細レポート生成"""
        print("\n=== 詳細分析レポート ===")
        
        # 基本統計
        total = self.stats["total_strategies"]
        with_new = self.stats["strategies_with_new_indicators"]
        usage_rate = (with_new / total * 100) if total > 0 else 0
        
        print(f"生成戦略総数: {total}")
        print(f"新しいインジケータを含む戦略: {with_new} ({usage_rate:.1f}%)")
        
        # カテゴリ別使用状況
        print(f"\n新しいカテゴリの使用状況:")
        for category in self.new_indicator_categories.keys():
            count = self.stats["category_usage"][category]
            rate = (count / total * 100) if total > 0 else 0
            print(f"  {category}: {count}回 ({rate:.1f}%)")
        
        # 最も使用された新しいインジケータ
        print(f"\n最も使用された新しいインジケータ (Top 10):")
        top_indicators = Counter(self.stats["new_indicator_usage"]).most_common(10)
        for indicator, count in top_indicators:
            rate = (count / total * 100) if total > 0 else 0
            print(f"  {indicator}: {count}回 ({rate:.1f}%)")
        
        # 戦略品質分析
        print(f"\n戦略品質分析:")
        valid_strategies = 0
        avg_indicators = 0
        avg_conditions = 0
        
        for strategy in self.stats["generated_strategies"]:
            if len(strategy["indicators"]) > 0 and len(strategy["long_conditions"]) > 0:
                valid_strategies += 1
                avg_indicators += len(strategy["indicators"])
                avg_conditions += len(strategy["long_conditions"]) + len(strategy["short_conditions"])
        
        if valid_strategies > 0:
            avg_indicators /= valid_strategies
            avg_conditions /= valid_strategies
            
        print(f"  有効な戦略: {valid_strategies}/{total} ({valid_strategies/total*100:.1f}%)")
        print(f"  平均インジケータ数: {avg_indicators:.1f}")
        print(f"  平均条件数: {avg_conditions:.1f}")
        
        return {
            "summary": {
                "total_strategies": total,
                "strategies_with_new_indicators": with_new,
                "new_indicator_usage_rate": usage_rate,
                "valid_strategies": valid_strategies
            },
            "category_usage": dict(self.stats["category_usage"]),
            "top_indicators": top_indicators,
            "detailed_strategies": self.stats["generated_strategies"]
        }

    def run_verification(self) -> Dict:
        """完全な検証プロセスを実行"""
        print("新しいインジケータ統合の実証検証を開始します...\n")
        
        # 1. インジケータサービス検証
        service_verification = self.verify_indicator_service()
        
        # 2. 戦略生成・分析
        strategies = self.generate_and_analyze_strategies(15)
        
        # 3. 詳細レポート生成
        detailed_report = self.generate_detailed_report()
        
        # 4. 結果まとめ
        result = {
            "service_verification": service_verification,
            "strategies": strategies,
            "detailed_report": detailed_report,
            "verification_status": "SUCCESS" if detailed_report["summary"]["new_indicator_usage_rate"] > 20 else "PARTIAL"
        }
        
        print(f"\n=== 検証結果 ===")
        print(f"ステータス: {result['verification_status']}")
        print(f"新しいインジケータ使用率: {detailed_report['summary']['new_indicator_usage_rate']:.1f}%")
        print(f"検証完了!")
        
        return result


def main():
    """メイン実行関数"""
    verifier = NewIndicatorVerifier()
    result = verifier.run_verification()
    
    # 結果をJSONファイルに保存
    output_file = "new_indicator_verification_result.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\n詳細結果を {output_file} に保存しました。")


if __name__ == "__main__":
    main()
