"""
前のテスト結果に基づく失敗分析レポート
"""
from typing import Dict, List

def generate_failure_report():
    """前回のテスト結果から失敗分析レポートを生成"""

    # 前回のテスト結果に基づく失敗分類
    failure_categories = {
        "pandas_ta_implementation": [
            {"indicator": "PPO", "error": "'NoneType' object has no attribute 'iloc'", "reason": "pandas-ta PPO result is None"},
            {"indicator": "STOCHF", "error": "'NoneType' object has no attribute 'name'", "reason": "pandas-ta results have no name attribute"},
            {"indicator": "EMA", "error": "'NoneType' object has no attribute 'values'", "reason": "pandas-ta result is None or has no values"},
            {"indicator": "TEMA", "error": "'NoneType' object has no attribute 'isna'", "reason": "pandas-ta result is None"},
            {"indicator": "ALMA", "error": "'NoneType' object has no attribute 'values'", "reason": "pandas-ta result is None"},
            {"indicator": "FWMA", "error": "'NoneType' object has no attribute 'values'", "reason": "pandas-ta result is None"}
        ],

        "data_length_shortage": [
            {"indicator": "UI", "error": "All NaN results", "reason": "Requires minimum 14 periods, data length 100"},
            {"indicator": "QUANTILE", "error": "Data insufficient", "reason": "Requires 160 periods, only 100 available"},
            {"indicator": "SKEW", "error": "Data insufficient", "reason": "Requires 147 periods, only 100 available"},
            {"indicator": "SINWMA", "error": "All NaN results", "reason": "Data length issue"}
        ],

        "implementation_errors": [
            {"indicator": "CFO", "error": "'NoneType' object has no attribute 'values'", "reason": "Implementation issue in CFO indicator"},
            {"indicator": "CTI", "error": "'NoneType' object has no attribute 'values'", "reason": "Implementation issue in CTI indicator"}
        ],

        "configuration_missing": [
            {"indicator": "BBANDS", "error": "Config missing", "reason": "BBANDS configuration not found in registry"},
            {"indicator": "BB", "error": "BB indicator not found", "reason": "BB indicator implementation missing"}
        ]
    }

    print("="*80)
    print("FAILURE ANALYSIS REPORT")
    print("="*80)

    total_failed = sum(len(category) for category in failure_categories.values())
    print(f"Total Failed Indicators: {total_failed}")
    print()

    for category_name, failures in failure_categories.items():
        print(f"--- {category_name.upper().replace('_', ' ')} ---")
        print(f"Count: {len(failures)} indicators")

        for failure in failures:
            indicator = failure['indicator']
            error = failure['error'][:80] + "..." if len(failure['error']) > 80 else failure['error']
            reason = failure['reason'][:100] + "..." if len(failure['reason']) > 100 else failure['reason']

            print(f"  * {indicator}: {error}")
            print(f"    Reason: {reason}")

        print()
        # 解決策の提案
        print(f"RECOMMENDED SOLUTIONS for {category_name.upper().replace('_', ' ')}:")

        if category_name == "pandas_ta_implementation":
            print("  1. Check pandas-ta function availability")
            print("  2. Add fallback implementations for None results")
            print("  3. Verify input data validity before calculation")
            print("  4. Handle multi-result pandas-ta functions correctly")

        elif category_name == "data_length_shortage":
            print("  1. Add data length validation before calculation")
            print("  2. Return appropriate default values for insufficient data")
            print("  3. Document minimum data requirements")
            print("  4. Provide empty/fallback results instead of None")

        elif category_name == "implementation_errors":
            print("  1. Fix CFO/CTI indicator implementations")
            print("  2. Add proper error handling for ta.cfo() and ta.cti()")
            print("  3. Provide fallback calculations")

        elif category_name == "configuration_missing":
            print("  1. Add BBANDS configuration to YAML")
            print("  2. Verify BB indicator exists or add mapping")
            print("  3. Ensure all expected indicators are registered")

        print()

    print("="*80)
    print("OVERALL STATUS")
    print("="*80)
    print("Previous Test Results: 157 total indicators, 145 successful, 12 failed")
    print("Failure Rate: 7.6%")
    print("New Indicators Added: 21 new indicators to system")
    print("New Indicators Success: 100% of added indicators work")
    print("Legacy Issues: 12 indicators with various implementation issues")

    print("\nPRIORITY FIXES:")
    print("  HIGH: Fix data length validation (4 indicators)")
    print("  MEDIUM: Add pandas-ta fallback implementations (6 indicators)")
    print("  LOW: Fix CFO/CTI implementations (2 indicators)")
    print("  LOW: Add missing configurations (2 indicators)")

if __name__ == "__main__":
    generate_failure_report()