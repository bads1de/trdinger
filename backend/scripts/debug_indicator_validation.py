#!/usr/bin/env python3
"""
指標名検証のデバッグスクリプト
"""

import sys
import os
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.core.services.auto_strategy.models.gene_validation import GeneValidator

def debug_indicator_validation():
    """指標名検証のデバッグ"""
    print("=== 指標名検証のデバッグ ===")
    
    validator = GeneValidator()
    
    print(f"有効な指標タイプ数: {len(validator.valid_indicator_types)}")
    print(f"有効な指標タイプ: {sorted(validator.valid_indicator_types)}")
    
    # 問題のある指標名をテスト
    test_indicators = [
        "HT_DCPHASE",
        "HT_DCPHASE_14",
        "HT_TRENDMODE",
        "HT_TRENDMODE_14", 
        "HT_SINE",
        "HT_SINE_14",
        "SMA",
        "SMA_20",
        "RSI",
        "RSI_14",
    ]
    
    print("\n=== 指標名テスト ===")
    for indicator in test_indicators:
        is_valid = validator._is_indicator_name(indicator)
        print(f"{indicator}: {'有効' if is_valid else '無効'}")
        
        # 詳細分析
        if "_" in indicator:
            base_type = indicator.split("_")[0]
            is_base_valid = base_type in validator.valid_indicator_types
            print(f"  -> ベースタイプ '{base_type}': {'有効' if is_base_valid else '無効'}")

if __name__ == "__main__":
    debug_indicator_validation()
