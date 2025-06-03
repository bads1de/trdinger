#!/usr/bin/env python3
"""
インポートエラーを一括修正するスクリプト
"""

import os
import re

def fix_file(file_path, old_pattern, new_pattern):
    """ファイル内の文字列を置換"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 置換実行
        new_content = re.sub(old_pattern, new_pattern, content)
        
        if content != new_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"✅ 修正完了: {file_path}")
            return True
        else:
            print(f"📝 変更なし: {file_path}")
            return False
    except Exception as e:
        print(f"❌ エラー: {file_path} - {e}")
        return False

def main():
    """メイン処理"""
    print("🔧 インポートエラー一括修正開始")
    print("=" * 50)
    
    # 修正対象ファイル
    files_to_fix = [
        "backend/app/core/services/indicators/momentum_indicators.py",
        "backend/app/core/services/indicators/volatility_indicators.py",
        "backend/app/core/services/indicators/volume_indicators.py",
    ]
    
    # 修正パターン
    patterns = [
        (r'TALibAdapter\.', 'MomentumAdapter.'),  # momentum_indicators.py用
    ]
    
    total_fixed = 0
    
    # momentum_indicators.pyの修正
    file_path = "backend/app/core/services/indicators/momentum_indicators.py"
    if os.path.exists(file_path):
        replacements = [
            (r'TALibAdapter\.williams_r', 'MomentumAdapter.williams_r'),
            (r'TALibAdapter\.momentum', 'MomentumAdapter.momentum'),
            (r'TALibAdapter\.roc', 'MomentumAdapter.roc'),
            (r'TALibAdapter\.adx', 'MomentumAdapter.adx'),
            (r'TALibAdapter\.aroon', 'MomentumAdapter.aroon'),
            (r'TALibAdapter\.mfi', 'MomentumAdapter.mfi'),
        ]
        
        for old_pattern, new_pattern in replacements:
            if fix_file(file_path, old_pattern, new_pattern):
                total_fixed += 1
    
    # volatility_indicators.pyの修正
    file_path = "backend/app/core/services/indicators/volatility_indicators.py"
    if os.path.exists(file_path):
        # インポート修正
        fix_file(file_path, r'from \.talib_adapter import TALibAdapter, TALibCalculationError', 
                'from .adapters import VolatilityAdapter, TALibCalculationError')
        
        # 使用箇所修正
        replacements = [
            (r'TALibAdapter\.atr', 'VolatilityAdapter.atr'),
            (r'TALibAdapter\.bollinger_bands', 'VolatilityAdapter.bollinger_bands'),
            (r'TALibAdapter\.natr', 'VolatilityAdapter.natr'),
            (r'TALibAdapter\.trange', 'VolatilityAdapter.trange'),
        ]
        
        for old_pattern, new_pattern in replacements:
            if fix_file(file_path, old_pattern, new_pattern):
                total_fixed += 1
    
    # volume_indicators.pyの修正
    file_path = "backend/app/core/services/indicators/volume_indicators.py"
    if os.path.exists(file_path):
        # インポート修正
        fix_file(file_path, r'from \.talib_adapter import TALibAdapter, TALibCalculationError', 
                'from .adapters import VolumeAdapter, TALibCalculationError')
        
        # 使用箇所修正
        replacements = [
            (r'TALibAdapter\.obv', 'VolumeAdapter.obv'),
            (r'TALibAdapter\.ad', 'VolumeAdapter.ad'),
            (r'TALibAdapter\.adosc', 'VolumeAdapter.adosc'),
        ]
        
        for old_pattern, new_pattern in replacements:
            if fix_file(file_path, old_pattern, new_pattern):
                total_fixed += 1
    
    print("=" * 50)
    print(f"📊 修正完了: {total_fixed}箇所")
    print("🎉 一括修正処理完了")

if __name__ == "__main__":
    main()
