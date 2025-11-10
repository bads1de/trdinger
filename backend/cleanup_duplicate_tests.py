#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重複テストファイルのクリーンアップスクリプト

このスクリプトは、以下のサフィックスを持つ重複テストファイルを削除します：
- _fixed.py
- _comprehensive.py
- _advanced.py
- _extended.py
- _detailed.py

削除前に、対応する基本ファイルが存在することを確認します。
"""

import sys
import io

# Windows環境での文字コード問題を回避
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
from pathlib import Path
from typing import List, Tuple

# 削除対象のサフィックス
SUFFIXES_TO_REMOVE = [
    "_fixed",
    "_comprehensive", 
    "_advanced",
    "_extended",
    "_detailed"
]

def find_duplicate_files(tests_dir: Path) -> List[Tuple[Path, str]]:
    """重複ファイルを検索"""
    duplicates = []
    
    for test_file in tests_dir.rglob("test_*.py"):
        for suffix in SUFFIXES_TO_REMOVE:
            if test_file.stem.endswith(suffix):
                # 基本ファイル名を取得
                base_name = test_file.stem[:-len(suffix)]
                base_file = test_file.parent / f"{base_name}.py"
                
                duplicates.append((test_file, base_file, suffix))
                break
    
    return duplicates

def main():
    tests_dir = Path(__file__).parent / "tests"
    
    if not tests_dir.exists():
        print(f"エラー: testsディレクトリが見つかりません: {tests_dir}")
        return
    
    print("=" * 80)
    print("重複テストファイルのクリーンアップ")
    print("=" * 80)
    
    # 重複ファイルを検索
    duplicates = find_duplicate_files(tests_dir)
    
    if not duplicates:
        print("\n[OK] 削除対象の重複ファイルは見つかりませんでした。")
        return
    
    print(f"\n[INFO] 削除対象ファイル: {len(duplicates)}件\n")
    
    # 削除対象をカテゴリ別に表示
    by_suffix = {}
    for dup_file, base_file, suffix in duplicates:
        if suffix not in by_suffix:
            by_suffix[suffix] = []
        by_suffix[suffix].append((dup_file, base_file))
    
    for suffix, files in sorted(by_suffix.items()):
        print(f"\n{suffix} サフィックス ({len(files)}件):")
        for dup_file, base_file in files:
            rel_path = dup_file.relative_to(tests_dir.parent)
            base_exists = "[OK]" if base_file.exists() else "[!]"
            print(f"  {base_exists} {rel_path}")
    
    print("\n" + "=" * 80)
    print(f"合計: {len(duplicates)}ファイルを削除します")
    print("=" * 80)
    
    # 確認
    response = input("\n削除を実行しますか？ (yes/no): ").strip().lower()
    
    if response != "yes":
        print("\n[CANCEL] キャンセルしました。")
        return
    
    # 削除実行
    deleted_count = 0
    skipped_count = 0
    
    print("\n[DELETE] 削除中...")
    for dup_file, base_file, suffix in duplicates:
        try:
            # 基本ファイルが存在しない場合はスキップ
            if not base_file.exists():
                print(f"  [SKIP] スキップ (基本ファイルなし): {dup_file.name}")
                skipped_count += 1
                continue
            
            dup_file.unlink()
            print(f"  [OK] 削除: {dup_file.relative_to(tests_dir.parent)}")
            deleted_count += 1
            
        except Exception as e:
            print(f"  [ERROR] エラー: {dup_file.name} - {e}")
            skipped_count += 1
    
    print("\n" + "=" * 80)
    print(f"[DONE] 削除完了: {deleted_count}ファイル")
    if skipped_count > 0:
        print(f"[WARN] スキップ: {skipped_count}ファイル")
    print("=" * 80)

if __name__ == "__main__":
    main()