"""
pandas-taに存在する関数を確認
"""
import pandas_ta as ta

# pandas-taの全関数をリストアップ
all_functions = [func for func in dir(ta) if not func.startswith('_')]
print("pandas-taの関数一覧:")
print(sorted(all_functions))

# LINREG関連を検索
linreg_functions = [func for func in all_functions if 'linreg' in func.lower()]
print(f"\nLINREG関連関数: {linreg_functions}")

# 他の関数もチェック
stoch_functions = [func for func in all_functions if 'stoch' in func.lower()]
print(f"STOCH関連関数: {stoch_functions}")

ao_functions = [func for func in all_functions if 'ao' in func.lower()]
print(f"AO関連関数: {ao_functions}")

aroon_functions = [func for func in all_functions if 'aroon' in func.lower()]
print(f"AROON関連関数: {aroon_functions}")

chop_functions = [func for func in all_functions if 'chop' in func.lower()]
print(f"CHOP関連関数: {chop_functions}")

bop_functions = [func for func in all_functions if 'bop' in func.lower()]
print(f"BOP関連関数: {bop_functions}")