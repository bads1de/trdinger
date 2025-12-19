import pandas as pd
import numpy as np
import logging
from app.services.indicators.indicator_orchestrator import TechnicalIndicatorService
from app.services.indicators.config.indicator_config import indicator_registry

# ログを無効化して出力を綺麗にする
logging.getLogger("app.services.indicators").setLevel(logging.CRITICAL)

def check_all_indicators():
    service = TechnicalIndicatorService()
    indicators = indicator_registry.list_indicators()
    
    # サンプルデータ作成
    df = pd.DataFrame({
        'Open': np.random.randn(100) + 100,
        'High': np.random.randn(100) + 102,
        'Low': np.random.randn(100) + 98,
        'Close': np.random.randn(100) + 100,
        'Volume': np.random.rand(100) * 1000
    })
    
    results = []
    print(f"Checking {len(indicators)} indicators...")
    
    for name in sorted(indicators):
        try:
            config = indicator_registry.get_indicator_config(name)
            # デフォルトパラメータ
            params = config.default_values or {}
            
            # 計算実行
            res = service.calculate_indicator(df, name, params)
            
            if res is not None:
                status = "OK"
            else:
                status = "Returned None"
        except Exception as e:
            status = f"ERROR: {str(e)}"
            
        results.append((name, status))
        print(f"[{status[:10]:<10}] {name}")

    print("\n--- Summary of Failures ---")
    errors = [r for r in results if r[1] != "OK"]
    for name, err in errors:
        print(f"{name}: {err}")
    
    print(f"\nTotal: {len(indicators)}, Passed: {len(indicators) - len(errors)}, Failed: {len(errors)}")

if __name__ == "__main__":
    check_all_indicators()