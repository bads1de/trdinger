import sys
import os
from pathlib import Path
import pandas as pd
from sqlalchemy import select

# プロジェクトルートをパスに追加
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "backend"))

from backend.database.connection import SessionLocal
from backend.database.models import FundingRateData

def inspect_funding_rates():
    db = SessionLocal()
    try:
        # 1. 最新の生データを確認
        print("--- Latest 10 Raw Funding Rates ---")
        stmt = select(FundingRateData).order_by(FundingRateData.timestamp.desc()).limit(10)
        results = db.execute(stmt).scalars().all()
        
        data = []
        for r in results:
            print(f"Time: {r.timestamp}, Rate: {r.funding_rate}")
            data.append(r.funding_rate)
            
        # 2. 全体の統計量を確認
        print("\n--- Statistics (All Data) ---")
        all_rates_stmt = select(FundingRateData.funding_rate)
        all_rates = db.execute(all_rates_stmt).scalars().all()
        
        if not all_rates:
            print("No data found.")
            return

        s = pd.Series(all_rates)
        print(s.describe())
        
        # 3. 分布の特性（0.01% = 0.0001 付近の分布）
        baseline = 0.0001
        print(f"\n--- Distribution Characteristics ---")
        print(f"Exact Baseline (0.01%): {(s == baseline).sum()} / {len(s)} ({ (s == baseline).mean():.1%})")
        print(f"Zero: {(s == 0).sum()}")
        print(f"Negative: {(s < 0).sum()}")
        print(f"High Premium (> 0.05%): {(s > 0.0005).sum()}")
        
    finally:
        db.close()

if __name__ == "__main__":
    inspect_funding_rates()
