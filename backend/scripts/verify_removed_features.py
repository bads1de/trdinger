"""削除された特徴量の検証 (2025-11-12更新)"""
import sys
from pathlib import Path

import pandas as pd
import sqlite3

# パスを追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.ml.feature_engineering.feature_engineering_service import (
    FeatureEngineeringService,
)

# データベースからサンプルデータを取得
conn = sqlite3.connect("C:/Users/buti3/trading/backend/trdinger.db")
df = pd.read_sql_query("SELECT * FROM ohlcv_data LIMIT 200", conn)
conn.close()

# DataFrameの準備
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.set_index("timestamp")

# 特徴量生成
service = FeatureEngineeringService()
result = service.calculate_advanced_features(df)

print("=" * 80)
print("削除された特徴量の検証 (2025-11-12)")
print("=" * 80)

# 削除推奨特徴量（19個）
# 高相関による削除（5個）
high_corr_removed = [
    "macd",  # macd_signalと高相関(r=0.957)
    "Stochastic_K",  # Stochastic_Dと高相関(r=0.952)
    "Near_Resistance",  # Near_Supportと高相関(r=1.000)
    "MA_Long",  # ADと高相関(r=0.950)
    "BB_Position",  # bb_position_20と高相関(r=0.968)
]

# 低重要度による削除（14個）
low_importance_removed = [
    "close_lag_24",
    "cumulative_returns_24",
    "Close_mean_20",
    "Local_Max",
    "Aroon_Up",
    "BB_Lower",
    "Resistance_Level",
    "BB_Middle",
    "stochastic_k",
    "rsi_14",
    "bb_lower_20",
    "bb_upper_20",
    "stochastic_d",
    "Local_Min",
]

# 全削除対象
removed_features = high_corr_removed + low_importance_removed

print(f"\n削除対象の特徴量: {len(removed_features)}個")
print(f"  高相関による削除: {len(high_corr_removed)}個")
print(f"  低重要度による削除: {len(low_importance_removed)}個")

# 検証
missing_count = 0
found_count = 0
found_features = []

print("\n=== 高相関による削除（5個） ===")
for feat in high_corr_removed:
    if feat in result.columns:
        print(f"  [NG] {feat} - まだ存在しています")
        found_count += 1
        found_features.append(feat)
    else:
        print(f"  [OK] {feat} - 正常に削除されました")
        missing_count += 1

print("\n=== 低重要度による削除（14個） ===")
for feat in low_importance_removed:
    if feat in result.columns:
        print(f"  [NG] {feat} - まだ存在しています")
        found_count += 1
        found_features.append(feat)
    else:
        print(f"  [OK] {feat} - 正常に削除されました")
        missing_count += 1

print(f"\n結果:")
print(f"  正常に削除: {missing_count}個 / {len(removed_features)}個")
print(f"  まだ存在: {found_count}個")

if found_features:
    print(f"\n存在する特徴量: {', '.join(found_features)}")

# 基本カラムを除いた特徴量数をカウント
basic_columns = ["open", "high", "low", "close", "volume"]
feature_columns = [col for col in result.columns if col not in basic_columns]

print(f"\n現在の特徴量数:")
print(f"  全カラム: {len(result.columns)}個")
print(f"  特徴量のみ: {len(feature_columns)}個 (基本カラム除く)")
print(f"  期待値: 60個 (79個 - 19個)")

if found_count == 0:
    print("\n[SUCCESS] すべての削除対象特徴量が正常に削除されました！")
    if len(feature_columns) == 60:
        print("[SUCCESS] 特徴量数も期待値（60個）と一致しています！")
    else:
        print(
            f"[INFO] 特徴量数は{len(feature_columns)}個です "
            f"(期待値60個との差: {len(feature_columns) - 60}個)"
        )
else:
    print(f"\n[WARNING] {found_count}個の特徴量がまだ存在しています")

print("=" * 80)
