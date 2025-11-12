"""削除された特徴量の検証"""
import sys
import pandas as pd
import sqlite3
from pathlib import Path

# パスを追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.ml.feature_engineering.feature_engineering_service import (
    FeatureEngineeringService,
)

# データベースからサンプルデータを取得
conn = sqlite3.connect('C:/Users/buti3/trading/backend/trdinger.db')
df = pd.read_sql_query('SELECT * FROM ohlcv_data LIMIT 200', conn)
conn.close()

# DataFrameの準備
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp')

# 特徴量生成
service = FeatureEngineeringService()
result = service.calculate_advanced_features(df)

print("=" * 80)
print("削除された特徴量の検証")
print("=" * 80)

# 削除推奨特徴量
removed_features = [
    'atr_20',
    'body_size',
    'lower_shadow',
    'price_change_1',
    'price_change_20',
    'price_change_5',
    'returns',
    'volume_ma_20'
]

print(f"\n削除対象の特徴量: {len(removed_features)}個")

missing_count = 0
found_count = 0

for feat in removed_features:
    if feat in result.columns:
        print(f"  [NG] {feat} - まだ存在しています")
        found_count += 1
    else:
        print(f"  [OK] {feat} - 正常に削除されました")
        missing_count += 1

print(f"\n結果:")
print(f"  正常に削除: {missing_count}個")
print(f"  まだ存在: {found_count}個")

if found_count == 0:
    print("\n[SUCCESS] すべての削除対象特徴量が正常に削除されました！")
else:
    print(f"\n[WARNING] {found_count}個の特徴量がまだ存在しています")

print(f"\n現在の特徴量数: {len(result.columns)}個")
print("=" * 80)
