"""全特徴量のリストを取得"""
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

print(f"生成された特徴量数: {len(result.columns)}個")
print("\n全特徴量リスト:")
all_features = sorted(result.columns)
for i, col in enumerate(all_features, 1):
    print(f"{i:3d}. {col}")

# 削除推奨特徴量
remove_features = [
    'atr_20',
    'body_size',
    'lower_shadow',
    'price_change_1',
    'price_change_20',
    'price_change_5',
    'returns',
    'volume_ma_20'
]

print(f"\n削除推奨特徴量: {len(remove_features)}個")
for feat in remove_features:
    print(f"  - {feat}")

# 残す特徴量（基本カラムとDB用カラムを除く）
essential_columns = ['open', 'high', 'low', 'close', 'volume']
db_columns = ['id', 'symbol', 'timeframe', 'created_at', 'updated_at']
# 大文字Returnsも除外（returnsの重複）
exclude_features = essential_columns + db_columns + remove_features + ['Returns']
keep_features = [col for col in all_features if col not in exclude_features]

print(f"\n残す特徴量: {len(keep_features)}個")

# feature_allowlist用のリストを出力
print("\n\nfeature_allowlist = [")
for i, feat in enumerate(sorted(keep_features)):
    comma = "," if i < len(keep_features) - 1 else ""
    print(f'    "{feat}"{comma}')
print("]")

print(f"\n合計: {len(keep_features)}個の特徴量")
