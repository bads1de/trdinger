"""特徴量生成の調査スクリプト"""
import sys
import pandas as pd
from pathlib import Path

# パスを追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.ml.feature_engineering.feature_engineering_service import (
    FeatureEngineeringService,
    DEFAULT_FEATURE_ALLOWLIST,
)
from database.connection import SessionLocal
from database.models import OHLCVData

print("=" * 80)
print("特徴量生成システムの調査")
print("=" * 80)

# 1. デフォルト設定の確認
print("\n[1] デフォルト設定")
print(f"  DEFAULT_FEATURE_ALLOWLIST: {DEFAULT_FEATURE_ALLOWLIST}")
print(f"  タイプ: {type(DEFAULT_FEATURE_ALLOWLIST)}")
if DEFAULT_FEATURE_ALLOWLIST is None:
    print("  => 全特徴量を使用（研究モード）")
else:
    print(f"  => {len(DEFAULT_FEATURE_ALLOWLIST)}個の特徴量に制限")

# 2. データベースからサンプルデータを取得
print("\n[2] データベースからサンプルデータを取得")
db = SessionLocal()
try:
    ohlcv_records = db.query(OHLCVData).filter(
        OHLCVData.symbol == "BTC/USDT:USDT",
        OHLCVData.timeframe == "1h"
    ).order_by(OHLCVData.timestamp.desc()).limit(500).all()
    
    print(f"  取得件数: {len(ohlcv_records)}件")
    
    if len(ohlcv_records) < 100:
        print("  警告: データが不足しています（最低100件必要）")
        sys.exit(1)
    
    # DataFrameに変換
    df = pd.DataFrame([{
        'timestamp': r.timestamp,
        'open': r.open,
        'high': r.high,
        'low': r.low,
        'close': r.close,
        'volume': r.volume,
    } for r in ohlcv_records])
    
    df = df.sort_values('timestamp').reset_index(drop=True)
    df.set_index('timestamp', inplace=True)
    
    print(f"  期間: {df.index[0]} ~ {df.index[-1]}")
    print(f"  基本カラム: {list(df.columns)}")
    
finally:
    db.close()

# 3. 特徴量エンジニアリングサービスで特徴量生成
print("\n[3] 特徴量生成を実行")
feature_service = FeatureEngineeringService()

try:
    features_df = feature_service.calculate_advanced_features(
        ohlcv_data=df,
        funding_rate_data=None,
        open_interest_data=None,
    )
    
    print(f"  生成された特徴量数: {len(features_df.columns)}個")
    print(f"  データ行数: {len(features_df)}行")
    print(f"  欠損値: {features_df.isnull().sum().sum()}個")
    
    # 4. 特徴量のカテゴリ分析
    print("\n[4] 特徴量のカテゴリ分析")
    
    categories = {
        'テクニカル指標': [],
        '価格関連': [],
        'ボリューム': [],
        'ボラティリティ': [],
        'モメンタム': [],
        '市場レジーム': [],
        '建玉残高(OI)': [],
        'ファンディングレート': [],
        '複合指標': [],
        '暗号通貨特化': [],
        '高度な特徴量': [],
        'その他': [],
    }
    
    for col in features_df.columns:
        col_lower = col.lower()
        
        if any(x in col_lower for x in ['rsi', 'macd', 'bb_', 'ma_', 'atr', 'ema', 'sma']):
            categories['テクニカル指標'].append(col)
        elif any(x in col_lower for x in ['price', 'high', 'low', 'close', 'open', 'range']):
            categories['価格関連'].append(col)
        elif 'volume' in col_lower:
            categories['ボリューム'].append(col)
        elif 'volatility' in col_lower or 'vol_' in col_lower:
            categories['ボラティリティ'].append(col)
        elif any(x in col_lower for x in ['momentum', 'roc']):
            categories['モメンタム'].append(col)
        elif any(x in col_lower for x in ['regime', 'trend']):
            categories['市場レジーム'].append(col)
        elif 'oi_' in col_lower or 'open_interest' in col_lower:
            categories['建玉残高(OI)'].append(col)
        elif 'fr_' in col_lower or 'funding' in col_lower:
            categories['ファンディングレート'].append(col)
        elif any(x in col_lower for x in ['heat', 'stress', 'balance']):
            categories['複合指標'].append(col)
        elif any(x in col_lower for x in ['crypto', 'correlation']):
            categories['暗号通貨特化'].append(col)
        elif any(x in col_lower for x in ['lag', 'diff', 'rolling', 'pct']):
            categories['高度な特徴量'].append(col)
        else:
            categories['その他'].append(col)
    
    for category, features in categories.items():
        if features:
            print(f"\n  [{category}] ({len(features)}個)")
            for feat in sorted(features)[:10]:  # 最初の10個のみ表示
                print(f"    - {feat}")
            if len(features) > 10:
                print(f"    ... 他{len(features) - 10}個")
    
    # 5. 統計情報
    print("\n[5] 特徴量の統計情報")
    print(f"  数値型: {features_df.select_dtypes(include=['float64', 'int64']).shape[1]}個")
    print(f"  オブジェクト型: {features_df.select_dtypes(include=['object']).shape[1]}個")
    print(f"  欠損値が多い特徴量（50%以上）:")
    
    missing_pct = (features_df.isnull().sum() / len(features_df) * 100).sort_values(ascending=False)
    high_missing = missing_pct[missing_pct > 50]
    if len(high_missing) > 0:
        for feat, pct in high_missing.head(10).items():
            print(f"    - {feat}: {pct:.1f}%")
    else:
        print("    なし（良好）")
    
    # 6. サンプルデータ
    print("\n[6] 生成された特徴量のサンプル（最初の5行、最初の10列）")
    print(features_df.iloc[:5, :10].to_string())
    
    print("\n" + "=" * 80)
    print("調査完了")
    print("=" * 80)
    
except Exception as e:
    print(f"\nエラー: {e}")
    import traceback
    traceback.print_exc()
