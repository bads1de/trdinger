import pandas as pd
import numpy as np
from app.services.ml.feature_engineering.time_anomaly_features import TimeAnomalyFeatures

def test_time_anomaly():
    # 2024-01-01 (月) からの1時間足データを作成
    index = pd.date_range(start='2024-01-01', periods=200, freq='h')
    df = pd.DataFrame(index=index)
    df['close'] = np.random.randn(len(index)).cumsum() + 100
    df['volume'] = np.random.randint(100, 1000, len(index))
    
    calculator = TimeAnomalyFeatures()
    result = calculator.calculate_features(df)
    
    print("Columns generated:", result.columns.tolist())
    print("\nSample features (first 5 rows):")
    print(result[['time_hour_sin', 'time_session_asia', 'time_is_weekend', 'time_is_monday']].head())
    
    # 検証
    assert 'time_hour_sin' in result.columns
    assert 'time_session_asia' in result.columns
    assert 'time_is_month_end' in result.columns
    
    # 相互作用特徴量のチェック
    if 'volume' in df.columns:
        assert 'time_interaction_vol_us' in result.columns
    if 'close' in df.columns:
        assert 'time_interaction_volatility_us' in result.columns
        assert 'time_adaptive_vol_ratio' in result.columns
        assert 'time_micro_illiquidity' in result.columns
        
    # 経過時間特徴量のチェック
    assert 'time_since_tokyo' in result.columns
    assert 'time_since_london' in result.columns
    
    # ロジック確認: 10:00 のデータ
    # 2024-01-01 10:00:00
    target_row = result.iloc[10]
    # 東京(0時開始)から10時間
    assert target_row['time_since_tokyo'] == 10
    # ロンドン(8時開始)から2時間
    assert target_row['time_since_london'] == 2

    # 月曜判定のチェック
    assert result.iloc[0]['time_is_monday'] == 1
    
    print("\nTest passed successfully!")

if __name__ == "__main__":
    test_time_anomaly()
