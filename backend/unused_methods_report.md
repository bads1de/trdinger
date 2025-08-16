# 未使用メソッド・関数レポート (60%信頼度)

## API 関数 (FastAPI エンドポイント)

### ML Management API

- `app/api/ml_management.py:29` - `get_models()`
- `app/api/ml_management.py:72` - `delete_all_models_legacy()`
- `app/api/ml_management.py:190` - `get_current_model()`
- `app/api/ml_management.py:227` - `get_ml_config()`

### ML Training API

- `app/api/ml_training.py:194` - `start_ml_training()`
- `app/api/ml_training.py:248` - `get_ml_training_status()`
- `app/api/ml_training.py:259` - `get_ml_model_info()`
- `app/api/ml_training.py:273` - `stop_ml_training()`

### Technical Indicators

- `app/services/indicators/technical_indicators/math_operators.py:57` - `div()`
- `app/services/indicators/technical_indicators/math_operators.py:94` - `sum_values()`
- `app/services/indicators/technical_indicators/math_transform.py:27` - `acos()`
- `app/services/indicators/technical_indicators/math_transform.py:50` - `asin()`
- `app/services/indicators/technical_indicators/math_transform.py:64` - `atan()`
- `app/services/indicators/technical_indicators/math_transform.py:100` - `ln()`

### Pattern Recognition

- `app/services/indicators/technical_indicators/pattern_recognition.py:82` - `cdl_hanging_man()`
- `app/services/indicators/technical_indicators/pattern_recognition.py:167` - `cdl_harami()`
- `app/services/indicators/technical_indicators/pattern_recognition.py:195` - `cdl_piercing()`
- `app/services/indicators/technical_indicators/pattern_recognition.py:223` - `cdl_dark_cloud_cover()`
- `app/services/indicators/technical_indicators/pattern_recognition.py:309` - `cdl_three_black_crows()`
- `app/services/indicators/technical_indicators/pattern_recognition.py:337` - `cdl_three_white_soldiers()`
- `app/services/indicators/technical_indicators/pattern_recognition.py:366` - `cdl_marubozu()`
- `app/services/indicators/technical_indicators/pattern_recognition.py:394` - `cdl_spinning_top()`
- `app/services/indicators/technical_indicators/pattern_recognition.py:426` - `doji()`
- `app/services/indicators/technical_indicators/pattern_recognition.py:431` - `hammer()`
- `app/services/indicators/technical_indicators/pattern_recognition.py:436` - `engulfing_pattern()`
- `app/services/indicators/technical_indicators/pattern_recognition.py:441` - `morning_star()`
- `app/services/indicators/technical_indicators/pattern_recognition.py:446` - `evening_star()`


## 統計情報

- **総未使用メソッド・関数数**: 約 200 個
- **最も多いカテゴリ**: ML 関連 (約 80 個)
- **API 関数**: 約 25 個
- **データベースリポジトリ**: 約 30 個
- **Auto Strategy 関連**: 約 50 個

## 推奨事項

1. **高優先度**: API エンドポイントの未使用関数は削除を検討
2. **中優先度**: ユーティリティ関数やヘルパーメソッドは将来使用される可能性があるため慎重に判断
3. **低優先度**: テスト関連のメソッドは開発中に使用される可能性があるため保持を推奨

注意: 60%信頼度のため、実際の使用状況を確認してから削除することを強く推奨します。
