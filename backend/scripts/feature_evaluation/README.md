# 特徴量評価スクリプト

機械学習モデルで使用する特徴量の重要度を評価し、低重要度の特徴量を特定するためのスクリプト群です。

## 概要

このディレクトリには、以下の4つのスクリプトが含まれています：

1. **`common_feature_evaluator.py`** - 共通ユーティリティクラス
2. **`detect_low_importance_features.py`** - XGBoost/LightGBMでの低重要度特徴検出
3. **`analyze_feature_importance.py`** - RandomForestでの特徴量重要度分析
4. **`evaluate_feature_performance.py`** - 複数モデルでの特徴量パフォーマンス評価
5. **`run_unified_analysis.py`** - 統合分析スクリプト（推奨）

## 推奨: 統合スクリプトの使用

**`run_unified_analysis.py`** を使用することで、3つの分析を一貫したラベル生成設定で統合して実行できます。

### 基本的な使い方

```bash
cd backend

# デフォルト設定で実行（設定ファイルからラベル生成設定を読み込み）
python -m scripts.feature_evaluation.run_unified_analysis

# 特定のプリセットを指定して実行
python -m scripts.feature_evaluation.run_unified_analysis --preset 4h_4bars

# パラメータをカスタマイズ
python -m scripts.feature_evaluation.run_unified_analysis \
    --symbol BTC/USDT:USDT \
    --timeframe 1h \
    --limit 2000 \
    --preset 1h_4bars_dynamic \
    --output-dir backend/results/feature_analysis
```

### コマンドライン引数

- `--symbol`: 分析対象シンボル（デフォルト: `BTC/USDT:USDT`）
- `--timeframe`: 時間足（デフォルト: `1h`）
- `--limit`: データ取得件数（デフォルト: `2000`）
- `--preset`: ラベル生成プリセット名（例: `4h_4bars`, `1h_4bars_dynamic`）
- `--output-dir`: 出力ディレクトリ（デフォルト: `backend/results/feature_analysis`）

### 利用可能なプリセット

以下のラベル生成プリセットが利用可能です：

- **`15m_4bars`**: 15分足、4本先（1時間先）、0.1%閾値
- **`30m_4bars`**: 30分足、4本先（2時間先）、0.15%閾値
- **`1h_4bars`**: 1時間足、4本先（4時間先）、0.2%閾値
- **`4h_4bars`**: 4時間足、4本先（16時間先）、0.2%閾値（デフォルト推奨）
- **`1d_4bars`**: 1日足、4本先（4日先）、0.5%閾値
- **`4h_4bars_dynamic`**: 4時間足、動的閾値（KBinsDiscretizer）
- **`1h_4bars_dynamic`**: 1時間足、動的閾値（KBinsDiscretizer）

### 出力ファイル

統合分析の結果は以下のファイルに保存されます：

```
backend/results/feature_analysis/
├── feature_analysis_20240101_120000.json  # タイムスタンプ付き結果
├── features_to_remove_20240101_120000.csv # 削除推奨特徴リスト
├── feature_analysis_latest.json           # 最新の結果
└── low_importance/                        # 個別分析結果
    ├── low_importance_features_report.md
    ├── feature_importance_detailed.csv
    └── features_to_remove_auto.json
```

### 出力JSON形式

```json
{
  "timestamp": "2024-01-01T12:00:00",
  "symbol": "BTC/USDT:USDT",
  "timeframe": "1h",
  "label_config": {
    "use_preset": true,
    "preset_name": "4h_4bars",
    "timeframe": "4h",
    "horizon_n": 4,
    "threshold": 0.002,
    "threshold_method": "FIXED"
  },
  "low_importance_analysis": {
    "total_features": 150,
    "low_importance_count": 50,
    "low_importance_features": ["feature1", "feature2", ...]
  },
  "importance_analysis": {
    "total_features": 150,
    "low_importance_count": 45,
    "low_importance_features": ["featureA", "featureB", ...]
  },
  "performance_analysis": {
    "models_used": ["lightgbm", "xgboost"],
    "recommendations": {
      "lightgbm": ["feature1", "feature2", ...],
      "xgboost": ["feature3", "feature4", ...]
    }
  },
  "recommended_production_allowlist": {
    "features_to_remove": ["feature1", "feature2", ...],
    "features_to_remove_count": 40,
    "note": "production allowlistは全特徴から削除推奨特徴を除外したもの"
  },
  "analysis_summary": {
    "total_features": 150,
    "analyses_completed": 3,
    "recommended_removal_count": 40,
    "models_used": ["lightgbm", "xgboost"]
  }
}
```

## ラベル生成設定

### 設定ファイルでの指定

`unified_config.py`の`LabelGenerationConfig`でデフォルト設定を指定できます：

```python
# 環境変数での設定例
ML__TRAINING__LABEL_GENERATION__USE_PRESET=true
ML__TRAINING__LABEL_GENERATION__DEFAULT_PRESET=4h_4bars
```

### プリセットの選択基準

- **短期取引（スキャルピング）**: `15m_4bars`, `30m_4bars`
- **デイトレード**: `1h_4bars`, `4h_4bars`
- **スイングトレード**: `1d_4bars`, `1d_7bars`
- **動的閾値（推奨）**: `4h_4bars_dynamic`, `1h_4bars_dynamic`

動的閾値（`*_dynamic`）は、データの分布に応じて自動的に閾値を調整するため、より汎用的です。

## 個別スクリプトの使用（非推奨）

個別のスクリプトも実行可能ですが、一貫性のために統合スクリプトの使用を推奨します。

### 1. 低重要度特徴検出

```bash
python scripts/feature_evaluation/detect_low_importance_features.py \
    --symbol BTC/USDT \
    --timeframe 1h \
    --lookback-days 90 \
    --threshold 0.2
```

### 2. 特徴量重要度分析

```bash
python -m scripts.feature_evaluation.analyze_feature_importance
```

### 3. 特徴量パフォーマンス評価

```bash
# 全モデルで評価
python -m scripts.feature_evaluation.evaluate_feature_performance --models all

# 特定モデルのみ
python -m scripts.feature_evaluation.evaluate_feature_performance --models lightgbm xgboost
```

## CommonFeatureEvaluator API

`CommonFeatureEvaluator`クラスは、以下の機能を提供します：

### データ取得

```python
from scripts.feature_evaluation.common_feature_evaluator import CommonFeatureEvaluator

evaluator = CommonFeatureEvaluator()
data = evaluator.fetch_data(
    symbol="BTC/USDT:USDT",
    timeframe="1h",
    limit=2000
)
```

### 特徴量生成

```python
features_df = evaluator.build_basic_features(
    data=data,
    skip_crypto_and_advanced=False
)
```

### ラベル生成（新機能）

```python
# 設定ファイルのプリセットを使用
labels = evaluator.create_labels_from_config(ohlcv_df)

# 特定のプリセットを指定
labels = evaluator.create_labels_from_config(
    ohlcv_df,
    preset_name="4h_4bars"
)

# ラベル生成設定情報を取得
label_config = evaluator.get_label_config_info()
```

### TimeSeriesSplit評価

```python
from sklearn.model_selection import TimeSeriesSplit

# 時系列CVで評価
cv_results = evaluator.time_series_cv(X, y, n_splits=5)
```

## トラブルシューティング

### データが取得できない

```bash
# OHLCVデータの確認
python -c "
from database.connection import SessionLocal
from database.repositories.ohlcv_repository import OHLCVRepository

db = SessionLocal()
repo = OHLCVRepository(db)
df = repo.get_ohlcv_dataframe('BTC/USDT:USDT', '1h', limit=100)
print(f'データ件数: {len(df)}')
db.close()
"
```

### メモリ不足エラー

- `--limit`パラメータを減らしてください（例: `--limit 1000`）
- より短い時間足を使用してください

### 分析時間が長すぎる

- データ件数を減らす（`--limit`）
- 特定のモデルのみで評価（`--models lightgbm`）

## 開発者向け情報

### テスト

```bash
cd backend
python -m pytest tests/ -k feature_evaluation
```

### コード品質チェック

```bash
cd backend
black scripts/feature_evaluation/
isort scripts/feature_evaluation/
flake8 scripts/feature_evaluation/
```

### 新しいプリセットの追加

`app/utils/label_generation/presets.py`の`get_common_presets()`関数に追加してください：

```python
def get_common_presets() -> Dict[str, Dict[str, Any]]:
    presets = {
        # ... 既存のプリセット
        "custom_preset": {
            "timeframe": "2h",
            "horizon_n": 6,
            "threshold": 0.0025,
            "threshold_method": ThresholdMethod.FIXED,
            "description": "カスタムプリセットの説明",
        },
    }
    return presets
```

## 参考リンク

- [ML Refactor Plan](../../ml_refactor_plan.md)
- [Unified Config](../../app/config/unified_config.py)
- [Label Generation Presets](../../app/utils/label_generation/presets.py)

## ライセンス

このプロジェクトのライセンスに従います。