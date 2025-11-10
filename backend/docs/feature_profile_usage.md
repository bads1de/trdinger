# 特徴量プロファイル使用ガイド

## 概要

特徴量プロファイル機能により、ML学習時に使用する特徴量セットを簡単に切り替えることができます。

- **research**: すべての特徴量を使用（研究・実験用）
- **production**: 厳選された高重要度特徴量のみ使用（本番環境用）

## 設定方法

### 1. 環境変数で設定

```bash
# productionプロファイルを使用
export ML__FEATURE_ENGINEERING__PROFILE=production

# researchプロファイルを使用（デフォルト）
export ML__FEATURE_ENGINEERING__PROFILE=research
```

### 2. プログラムから設定

#### BaseMLTrainerを使用する場合

```python
from app.services.ml.base_ml_trainer import BaseMLTrainer
from app.config.unified_config import unified_config

# 設定から自動読み込み
unified_config.ml.feature_engineering.profile = "production"

trainer = BaseMLTrainer(
    trainer_config={"type": "single", "model_type": "lightgbm"}
)

# 学習実行（設定されたprofileが自動使用される）
result = trainer.train_model(
    training_data=ohlcv_data,
    save_model=True,
    model_name="my_model"
)

print(f"使用特徴量数: {result['feature_count']}")
```

#### MLTrainingServiceを使用する場合

```python
from app.services.ml.ml_training_service import MLTrainingService

service = MLTrainingService(
    trainer_type="single",
    single_model_config={"model_type": "lightgbm"}
)

# パラメータで明示的に指定（設定より優先）
result = service.train_model(
    training_data=ohlcv_data,
    feature_profile="production",  # productionプロファイルを指定
    save_model=True,
    model_name="production_model"
)

print(f"使用特徴量数: {result['feature_count']}")
```

### 3. API経由で設定

```bash
# API リクエスト例
curl -X POST "http://localhost:8000/api/ml-training/train" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTC/USDT:USDT",
    "timeframe": "4h",
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    "feature_profile": "production",
    "save_model": true
  }'
```

## プロファイルの内容

### Research プロファイル

- すべての特徴量を使用（100+ 特徴量）
- 研究・実験・特徴量重要度分析に最適
- 計算時間が長い
- メモリ使用量が多い

### Production プロファイル

- 厳選された高重要度特徴量のみ（約50-70特徴量）
- 本番環境・リアルタイム推論に最適
- 計算時間が短い
- メモリ使用量が少ない
- 予測精度は研究用と同等レベルを維持

含まれる特徴量：
- 基本テクニカル指標（RSI, MACD, MA, BB, ATR）
- ボリューム関連（Volume_MA_Ratio, Volume_Trend）
- ボラティリティ関連（Volatility_20, Volatility_Ratio）
- モメンタム指標（Momentum_14, ROC_10）
- 価格関連（Price_Change_Pct, High_Low_Range）
- 市場レジーム（Market_Regime, Trend_Strength）
- 建玉残高関連（OI_Change_Rate_24h, Volatility_Adjusted_OI）
- 複合指標（FR_OI_Ratio, Market_Heat_Index）
- 暗号通貨特化特徴量（Price_Volume_Correlation, Funding_Rate_Impact）

## 使用例

### 例1: 研究用モデルの学習

```python
from app.services.ml.ml_training_service import MLTrainingService

service = MLTrainingService(trainer_type="single")

# すべての特徴量を使用して学習
result = service.train_model(
    training_data=data,
    feature_profile="research",
    use_cross_validation=True,
    cv_splits=5,
    save_model=True,
    model_name="research_model_v1"
)

print(f"学習完了")
print(f"特徴量数: {result['feature_count']}")
print(f"精度: {result.get('accuracy', 'N/A')}")
```

### 例2: 本番用モデルの学習

```python
from app.services.ml.ml_training_service import MLTrainingService

service = MLTrainingService(trainer_type="single")

# 厳選された特徴量のみを使用して学習
result = service.train_model(
    training_data=data,
    feature_profile="production",
    use_cross_validation=True,
    cv_splits=5,
    save_model=True,
    model_name="production_model_v1"
)

print(f"学習完了")
print(f"特徴量数: {result['feature_count']}")
print(f"精度: {result.get('accuracy', 'N/A')}")
```

### 例3: プロファイル比較

```python
from app.services.ml.ml_training_service import MLTrainingService

service = MLTrainingService(trainer_type="single")

# Researchプロファイルで学習
result_research = service.train_model(
    training_data=data,
    feature_profile="research",
    save_model=False
)

# Productionプロファイルで学習
result_production = service.train_model(
    training_data=data,
    feature_profile="production",
    save_model=False
)

print("=== プロファイル比較 ===")
print(f"Research  - 特徴量数: {result_research['feature_count']}, 精度: {result_research.get('accuracy', 'N/A')}")
print(f"Production - 特徴量数: {result_production['feature_count']}, 精度: {result_production.get('accuracy', 'N/A')}")
```

## カスタムallowlist

特定の特徴量セットを使用したい場合は、カスタムallowlistを指定できます：

```python
from app.config.unified_config import unified_config

# カスタムallowlistを設定
unified_config.ml.feature_engineering.custom_allowlist = [
    "RSI_14",
    "MACD",
    "MA_Short_7",
    "MA_Long_25",
    "Volume_MA_Ratio",
    "ATR_14",
]

# これ以降の学習ではカスタムallowlistが使用される
```

## ロギング

プロファイル使用時には以下のログが出力されます：

```
INFO: 📊 特徴量計算を実行中（profile: production）...
INFO: ✅ 特徴量生成完了: 68個の特徴量
INFO: 特徴量プロファイル 'production' を適用中...
INFO: プロファイル 'production' 適用完了: 150個 → 68個の特徴量 (82個をドロップ)
```

## ベストプラクティス

1. **研究フェーズ**: `research`プロファイルで特徴量重要度分析を実施
2. **最適化フェーズ**: 重要度分析結果をもとに`production`プロファイルを調整
3. **本番デプロイ**: `production`プロファイルでモデルを学習し、本番環境にデプロイ
4. **定期見直し**: 定期的に`research`プロファイルで再分析し、`production`プロファイルを更新

## トラブルシューティング

### プロファイルが反映されない

```python
# キャッシュをクリア
from app.services.ml.feature_engineering.feature_engineering_service import FeatureEngineeringService

service = FeatureEngineeringService()
service.clear_cache()
```

### 特徴量数が期待と異なる

ログを確認して、どのプロファイルが実際に使用されているか確認してください：

```python
import logging
logging.basicConfig(level=logging.INFO)
```

## 関連ファイル

- [`FeatureEngineeringService`](../app/services/ml/feature_engineering/feature_engineering_service.py) - プロファイル実装
- [`FeatureEngineeringConfig`](../app/config/unified_config.py) - プロファイル設定
- [`BaseMLTrainer`](../app/services/ml/base_ml_trainer.py) - トレーナー統合
- [`MLTrainingService`](../app/services/ml/ml_training_service.py) - サービス統合

## 参考資料

- [特徴量重要度分析ガイド](./feature_importance_analysis.md)
- [ML学習ガイド](./ml_training_guide.md)