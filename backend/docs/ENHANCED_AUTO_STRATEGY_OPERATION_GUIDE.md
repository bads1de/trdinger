# オートストラテジー強化システム運用ガイド

## 概要

このガイドでは、オートストラテジー強化システムの新機能の使用方法、最適なパラメータ設定、トラブルシューティング方法について説明します。

## 🚀 新機能一覧

### 1. フィットネス共有（Fitness Sharing）
- **目的**: 戦略の多様性向上
- **効果**: 類似戦略の評価を下げ、多様な戦略の共存を促進

### 2. ML予測確率指標
- **指標**: ML_UP_PROB, ML_DOWN_PROB, ML_RANGE_PROB
- **効果**: 機械学習による価格予測を戦略生成に活用

### 3. ショートバイアス突然変異
- **目的**: ショート戦略の生成強化
- **効果**: ロング・ショートバランスの改善

### 4. 高度な特徴量エンジニアリング
- **機能**: ファンディングレート、建玉残高を活用した特徴量
- **効果**: 市場の歪みや偏りを検出

## ⚙️ パラメータ設定指針

### GA設定（GAConfig）

#### 基本設定
```python
config = GAConfig()
config.population_size = 50      # 個体数（推奨: 30-100）
config.generations = 30          # 世代数（推奨: 20-50）
config.crossover_rate = 0.7      # 交叉率（推奨: 0.6-0.8）
config.mutation_rate = 0.1       # 突然変異率（推奨: 0.05-0.15）
```

#### フィットネス共有設定
```python
config.enable_fitness_sharing = True    # フィットネス共有を有効化
config.sharing_radius = 0.1             # 共有半径（推奨: 0.05-0.2）
config.sharing_alpha = 1.0              # 共有関数の形状（推奨: 0.5-2.0）
```

#### ショートバイアス設定
```python
config.enable_short_bias_mutation = True  # ショートバイアスを有効化
config.short_bias_rate = 0.3              # バイアス適用率（推奨: 0.2-0.5）
```

#### フィットネス重み設定
```python
config.fitness_weights = {
    "total_return": 0.25,      # 総リターン
    "sharpe_ratio": 0.35,      # シャープレシオ
    "max_drawdown": 0.2,       # 最大ドローダウン
    "win_rate": 0.1,           # 勝率
    "balance_score": 0.1       # ロング・ショートバランス
}
```

### 推奨設定パターン

#### 1. バランス重視設定
```python
config.enable_fitness_sharing = True
config.sharing_radius = 0.15
config.enable_short_bias_mutation = True
config.short_bias_rate = 0.4
config.fitness_weights["balance_score"] = 0.15
```

#### 2. 高性能重視設定
```python
config.enable_fitness_sharing = True
config.sharing_radius = 0.1
config.enable_short_bias_mutation = False
config.fitness_weights["total_return"] = 0.4
config.fitness_weights["sharpe_ratio"] = 0.4
```

#### 3. 多様性重視設定
```python
config.enable_fitness_sharing = True
config.sharing_radius = 0.2
config.sharing_alpha = 1.5
config.enable_short_bias_mutation = True
config.short_bias_rate = 0.3
```

## 🔧 運用手順

### 1. システム起動前の確認

#### 必要ライブラリの確認
```bash
cd backend
python run_library_tests.py
```

#### ML指標の動作確認
```bash
python run_ml_indicator_tests.py
```

### 2. 基本的な運用フロー

#### Step 1: データ準備
- OHLCV価格データ
- ファンディングレートデータ（オプション）
- 建玉残高データ（オプション）

#### Step 2: GA設定
```python
from app.core.services.auto_strategy.models.ga_config import GAConfig

config = GAConfig()
# パラメータ設定...
```

#### Step 3: 戦略生成実行
```python
from app.core.services.auto_strategy.auto_strategy_service import AutoStrategyService

service = AutoStrategyService()
result = service.generate_strategies(config, market_data)
```

#### Step 4: 結果評価
- 生成された戦略の多様性確認
- ロング・ショートバランス確認
- パフォーマンス指標確認

### 3. ML機能の活用

#### MLモデルの学習
```python
from app.core.services.ml import MLSignalGenerator

generator = MLSignalGenerator()
result = generator.train_model(training_data)
```

#### 学習済みモデルの読み込み
```python
generator.load_model("path/to/model.pkl")
```

## 🚨 トラブルシューティング

### よくある問題と解決方法

#### 1. メモリ不足エラー
**症状**: `MemoryError` が発生
**原因**: 大量のデータ処理
**解決方法**:
- データサイズを制限（最新50,000行など）
- バッチサイズを小さくする
- 不要なキャッシュをクリア

```python
# キャッシュクリア
feature_service.clear_cache()

# データサイズ制限
data = data.tail(10000)
```

#### 2. ML指標計算エラー
**症状**: ML指標が計算されない
**原因**: モデル未学習、データ不足
**解決方法**:
- モデルの学習状態確認
- 十分なデータ量の確保

```python
# モデル状態確認
status = ml_service.get_model_status()
print(status)

# 最小データ量確認（100行以上推奨）
if len(data) < 100:
    print("データ量が不足しています")
```

#### 3. フィットネス共有が効かない
**症状**: 戦略の多様性が向上しない
**原因**: 共有半径の設定不適切
**解決方法**:
- 共有半径を調整（0.05-0.2）
- 個体数を増加

```python
config.sharing_radius = 0.15  # 調整
config.population_size = 100  # 増加
```

#### 4. ショート戦略が生成されない
**症状**: ロング戦略ばかり生成される
**原因**: ショートバイアス設定不適切
**解決方法**:
- ショートバイアス率を上げる
- ショート特化指標を追加

```python
config.short_bias_rate = 0.5  # 増加
config.allowed_indicators.extend(['ML_DOWN_PROB'])
```

### エラーログの確認

#### ログレベル設定
```python
import logging
logging.basicConfig(level=logging.INFO)
```

#### 主要なログメッセージ
- `✅ 正常処理`: 成功メッセージ
- `⚠️ 警告`: 注意が必要な状況
- `❌ エラー`: 処理失敗

### パフォーマンス最適化

#### 1. キャッシュ活用
```python
# キャッシュ情報確認
cache_info = feature_service.get_cache_info()
print(cache_info)
```

#### 2. データ型最適化
- 自動的にfloat64→float32に変換
- メモリ使用量約50%削減

#### 3. 並列処理設定
```python
config.parallel_processes = 4  # CPUコア数に応じて調整
```

## 📊 監視とメンテナンス

### 定期的な確認項目

#### 1. システム状態
- メモリ使用量
- キャッシュサイズ
- エラーログ

#### 2. 戦略品質
- 多様性指標
- バランススコア
- パフォーマンス指標

#### 3. MLモデル
- 予測精度
- 特徴量重要度
- モデル更新頻度

### メンテナンス手順

#### 週次メンテナンス
1. キャッシュクリア
2. ログファイル整理
3. パフォーマンス確認

#### 月次メンテナンス
1. MLモデル再学習
2. パラメータ最適化
3. システム更新

## 📞 サポート

### 技術サポート
- ログファイルの提供
- 設定ファイルの確認
- エラーメッセージの詳細

### 改善要望
- 新機能の提案
- パフォーマンス改善
- UI/UX改善

---

このガイドを参考に、オートストラテジー強化システムを効果的に運用してください。
