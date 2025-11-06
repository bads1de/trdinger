# 上レンジ/下レンジ予測改善実装計画（CPU 環境最適化版）

## 📋 概要

**目的**: CPU 環境（AMD Radeon 非対応）における個人レベルの ML 予測精度向上

**環境制約**:

- GPU: AMD Radeon（PyTorch/CUDA 非対応）
- 実行環境: 主に CPU
- メモリ目標: 1-2GB 以内
- 既存実装: XGBoost、LightGBM、TabNet

**削除要素**: LSTM/GRU、Transformer、DRL、CatBoost（将来検討）

**維持要素**: LightGBM（最優先）、XGBoost、TabNet（特徴抽出）

---

## 🎯 Phase 1: CPU 最適化アンサンブル（3-4 週間）

### 概要

**目的**: CPU 環境で最高性能を引き出す軽量アンサンブル構築

**主要コンポーネント**:

1. **TabNetFeatureExtractor** - TabNet を特徴抽出専用に最適化
2. **LightweightStackingEnsemble** - CPU 最適化スタッキング
3. **RandomForestMetaLearner** - 過学習抑制メタモデル（2025 年最新研究）
4. **Intel oneDAL 統合** - daal4py 変換で推論 2-3 倍高速化
5. **Intel Extension for scikit-learn** - sklearnex による透過的な高速化
6. **CPU 最適化設定** - 各モデルの並列処理最適化

**期待効果**: 予測精度 **4-7%向上**、推論速度 **2-5 倍向上**（sklearnex 含む）

### 実装箇所

```
backend/app/services/ml/
├── models/
│   └── tabnet_feature_extractor.py      # 新規
├── ensemble/
│   ├── lightweight_stacking.py          # 新規
│   └── random_forest_meta.py            # 新規★
├── optimization/
│   ├── daal4py_converter.py             # 新規★
│   └── sklearnex_optimizer.py           # 新規★★
└── config/ml_config.py                  # CPU最適化設定追加

backend/tests/ml/
├── test_tabnet_feature_extractor.py
├── test_lightweight_stacking.py
├── test_random_forest_meta.py           # 新規★
├── test_daal4py_converter.py            # 新規★
└── test_sklearnex_optimizer.py          # 新規★★
```

### 主要クラス

#### TabNetFeatureExtractor

```python
class TabNetFeatureExtractor:
    """TabNet特徴抽出器（CPU最適化、Attention機構）"""
    def __init__(self, n_steps=3, n_d=8, n_a=8, device_name='cpu'): ...
    def fit(self, X_train, y_train, X_val=None, y_val=None) -> Dict: ...
    def extract_features(self, X, top_n=50) -> Tuple[np.ndarray, Dict]: ...
    def predict_proba(self, X) -> np.ndarray: ...
```

#### LightweightStackingEnsemble

```python
class LightweightStackingEnsemble(BaseEnsemble):
    """CPU最適化スタッキング（LightGBM + XGBoost + TabNet）"""
    def __init__(self, config, automl_config=None): ...
    def fit(self, X_train, y_train, X_test=None, y_test=None) -> Dict: ...
    def _train_base_models(self, X_train, y_train) -> Tuple: ...
    def _train_meta_learner(self, meta_features, y_train) -> Dict: ...
    def predict(self, X) -> np.ndarray: ...
    def predict_proba(self, X) -> np.ndarray: ...
```

#### RandomForestMetaLearner

```python
class RandomForestMetaLearner:
    """Random Forestメタモデル（過学習抑制、2025年最新研究）
    
    Note: Random Forestはバギング（Bootstrap Aggregating）を内蔵しているため、
    独立したバギング実装は不要です。
    """
    def __init__(self, n_estimators=100, max_depth=10, n_jobs=-1): ...
    def fit(self, X, y) -> Dict: ...
    def predict(self, X) -> np.ndarray: ...
    def predict_proba(self, X) -> np.ndarray: ...
```

#### Daal4pyConverter

```python
class Daal4pyConverter:
    """Intel oneDAL変換（推論2-3倍高速化）"""
    def __init__(self): ...
    def convert_xgboost(self, xgb_model, model_id) -> Optional[Any]: ...
    def convert_lightgbm(self, lgb_model, model_id) -> Optional[Any]: ...
    def predict(self, model_id, X) -> Optional[np.ndarray]: ...
    def get_speedup_info(self) -> Dict: ...
```

#### SklearnexOptimizer

```python
class SklearnexOptimizer:
    """Intel Extension for scikit-learn最適化（透過的パッチ適用）"""
    def __init__(self): ...
    def patch_sklearn(self) -> Dict: ...
    def unpatch_sklearn(self) -> None: ...
    def get_optimization_status(self) -> Dict: ...
```

### CPU 最適化パラメータ

**LightGBM**:

```python
{'device': 'cpu', 'n_jobs': -1, 'num_threads': -1, 'boosting_type': 'gbdt'}
```

**XGBoost**:

```python
{'n_jobs': -1, 'tree_method': 'hist', 'max_bin': 256}
```

**TabNet**:

```python
{'device_name': 'cpu', 'n_steps': 3, 'batch_size': 256, 'max_epochs': 30}
```

**NumPy/MKL 環境変数**:

```python
import os
os.environ['MKL_NUM_THREADS'] = str(os.cpu_count())
os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())
```

**Intel Extension for scikit-learn**:

```python
from sklearnex import patch_sklearn, unpatch_sklearn
patch_sklearn()  # 透過的な高速化
# 通常のscikit-learnと同じAPIを使用
```

### テスト

**主要テストケース**:

- `test_tabnet_feature_extractor.py`: init、fit、extract_features、CPU 最適化
- `test_lightweight_stacking.py`: init、fit、predict、CPU 設定
- `test_random_forest_meta.py`: init、fit、predict、特徴重要度
- `test_daal4py_converter.py`: 変換、予測、高速化確認
- `test_sklearnex_optimizer.py`: パッチ適用、最適化状態、高速化確認

### 完了条件

- [ ] 全クラス実装完了（4 クラス）
- [ ] CPU 最適化パラメータ設定
- [ ] Intel oneDAL 統合とフォールバック
- [ ] ユニットテスト全パス（80%カバレッジ）
- [ ] 予測精度 4%以上向上
- [ ] 推論速度 2 倍以上（oneDAL 利用時）
- [ ] メモリ使用量 2GB 以内
- [ ] CPU 使用率 80%以上

---

## 🎯 Phase 2: 市場マイクロストラクチャー指標（2-3 週間）

### 概要

**目的**: 市場の微細構造を捉える高度な指標追加

**期待効果**: 予測精度 **2-4%向上**

### 実装する指標

#### 2.1 出来高プロファイル指標

**ファイル**: `backend/app/services/ml/feature_engineering/volume_profile_features.py`

```python
def calculate_volume_profile(df, price_col='close', volume_col='volume', n_bins=50) -> pd.DataFrame:
    """
    出来高プロファイル計算
    - POC（Point of Control）: 最大出来高価格
    - VAH/VAL: 出来高70%エリア
    - 価格-POC距離、VAエリア判定
    """
```

#### 2.2 オーダーブック不均衡指標

**ファイル**: `backend/app/services/ml/feature_engineering/orderbook_features.py`

```python
def calculate_orderbook_imbalance(df, bid_col='bid_volume', ask_col='ask_volume') -> pd.DataFrame:
    """
    オーダーブック不均衡計算
    - 基本不均衡、ローリング平均/標準偏差、累積不均衡
    """
```

#### 2.3 資金調達率指標

**ファイル**: `backend/app/services/ml/feature_engineering/funding_rate_features.py`

```python
def calculate_funding_rate_features(df, funding_rate_col='funding_rate') -> pd.DataFrame:
    """
    資金調達率特徴量計算
    - 基本統計量（8h/24h/72h）、累積、変化率、極端値検出
    """
```

#### 2.4 マーケットマイクロストラクチャー拡張指標

**ファイル**: `backend/app/services/ml/feature_engineering/microstructure_features.py`

```python
def calculate_microstructure_features(df, orderbook_df=None) -> pd.DataFrame:
    """
    マーケットマイクロストラクチャー指標計算
    - オーダーブック不均衡: (bid_volume - ask_volume) / (bid_volume + ask_volume)
    - 出来高加重ミッド価格: (bid_price * ask_volume + ask_price * bid_volume) / total_volume
    - トレード完了確率指標（pT）
    """
```

### 完了条件

- [ ] 出来高プロファイル実装
- [ ] オーダーブック不均衡実装
- [ ] 資金調達率実装
- [ ] マーケットマイクロストラクチャー拡張実装
- [ ] 各指標のユニットテスト
- [ ] 予測精度 2%以上向上
- [ ] 特徴量重要度分析で有効性確認

---

## 🎯 Phase 2.75: 高度な特徴選択（mRMR 統合）（1-2 週間）★NEW

### 概要

**目的**: Boruta に mRMR を統合し、特徴選択精度を向上

**期待効果**: 予測精度 **1-3%追加向上**、特徴削減率 **35-40%**

### 主要コンポーネント

#### MRMRSelector

**ファイル**: `backend/app/services/ml/feature_engineering/mrmr_selector.py`

```python
class MRMRSelector:
    """mRMR特徴選択（最大関連性・最小冗長性）"""
    def __init__(self, n_features=10): ...
    def fit(self, X, y) -> Dict: ...
    def transform(self, X) -> pd.DataFrame: ...
    def get_feature_scores(self) -> Dict: ...
```

**機能**: 相互情報量ベースの特徴選択、冗長性削減

#### BoMGeneIntegrator

**ファイル**: `backend/app/services/ml/feature_engineering/bomgene_integrator.py`

```python
class BoMGeneIntegrator:
    """BorutaとmRMRの統合（BoMGene手法）"""
    def __init__(self, max_features=20): ...
    def fit_transform(self, X, y) -> Tuple[pd.DataFrame, Dict]: ...
    def _mrmr_initial_selection(self, X, y) -> List: ...
    def _boruta_verification(self, X, y, initial_features) -> List: ...
    def _iterative_refinement(self, X, y, features) -> List: ...
```

**3 ステップアプローチ**:

1. mRMR で初期候補選択（10-20 特徴）
2. Boruta で検証と追加発見
3. 相互情報量による反復的洗練

### 完了条件

- [ ] MRMRSelector 実装
- [ ] BoMGeneIntegrator 実装
- [ ] 特徴削減率 35-40%
- [ ] 予測精度維持または向上
- [ ] ユニットテスト全パス
- [ ] Boruta 単独比較で優位性確認

---

## 🎯 Phase 2.5: 高度な特徴選択と相互作用（2-3 週間）★NEW

### 概要

**目的**: 特徴選択の高度化と相互作用特徴量自動生成

**期待効果**: 予測精度 **3-5%向上**、過学習リスク **30%削減**

### 主要コンポーネント

#### BorutaSelector

**ファイル**: `backend/app/services/ml/feature_engineering/boruta_selector.py`

```python
class BorutaSelector:
    """Boruta特徴選択（RandomForestベース、シャドウ特徴比較）"""
    def __init__(self, n_estimators=100, max_iter=100, alpha=0.05): ...
    def fit(self, X, y) -> Dict: ...
    def transform(self, X) -> pd.DataFrame: ...
    def _create_shadow_features(self, X) -> np.ndarray: ...
```

**機能**: シャドウ特徴との統計的比較、反復的選択、過学習抑制

#### InteractionGenerator

**ファイル**: `backend/app/services/ml/feature_engineering/interaction_generator.py`

```python
class InteractionGenerator:
    """相互作用特徴量自動生成（事前定義ルールベース）"""
    def __init__(self): ...
    def generate(self, df) -> pd.DataFrame: ...
    def get_generated_info(self) -> Dict: ...
```

**8 種類の相互作用ルール**:

1. FR × OI 相関
2. Volume × Volatility
3. Price/POC 比率
4. Multi-timeframe momentum
5. BB × RSI 組み合わせ
6. VP × Price Momentum
7. Microstructure × Price Momentum
8. Multi-timeframe Volume Profile

### 完了条件

- [ ] BorutaSelector 実装
- [ ] InteractionGenerator 実装
- [ ] 特徴削減率 30%以上
- [ ] 相互作用特徴 8 種類生成
- [ ] 予測精度維持または向上
- [ ] ユニットテスト全パス
- [ ] 過学習リスク定量的削減確認

---

## 🎯 Phase 3: モデル軽量化・推論最適化（2-3 週間）★NEW

### 概要

**目的**: モデル圧縮と推論速度の最適化

**期待効果**: メモリ使用量 **30-40%削減**、推論速度 **1.5-2 倍向上**

### 主要コンポーネント

#### ModelQuantizer

**ファイル**: `backend/app/services/ml/optimization/model_quantizer.py`

```python
class ModelQuantizer:
    """モデル量子化（Post-Training Quantization）"""
    def __init__(self, bits=8): ...
    def quantize_model(self, model) -> Tuple[Any, Dict]: ...
    def quantize_weights(self, weights) -> Tuple: ...
    def dequantize_weights(self, quantized, scale, min_val) -> np.ndarray: ...
```

#### ModelPruner

**ファイル**: `backend/app/services/ml/optimization/model_pruner.py`

```python
class ModelPruner:
    """構造化プルーニング（ツリーモデル最適化）"""
    def __init__(self, prune_ratio=0.2): ...
    def prune_tree_model(self, model) -> Tuple[Any, Dict]: ...
    def analyze_feature_importance(self, model) -> Dict: ...
    def get_pruning_stats(self) -> Dict: ...
```

### 実装する最適化

1. **Post-Training Quantization (PTQ)**: 32bit → 8bit 変換
2. **構造化プルーニング**: 重要度の低い枝の削減（20-30%）
3. **ハイパーパラメータ調整**: メモリ効率重視の設定
4. **推論パイプライン最適化**: バッチ処理とキャッシング

### 完了条件

- [ ] ModelQuantizer 実装
- [ ] ModelPruner 実装
- [ ] メモリ使用量 30%以上削減
- [ ] 推論速度 1.5 倍以上向上
- [ ] 精度劣化 1%以内
- [ ] ユニットテスト全パス

---

## 📊 依存関係

### 削除

- ❌ torch、torch-geometric、stable-baselines3、catboost

### 維持

- ✅ lightgbm、xgboost、pytorch-tabnet（CPU モード）、scikit-learn、pandas、numpy

### 追加（オプション）

- daal4py（Intel oneDAL）

---

## 📅 実装スケジュール（8-11 週間）

| Week | Phase      | 内容                                                 |
| ---- | ---------- | ---------------------------------------------------- |
| 1    | Phase 1    | TabNetFeatureExtractor 実装・テスト                  |
| 2    | Phase 1    | LightweightStackingEnsemble 実装・統合テスト         |
| 3    | Phase 1    | RandomForestMetaLearner + SklearnexOptimizer 実装    |
| 4    | Phase 1    | Daal4pyConverter 実装・速度ベンチマーク              |
| 5    | Phase 2    | 出来高プロファイル・マイクロストラクチャー実装        |
| 6    | Phase 2    | オーダーブック・資金調達率実装                        |
| 7    | Phase 2.75 | MRMRSelector + BoMGeneIntegrator 実装                |
| 8    | Phase 2.5  | InteractionGenerator 実装・相互作用評価              |
| 9    | Phase 3    | ModelQuantizer + ModelPruner 実装                    |
| 10   | Phase 3    | 推論パイプライン最適化・ベンチマーク                  |
| 11   | 最終       | 統合テスト・パフォーマンステスト・ドキュメント        |

---

## 📈 KPI と成功基準

### 予測精度向上

| 指標                  | 現状         | 目標    | Phase 1 | Phase 2 | Phase 2.75 | Phase 2.5 | Phase 3 |
| --------------------- | ------------ | ------- | ------- | ------- | ---------- | --------- | ------- |
| **Accuracy**          | ベースライン | +12-20% | +4-7%   | +2-4%   | +1-3%      | +3-5%     | ±0-1%   |
| **F1 Score**          | ベースライン | +10-17% | +3-6%   | +2-3%   | +1-2%      | +2-4%     | ±0-1%   |
| **Balanced Accuracy** | ベースライン | +9-16%  | +3-5%   | +1-3%   | +1-2%      | +2-4%     | ±0-1%   |
| **RMSE 削減**         | ベースライン | -13-22% | -8-12%  | -2-6%   | -1-2%      | -2-2%     | ±0%     |

### パフォーマンス指標

| 指標                           | 目標                      | 備考                 |
| ------------------------------ | ------------------------- | -------------------- |
| **学習時間**                   | 10000 サンプルで 5 分以内 | 維持                 |
| **推論速度**                   | 2000-3000 サンプル/秒     | oneDAL 利用時        |
| **推論速度（sklearnex）**      | 2500-4000 サンプル/秒     | sklearnex 利用時     |
| **推論速度（通常）**           | 1000 サンプル/秒          | 最適化未使用時       |
| **メモリ使用量（最適化前）**   | 1-2GB 以内                | Phase 1-2 完了時     |
| **メモリ使用量（最適化後）**   | 0.6-1.4GB                 | Phase 3 完了後       |
| **メモリ削減率**               | 30-40%                    | Phase 3 完了後       |
| **CPU 使用率**                 | 80%以上                   | 並列処理時           |
| **特徴削減率（Boruta）**       | 30%以上                   | Phase 2.5 適用後     |
| **特徴削減率（BoMGene）**      | 35-40%                    | Phase 2.75 適用後    |

### 実装品質

- **テストカバレッジ**: 80%以上
- **コードスタイル**: Black、Flake8、MyPy 準拠
- **ドキュメント**: 全クラス・関数に docstring
- **新規ファイル数**: 14 ファイル（実装 7 + テスト 7）

---

## ⚠️ リスクと対策

| リスク             | 対策                               |
| ------------------ | ---------------------------------- |
| TabNet の CPU 性能 | n_steps=3 削減、最悪 TabNet 無効化 |
| メモリ不足         | バッチ処理、段階的計算、早期削除   |
| 予測精度未達       | ハイパーパラメータ追加チューニング |
| 実装期間遅延       | Phase 2 一部後回し、段階的リリース |

---

## 🔮 将来の検討事項

以下は個人環境・CPU 制約で除外したが、将来検討価値あり：

### 1. Temporal Fusion Transformer（TFT）

- **特徴**: Attention + 自己相関、長期依存性モデリング
- **課題**: GPU 必須、計算コスト高、メモリ消費大
- **検討条件**: GPU 環境利用可能時
- **期待効果**: 5-10%追加向上（GPU 環境）

### 2. CatBoost 統合（4 モデルスタッキング）

- **特徴**: カテゴリ変数自動処理、Ordered Boosting
- **課題**: 学習時間長、メモリやや多い
- **検討条件**: Phase 1-2.5 完了後、さらなる精度向上必要時
- **期待効果**: 2-4%追加向上

### 3. 深層学習（LSTM/GRU/Transformer）

- **課題**: PyTorch AMD 非対応、CPU 極めて遅い
- **検討条件**: NVIDIA GPU 利用可能時
- **期待効果**: 5-15%追加向上（GPU 環境）

### 4. 強化学習（DRL）

- **課題**: 学習時間非常に長い、報酬設計難、安定性問題
- **検討条件**: GPU 環境 + 十分な計算リソース
- **期待効果**: 動的最適化（精度向上不明）

---

## 📚 参考文献

### 機械学習・アンサンブル学習

1. Şimşek (2025) - "Stacked Generalization Model for BIST100 Index" - Research Square
2. Biswas et al. (2025) - "Stock Price Prediction Using Stacked Heterogeneous Ensemble" - MDPI Mathematics
3. Mazinani et al. (2025) - "Transformer-based Cryptocurrency Prediction" - Journal of Big Data
4. Urooj et al. (2024) - "Ensemble ML vs Deep Learning Methods" - 比較分析

### 特徴量エンジニアリング

5. "Deep Learning for Stock Market Prediction" - ResearchGate（Boruta アルゴリズム）
6. "Forward Feature Selection: Empirical Analysis" - ResearchGate

### Intel 最適化・モデル圧縮

7. Intel Developer (2024) - "Faster XGBoost/LightGBM on CPU"
8. Intel Distribution for Python Release Notes - daal4py 変換
9. Intel oneAPI Base Toolkit (2025) - https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html
10. Benchmarking Classical ML on Google Cloud - https://cloud.google.com/blog/products/data-analytics/benchmarking-classical-ml
11. Model Compression for LLMs - https://arxiv.org/html/2504.11651v2

### 特徴選択・マイクロストラクチャー

12. BoMGene: Boruta-mRMR Integration (2024) - https://arxiv.org/html/2510.00907v1
13. Limit Order Book Microstructure (2025) - https://www.emergentmind.com/topics/limit-order-book-microstructure
14. TimeGPT Cryptocurrency Forecasting (2024) - https://www.mdpi.com/2571-9394/7/3/48

### 技術ドキュメント

- [LightGBM Parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html)
- [XGBoost Tree Methods](https://xgboost.readthedocs.io/en/stable/treemethod.html)
- [TabNet Paper](https://arxiv.org/abs/1908.07442)
- [Boruta Algorithm](https://www.jstatsoft.org/article/view/v036i11)

---

## 🎯 まとめ

### 主要な変更点

1. GPU 依存完全削除（将来検討事項として記録）
2. LightGBM 中心構成（最速・最軽量）
3. Random Forest メタモデル（2025 年最新研究で実証）
4. Intel oneDAL 統合（CPU 推論 2-3 倍高速化）
5. **Intel Extension 統合**: sklearnex による透過的高速化
6. **mRMR 統合**: BoMGene 手法で特徴選択精度向上
7. **モデル軽量化**: 量子化・プルーニングで推論最適化
8. **マイクロストラクチャー拡張**: オーダーブック詳細分析
9. **相互作用特徴量**: ドメイン知識ベース 8 種類
10. **現実的期間**: 8-11 週間

### 期待される効果

- **予測精度向上**: 合計 **12-20%**（保守的見積もり）
- **推論速度向上**: **2-5 倍**（sklearnex + oneDAL 利用時）
- **メモリ削減**: **30-40%**（量子化・プルーニング後）
- **過学習リスク削減**: **35-40%**（BoMGene 適用後）
- **実装期間**: 8-11 週間
- **メモリ使用量（最適化後）**: 0.6-1.4GB
- **CPU 使用率**: 80%以上
- **個人環境で実用可能**: GPU 不要

### 次のステップ

1. Phase 1 開始: `TabNetFeatureExtractor`実装
2. 段階的評価: 各週ごと進捗確認
3. 柔軟な調整: 精度・性能に応じて微調整
4. 将来の拡張: GPU 環境時に TFT/CatBoost 検討
