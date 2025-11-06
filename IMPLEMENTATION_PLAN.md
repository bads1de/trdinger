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
5. **CPU 最適化設定** - 各モデルの並列処理最適化

**期待効果**: 予測精度 **4-7%向上**、推論速度 **2-3 倍向上**

### 実装箇所

```
backend/app/services/ml/
├── models/
│   └── tabnet_feature_extractor.py      # 新規
├── ensemble/
│   ├── lightweight_stacking.py          # 新規
│   └── random_forest_meta.py            # 新規★
├── optimization/
│   └── daal4py_converter.py             # 新規★
└── config/ml_config.py                  # CPU最適化設定追加

backend/tests/ml/
├── test_tabnet_feature_extractor.py
├── test_lightweight_stacking.py
├── test_random_forest_meta.py           # 新規★
└── test_daal4py_converter.py            # 新規★
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
    """Random Forestメタモデル（過学習抑制、2025年最新研究）"""
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

### テスト

**主要テストケース**:

- `test_tabnet_feature_extractor.py`: init、fit、extract_features、CPU 最適化
- `test_lightweight_stacking.py`: init、fit、predict、CPU 設定
- `test_random_forest_meta.py`: init、fit、predict、特徴重要度
- `test_daal4py_converter.py`: 変換、予測、高速化確認

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

## 🎯 Phase 2: 市場マイクロストラクチャー指標（3-4 週間）

### 概要

**目的**: 市場の微細構造を捉える高度な指標追加

**期待効果**: 予測精度 **3-6%向上**

### 実装する指標

#### 2.1 オンチェーンメトリクス（2024 年 VLDB 論文ベース）★NEW

**ファイル**: `backend/app/services/ml/feature_engineering/onchain_features.py`

```python
def calculate_onchain_features(df, network_data=None, exchange_flow_data=None) -> pd.DataFrame:
    """
    オンチェーンメトリクス計算
    - ネットワーク活動: アクティブアドレス、TX数/ボリューム、ハッシュレート
    - 取引所フロー: 入出金量、ネットフロー、クジラ追跡
    - NVT比率: ネットワーク価値評価
    """
```

**データソース**: Glassnode API、CryptoQuant API、Blockchain.com API

#### 2.2 出来高プロファイル指標

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

#### 2.3 オーダーブック不均衡指標

**ファイル**: `backend/app/services/ml/feature_engineering/orderbook_features.py`

```python
def calculate_orderbook_imbalance(df, bid_col='bid_volume', ask_col='ask_volume') -> pd.DataFrame:
    """
    オーダーブック不均衡計算
    - 基本不均衡、ローリング平均/標準偏差、累積不均衡
    """
```

#### 2.4 資金調達率指標

**ファイル**: `backend/app/services/ml/feature_engineering/funding_rate_features.py`

```python
def calculate_funding_rate_features(df, funding_rate_col='funding_rate') -> pd.DataFrame:
    """
    資金調達率特徴量計算
    - 基本統計量（8h/24h/72h）、累積、変化率、極端値検出
    """
```

### 完了条件

- [ ] オンチェーンメトリクス実装
- [ ] 出来高プロファイル実装
- [ ] オーダーブック不均衡実装
- [ ] 資金調達率実装
- [ ] 各指標のユニットテスト
- [ ] 予測精度 3%以上向上
- [ ] 特徴量重要度分析で有効性確認

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
7. オンチェーン/価格比率
8. 取引所フロー × Volatility

### 完了条件

- [ ] BorutaSelector 実装
- [ ] InteractionGenerator 実装
- [ ] 特徴削減率 30%以上
- [ ] 相互作用特徴 8 種類生成
- [ ] 予測精度維持または向上
- [ ] ユニットテスト全パス
- [ ] 過学習リスク定量的削減確認

---

## 📊 依存関係

### 削除

- ❌ torch、torch-geometric、stable-baselines3、catboost

### 維持

- ✅ lightgbm、xgboost、pytorch-tabnet（CPU モード）、scikit-learn、pandas、numpy

### 追加（オプション）

- daal4py（Intel oneDAL）

---

## 📅 実装スケジュール（7-9 週間）

| Week | Phase     | 内容                                               |
| ---- | --------- | -------------------------------------------------- |
| 1    | Phase 1   | TabNetFeatureExtractor 実装・テスト                |
| 2    | Phase 1   | LightweightStackingEnsemble 実装・統合テスト       |
| 3    | Phase 1   | RandomForestMetaLearner 実装・比較評価             |
| 4    | Phase 1   | Daal4pyConverter 実装・速度ベンチマーク            |
| 5    | Phase 2   | オンチェーンメトリクス実装・API 統合               |
| 6    | Phase 2   | 出来高プロファイル・オーダーブック・資金調達率実装 |
| 7    | Phase 2.5 | BorutaSelector 実装・特徴削減評価                  |
| 8    | Phase 2.5 | InteractionGenerator 実装・相互作用評価            |
| 9    | 最終      | 統合テスト・パフォーマンステスト・ドキュメント     |

---

## 📈 KPI と成功基準

### 予測精度向上

| 指標                  | 現状         | 目標    | Phase 1 | Phase 2 | Phase 2.5 |
| --------------------- | ------------ | ------- | ------- | ------- | --------- |
| **Accuracy**          | ベースライン | +10-18% | +4-7%   | +3-6%   | +3-5%     |
| **F1 Score**          | ベースライン | +8-15%  | +3-6%   | +3-5%   | +2-4%     |
| **Balanced Accuracy** | ベースライン | +7-14%  | +3-5%   | +2-5%   | +2-4%     |
| **RMSE 削減**         | ベースライン | -12-20% | -8-12%  | -4-8%   | N/A       |

### パフォーマンス指標

| 指標                 | 目標                      | 備考            |
| -------------------- | ------------------------- | --------------- |
| **学習時間**         | 10000 サンプルで 5 分以内 | 維持            |
| **推論速度**         | 2000-3000 サンプル/秒     | oneDAL 利用時   |
| **推論速度（通常）** | 1000 サンプル/秒          | oneDAL 未使用時 |
| **メモリ使用量**     | 1-2GB 以内                | 維持            |
| **CPU 使用率**       | 80%以上                   | 並列処理時      |
| **特徴削減率**       | 30%以上                   | Boruta 適用後   |

### 実装品質

- **テストカバレッジ**: 80%以上
- **コードスタイル**: Black、Flake8、MyPy 準拠
- **ドキュメント**: 全クラス・関数に docstring
- **新規ファイル数**: 12 ファイル（実装 6 + テスト 6）

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

### 5. オンチェーンデータソース拡張

- **追加候補**: Santiment、IntoTheBlock、Nansen、Dune Analytics
- **検討条件**: Phase 2 実装後、有効性確認、API 予算確保
- **期待効果**: 2-5%追加向上

---

## 📚 参考文献

### 機械学習・アンサンブル学習

1. Şimşek (2025) - "Stacked Generalization Model for BIST100 Index" - Research Square
2. Biswas et al. (2025) - "Stock Price Prediction Using Stacked Heterogeneous Ensemble" - MDPI Mathematics
3. Mazinani et al. (2025) - "Transformer-based Cryptocurrency Prediction" - Journal of Big Data
4. Urooj et al. (2024) - "Ensemble ML vs Deep Learning Methods" - 比較分析

### 市場マイクロストラクチャー・オンチェーン分析

5. "On-chain to Macro: Data Source Diversity" (2024) - VLDB
6. Chainalysis (2025) - "Crypto Crime Report 2025"
7. "Crypto Foretell" (2025) - Journal of Big Data

### 特徴量エンジニアリング

8. "Deep Learning for Stock Market Prediction" - ResearchGate（Boruta アルゴリズム）
9. "Forward Feature Selection: Empirical Analysis" - ResearchGate

### CPU 最適化

10. Intel Developer (2024) - "Faster XGBoost/LightGBM on CPU"
11. Intel Distribution for Python Release Notes - daal4py 変換

### 技術ドキュメント

- [LightGBM Parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html)
- [XGBoost Tree Methods](https://xgboost.readthedocs.io/en/stable/treemethod.html)
- [TabNet Paper](https://arxiv.org/abs/1908.07442)
- [Glassnode API](https://docs.glassnode.com/)
- [CryptoQuant API](https://cryptoquant.com/docs)
- [Boruta Algorithm](https://www.jstatsoft.org/article/view/v036i11)

---

## 🎯 まとめ

### 主要な変更点

1. GPU 依存完全削除（将来検討事項として記録）
2. LightGBM 中心構成（最速・最軽量）
3. Random Forest メタモデル（2025 年最新研究で実証）
4. Intel oneDAL 統合（CPU 推論 2-3 倍高速化）
5. オンチェーンデータ統合（2024 年 VLDB 論文で最重要特徴）
6. 高度な特徴選択（Boruta、過学習 30%削減）
7. 相互作用特徴量（ドメイン知識ベース 8 種類）
8. 現実的期間（7-9 週間）

### 期待される効果

- **予測精度向上**: 合計 **10-18%**（保守的見積もり）
- **推論速度向上**: **2-3 倍**（Intel oneDAL 利用時）
- **過学習リスク削減**: **30%**（Boruta 適用後）
- **実装期間**: 7-9 週間
- **メモリ使用量**: 1-2GB
- **CPU 使用率**: 80%以上
- **個人環境で実用可能**: GPU 不要

### 次のステップ

1. Phase 1 開始: `TabNetFeatureExtractor`実装
2. 段階的評価: 各週ごと進捗確認
3. 柔軟な調整: 精度・性能に応じて微調整
4. 将来の拡張: GPU 環境時に TFT/CatBoost 検討
