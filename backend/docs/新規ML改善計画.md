# 新規 ML 改善計画

## 1. 概要

現在の ML モデルの性能（macro F1 スコア約 41.4%）を改善するため、特に効果が大きいと予想される以下の 2 つの機能改善を優先的に実装します。本計画は、TDD（テスト駆動開発）アプローチに基づき、各機能が堅牢に実装されることを目指します。

- **フェーズ 1: ラベリング戦略の改善**: ノイズの多い現在のラベリングを見直し、より明確な教師データを作成することで、モデルの学習効率と精度を根本から改善します。
- **フェーズ 2: クラス不均衡対策**: 予測したい「UP/DOWN」クラスが「RANGE」クラスに比べて少ないというデータ不均衡問題に対処し、モデルが希少な重要イベントを見逃さないようにします。

---

## 2. フェーズ 1: ラベリング戦略の改善

### 2.1 目的

現在のラベリング（16 時間ホライズン、0.2%閾値）はノイズが多く、クラス間の境界が曖昧になる原因となっています。より短いホライズンと適切な閾値を持つ新しいラベル生成プリセットを導入し、教師データの質を向上させます。

### 2.2 実装内容

#### 2.2.1 新しいプリセットの追加

**ファイル:** `backend/app/utils/label_generation/presets.py`

`get_common_presets` 関数に、以下の新しいプリセット定義を追加します。

```python
# 4時間足、0.5%閾値プリセット
"4h_4bars_050": {
    "timeframe": "4h",
    "horizon_n": 4,
    "threshold": 0.005,  # 0.5%
    "threshold_method": ThresholdMethod.FIXED,
    "description": "4時間足、4本先（16時間先）、0.5%閾値"
},
"4h_4bars_100": {
    "timeframe": "4h",
    "horizon_n": 4,
    "threshold": 0.010,  # 1.0%
    "threshold_method": ThresholdMethod.FIXED,
    "description": "4時間足、4本先（16時間先）、1.0%閾値"
},
# （注: 8h足はSUPPORTED_TIMEFRAMESに追加が必要なため、一旦保留）
```

#### 2.2.2 デフォルト設定の更新

**ファイル:** `backend/app/config/unified_config.py`

`LabelGenerationConfig` データクラスのデフォルト値を、新しい推奨プリセットに変更します。

```python
@dataclass
class LabelGenerationConfig:
    # デフォルトプリセットをノイズの少ないものに変更
    default_preset: str = "4h_4bars_dynamic" # または "4h_4bars_050"
    timeframe: str = "4h"
    horizon_n: int = 4
    threshold: float = 0.005
    # ... 以下同じ
```

### 2.3 テスト戦略

**新規テストファイル:** `backend/tests/ml/test_label_generation_improvements.py`

TDD に基づき、まず以下のテストを記述し、失敗することを確認してから実装に着手します。

```python
import pytest
from app.utils.label_generation.presets import get_common_presets

class TestLabelGenerationImprovements:
    def test_new_presets_exist(self):
        """新しいプリセットが定義されていることを確認"""
        presets = get_common_presets()
        assert "4h_4bars_050" in presets
        assert "4h_4bars_100" in presets

    def test_threshold_050_reduces_noise(self, sample_ohlcv_data):
        """閾値を上げることでノイズ（RANGEラベル）が減ることを確認"""
        labels_020 = forward_classification_preset(sample_ohlcv_data, threshold=0.002)
        labels_050 = forward_classification_preset(sample_ohlcv_data, threshold=0.005)

        range_ratio_020 = (labels_020 == "RANGE").sum() / len(labels_020.dropna())
        range_ratio_050 = (labels_050 == "RANGE").sum() / len(labels_050.dropna())

        assert range_ratio_050 < range_ratio_020
```

### 2.4 期待効果

- macro F1 スコアを **+5〜10 ポイント** 改善
- DOWN/UP クラスの予測精度向上
- RANGE クラスへのバイアス軽減

---

## 3. フェーズ 2: クラス不均衡対策

### 3.1 目的

モデルが多数派の「RANGE」クラスに過剰適合するのを防ぎ、少数派である「UP」「DOWN」クラスの誤分類コストを引き上げることで、予測性能のバランスを改善します。

### 3.2 実装内容

#### 3.2.1 `class_weight` の動的設定

**ファイル:** `backend/app/services/ml/models/lightgbm.py` （および `xgboost.py`）

`_train_model_impl` メソッドを修正し、学習パラメータとして渡された `class_weight` 設定をモデルに適用できるようにします。

```python
# _train_model_impl内
def _train_model_impl(self, ..., **kwargs):
    # ...
    # training_paramsからclass_weightを取得
    class_weight_setting = kwargs.get("class_weight") # e.g., 'balanced' or a dict

    # LightGBM/XGBoostのパラメータに変換
    params = { ... }
    if class_weight_setting:
        params["class_weight"] = class_weight_setting

    # ... モデル学習
```

#### 3.2.2 SMOTE/ADASYN によるオーバーサンプリング

**新規ファイル:** `backend/app/services/ml/data_processing/sampling.py`

`imblearn`ライブラリを使用して、少数派クラスをオーバーサンプリングする`ImbalanceSampler`クラスを実装します。

```python
from imblearn.over_sampling import SMOTE, ADASYN

class ImbalanceSampler:
    def __init__(self, method: str = "smote", random_state: int = 42):
        if method == "smote":
            self.sampler = SMOTE(random_state=random_state)
        elif method == "adasyn":
            self.sampler = ADASYN(random_state=random_state)
        else:
            raise ValueError(f"Unknown sampling method: {method}")

    def fit_resample(self, X, y):
        return self.sampler.fit_resample(X, y)
```

**ファイル:** `backend/app/services/ml/base_ml_trainer.py`

クロスバリデーションのループ内で、訓練データにのみ`ImbalanceSampler`を適用するロジックを追加します。これにより、テストデータへのリークを防ぎます。

```python
# _time_series_cross_validate内
# ...
for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
    X_train_cv, y_train_cv = X.iloc[train_idx], y.iloc[train_idx]

    if use_smote: # training_paramsから取得
        sampler = ImbalanceSampler(method=smote_method)
        X_train_cv, y_train_cv = sampler.fit_resample(X_train_cv, y_train_cv)

    # ...以降の学習処理
```

#### 3.2.3 設定の追加

**ファイル:** `backend/app/config/unified_config.py`

`MLTrainingConfig`クラスに、これらの機能を制御するための設定項目を追加します。

```python
class MLTrainingConfig(BaseSettings):
    # ...
    # クラス不均衡対策
    use_class_weight: bool = Field(default=False, description="class_weightを使用するか")
    class_weight_mode: str = Field(default="balanced", description="class_weightモード ('balanced' or custom dict)")
    use_smote: bool = Field(default=False, description="SMOTE/ADASYNを使用するか")
    smote_method: str = Field(default="smote", description="サンプリング方法 ('smote' or 'adasyn')")
```

### 3.3 テスト戦略

**新規テストファイル:** `backend/tests/ml/test_class_imbalance.py`

SMOTE が少数派クラスのサンプル数を正しく増やすか、`class_weight='balanced'`が適切な重みを計算するかを検証する単体テストを記述します。

```python
class TestClassImbalance:
    def test_smote_increases_minority_samples(self):
        """SMOTEが少数派クラスを増やすことを確認"""
        # ... 不均衡データを作成 ...
        sampler = ImbalanceSampler(method="smote")
        _, y_resampled = sampler.fit_resample(X, y)
        assert y_resampled.value_counts().min() > y.value_counts().min()

    def test_class_weight_passed_to_model(self):
        """class_weightパラメータがモデルに渡されることを確認（モック使用）"""
        # ... MLTrainingServiceのtrain_modelを呼び出し、
        # LightGBM/XGBoostのfit/trainメソッドが適切な重みパラメータと共に
        # 呼び出されたことをアサートする ...
        pass
```

### 3.4 期待効果

- macro F1 スコアを **+3〜7 ポイント** 改善
- 特に DOWN/UP クラスの F1 スコア向上
- クラス間の予測バランス改善

---

## 4. 優先順位とロードマップ

1. **優先度 1: フェーズ 1（ラベリング戦略の改善）**

   - **理由**: モデル性能の基盤となる教師データの質を改善するため、最も優先度が高い。
   - **実装工数**: 約 2-3 日

2. **優先度 2: フェーズ 2（クラス不均衡対策）**
   - **理由**: 改善されたラベルに対して不均衡対策を行うことで、さらなる性能向上が期待できる。
   - **実装工数**: 約 2-4 日（`imblearn`の導入と CV ループへの統合を含む）
