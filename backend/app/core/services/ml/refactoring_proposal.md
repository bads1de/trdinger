# ML サービスのリファクタリング提案

`backend/app/core/services/ml` パッケージの現状コードを分析し、保守性、可読性、テスト容易性を向上させるためのリファクタリング案を以下に提案します。

## 1. 全体的な課題

現在の実装は機能していますが、いくつかの構造的な問題を抱えています。

- **責務の重複**: 特に `MLSignalGenerator` と `BaseMLTrainer` エコシステム間で、学習やデータ準備のロジックが大きく重複しています。
- **設計原則の改善点**: SOLID 原則、特に単一責任の原則と依存性逆転の原則に関して改善の余地があります。
- **コードの冗長性**: 同じような処理の繰り返しや、冗長な記述が見られます。

## 2. 主要なリファクタリングポイント

### 2.1. `MLSignalGenerator` の廃止と機能統合 (最優先)

**問題点:**

- [`signal_generator.py`](backend/app/core/services/ml/signal_generator.py) は、[`lightgbm_trainer.py`](backend/app/core/services/ml/lightgbm_trainer.py) や [`base_ml_trainer.py`](backend/app/core/services/ml/base_ml_trainer.py) と同様の学習、データ準備、予測ロジックを独自に実装しており、コードが著しく重複しています。
- 例えば、`MLSignalGenerator.train()` ([`backend/app/core/services/ml/signal_generator.py:127`](backend/app/core/services/ml/signal_generator.py:127)) は `LightGBMTrainer._train_model_impl()` ([`backend/app/core/services/ml/lightgbm_trainer.py:107`](backend/app/core/services/ml/lightgbm_trainer.py:107)) とほぼ同じ内容です。
- このクラスの存在は、メンテナンスコストを増大させ、バグの温床となります。

**提案:**

1.  **`MLSignalGenerator` クラスを完全に廃止します。**
2.  シグナル生成が必要な箇所では、[`ml_training_service.py`](backend/app/core/services/ml/ml_training_service.py) のインスタンスを使用し、その `predict()` メソッドを呼び出すように変更します。
3.  `MLSignalGenerator` にのみ存在する特定のロジックがあれば、それを `BaseMLTrainer` または `LightGBMTrainer` に移植します。

### 2.2. 依存関係の逆転 (設定オブジェクトの注入)

**問題点:**

- 多くのクラス（例: [`base_ml_trainer.py:66`](backend/app/core/services/ml/base_ml_trainer.py:66), [`ml_training_service.py:57`](backend/app/core/services/ml/ml_training_service.py:57)）が、グローバルな `ml_config` オブジェクトを直接インポートして使用しています。
- これにより、各クラスは特定のコンフィグ実装に強く結合しており、テストが困難になっています。

**提案:**

- 各クラスのコンストラクタ（`__init__`）で、設定オブジェクトを引数として受け取るように変更します（依存性の注入）。
- これにより、テスト時にモックの設定オブジェクトを簡単に渡すことができ、コンポーネントの再利用性も向上します。

**修正例 (`BaseMLTrainer`):**

```python
# 修正前
from .config import ml_config

class BaseMLTrainer(ABC):
    def __init__(self, automl_config: Optional[Dict[str, Any]] = None):
        self.config = ml_config
        # ...

# 修正後
from .config import MLConfig # 仮の型

class BaseMLTrainer(ABC):
    def __init__(self, config: MLConfig, automl_config: Optional[Dict[str, Any]] = None):
        self.config = config
        # ...
```

### 2.3. 冗長なコードの削減

**問題点:**

- [`base_ml_trainer.py:197-235`](backend/app/core/services/ml/base_ml_trainer.py:197) におけるモデルのメタデータ構築は、`dict.get()` の呼び出しが連続しており、非常に冗長です。
- [`lightgbm_trainer.py:60-71`](backend/app/core/services/ml/lightgbm_trainer.py:60) の予測時の特徴量選択ロジックが複雑になっています。

**提案:**

- **メタデータ構築の簡素化**: ループや `dict.update` を使用して、コードを簡潔にします。

  ```python
  # 修正例
  model_metadata = {}
  metric_keys = [
      "accuracy", "precision", "recall", "f1_score", "auc_roc",
      "auc_pr", "balanced_accuracy", "matthews_corrcoef", "cohen_kappa",
      "specificity", "sensitivity", "npv", "ppv", "log_loss", "brier_score"
  ]
  for key in metric_keys:
      model_metadata[key] = training_result.get(key, 0.0)

  other_info = {
      "training_samples": training_result.get("train_samples", 0),
      "test_samples": training_result.get("test_samples", 0),
      "feature_count": len(self.feature_columns) if self.feature_columns else 0,
  }
  model_metadata.update(other_info)
  ```

- **特徴量選択の簡素化**: 学習時に使用した特徴量リスト `self.feature_columns` を信頼し、予測時もそれに基づいてデータを整形することで、ロジックを単純化できます。

### 2.4. `MLTrainingService` の責務の明確化

**問題点:**

- [`ml_training_service.py:100-114`](backend/app/core/services/ml/ml_training_service.py:100) の `train_model` メソッド内で、`automl_config` の有無によって `LightGBMTrainer` のインスタンスを動的に再生成しています。
- これにより、`MLTrainingService` の状態が呼び出しごとに変化する可能性があり、クラスの責務が曖昧になっています。

**提案:**

- `MLTrainingService` のコンストラクタで、`trainer` のインスタンスを外部から注入するようにします。
- `automl_config` を使用したい場合は、クライアント側で `LightGBMTrainer(automl_config=...)` を生成し、それを `MLTrainingService` に渡すようにします。これにより、`MLTrainingService` は純粋なサービス提供に専念できます。

### 2.5. `PerformanceExtractor` の設計見直し

**問題点:**

- [`performance_extractor.py`](backend/app/core/services/ml/performance_extractor.py) は静的メソッドのみで構成されており、実質的に名前空間として機能しています。クラスとしてインスタンス化されることがないため、クラス定義は冗長です。

**提案:**

- `PerformanceExtractor` クラスを廃止し、モジュールレベルの関数（例: `extract_performance_metrics(...)`）として提供します。
- ファイル名も `performance_utils.py` のような、ユーティリティモジュールであることが分かりやすい名前に変更することを検討します。

## 3. リファクタリングの進め方（案）

1.  **Step 1: `MLSignalGenerator` の削除**
    - 最も影響範囲が広く、効果も大きいため、最初に実施します。
    - 関連するテストコードを修正し、`MLTrainingService` を使うように変更します。
2.  **Step 2: 依存関係の逆転**
    - 各クラスに設定オブジェクトを注入できるように `__init__` を変更します。
    - これに伴い、インスタンスを生成している箇所の修正も必要です。
3.  **Step 3: その他の改善**
    - `MLTrainingService` の責務を明確化し、冗長なコードをクリーンアップします。
    - `PerformanceExtractor` をユーティリティモジュールに変更します。

これらのリファクタリングにより、コードベース全体の健全性が向上し、将来の機能追加や変更が容易になります。
