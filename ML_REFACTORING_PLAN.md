# ML サービス リファクタリング計画

## 1. 概要

`backend/app/services/ml` ディレクトリ配下のコードベースにおいて、責務の分離、重複コードの排除、およびメンテナンス性の向上を目的としたリファクタリング計画です。

## 2. 削除対象ファイル

以下のファイルは、新しい実装に置き換えられているか、使用されていないため削除します。

- **`backend/app/services/ml/stacking_service.py`**
  - 理由: `backend/app/services/ml/ensemble/stacking.py` (`StackingEnsemble`) に機能が統合されており、`EnsembleTrainer` からも参照されていないため。

## 3. ファイル移動と構造整理

以下のファイルは、より適切なディレクトリに移動し、パッケージ構造を整理します。

- **`backend/app/services/ml/meta_labeling_service.py`**
  - 移動先: `backend/app/services/ml/ensemble/meta_labeling.py`
  - 理由: メタラベリングはアンサンブル学習（スタッキング）の一部として機能するため、`ensemble` パッケージに配置するのが適切です。

## 4. 責務の分離とコード削減

### 4.1. ラベル生成ロジックの分離 (`BaseMLTrainer`)

- **現状**: `BaseMLTrainer._prepare_training_data` 内に、ラベル生成の設定解析、プリセット適用、カスタムロジックが混在しています。
- **改善案**: `LabelGenerationService` (または `LabelManager`) クラスを作成し、ラベル生成に関する全ての責務を委譲します。
- **メリット**: トレーナークラスのコード行数を削減し、ラベル生成ロジックの変更がトレーナーに影響を与えないようにします。

### 4.2. 最適化ロジックの分離 (`MLTrainingService`)

- **現状**: `MLTrainingService` 内に Optuna を使用したハイパーパラメータ最適化のロジック (`_train_with_optimization`, `_create_objective_function`) が直接実装されています。
- **改善案**: `OptimizationService` (または `HyperparameterTuner`) クラスを作成し、最適化プロセスを委譲します。
- **メリット**: `MLTrainingService` を学習のオーケストレーションのみに集中させ、可読性を向上させます。

### 4.3. 特徴量後処理の分離 (`FeatureEngineeringService`)

- **現状**: `calculate_advanced_features` メソッド内で、特徴量の計算だけでなく、NaN 処理、クリッピング、データ型最適化などの後処理が行われています。
- **改善案**: `FeaturePostProcessor` クラスを作成し、計算後のデータ加工ロジックを分離します。
- **メリット**: 特徴量エンジニアリングのメインフローが明確になり、後処理ロジックの単体テストが容易になります。

### 4.4. RNN モデルの共通化 (`models/lstm_model.py`, `models/gru_model.py`)

- **現状**: `LSTMModel` と `GRUModel` の実装（データセット作成、学習ループ、予測ロジック）がほぼ完全に重複しています。
- **改善案**: `BaseRNNModel` (または `BaseTorchModel`) を作成し、共通ロジックを集約します。
- **メリット**: コードの重複を排除し、新しい RNN 系モデル（Bi-LSTM など）の追加が容易になります。

### 4.5. 最適化ロジックの委譲 (`MLTrainingService`)

- **現状**: `MLTrainingService` が `OptunaOptimizer` を使用していますが、目的関数の構築やパラメータ空間の設定など、最適化の詳細に深く関与しています。
- **改善案**: `OptimizationService` を拡張し、トレーナーとデータを受け取って最適化を実行する高レベル API を提供します。
- **メリット**: `MLTrainingService` のコードを大幅に削減し、トレーニングフローと最適化フローを明確に分離できます。

### 4.6. 前処理パイプラインの統合 (`preprocessing/pipeline.py`)

- **現状**: `FeatureEngineeringService` が独自にスケーリングや後処理を行っている部分がありますが、`preprocessing` パッケージには `scikit-learn` ベースのパイプライン構築機能が既に存在します。
- **改善案**: `FeatureEngineeringService` から汎用的な前処理（スケーリング、特徴量選択）を切り離し、`preprocessing` パッケージのパイプラインを活用するようにします。
- **メリット**: 前処理ロジックの一元管理と、`scikit-learn` エコシステムとの親和性が向上します。

### 4.7. 特徴量エンジニアリングの再構築 (`feature_engineering`)

- **現状**: `AdvancedFeatureEngineer` (`advanced_features.py`) が肥大化しており、`TechnicalFeatureCalculator` (`technical_features.py`) と機能が重複しています（例: RSI, CCI の計算）。また、一部のクラスが `BaseFeatureCalculator` を継承しておらず、一貫性が欠けています。
- **改善案**:
  - `AdvancedFeatureEngineer` を解体し、機能単位（MTF、Range 検出、市場ダイナミクスなど）で新しい計算クラス（`MTFFeatureCalculator`, `RangeFeatureCalculator` 等）に分割します。
  - 重複するテクニカル指標計算を `TechnicalFeatureCalculator` に統合します。
  - 全ての特徴量計算クラスで `BaseFeatureCalculator` を継承し、エラーハンドリングや検証ロジックを統一します。
- **メリット**: コードの重複排除、可読性向上、テスト容易性の向上、および新しい特徴量の追加が容易になります。

### 4.8. 特徴量選択の統合 (`feature_selection`)

- **現状**: `FeatureSelector` (`feature_selector.py`) が独自に実装されていますが、`preprocessing` パッケージにも類似のパイプライン機能が存在します。
- **改善案**: `FeatureSelector` の高度な機能（アンサンブル選択など）を `preprocessing` パッケージの Transformer として再実装または統合します。
- **メリット**: 前処理パイプラインの一元化と、`scikit-learn` エコシステムとの完全な互換性を確保できます。

## 5. 実施手順

1. **不要ファイルの削除**: `stacking_service.py` を削除。 (完了)
2. **ファイルの移動**: `meta_labeling_service.py` を移動し、関連するインポートを修正。 (完了)
3. **ラベル生成の分離**: `LabelGenerationService` を作成し、`BaseMLTrainer` をリファクタリング。 (完了)
4. **RNN モデルの共通化**: `BaseRNNModel` を作成し、`LSTMModel`, `GRUModel` をリファクタリング。 (完了)
5. **最適化ロジックの分離**: `OptimizationService` を拡張し、`MLTrainingService` をリファクタリング。 (完了)
6. **特徴量エンジニアリングの再構築**: `AdvancedFeatureEngineer` を解体・統合し、`BaseFeatureCalculator` を適用。
7. **特徴量後処理・選択の分離と統合**: `FeaturePostProcessor` を作成し、`preprocessing` パッケージとの連携を強化。

## 6. 期待される効果

- **コードの凝集度向上**: 各クラスが単一の責任を持つようになり、理解しやすくなります。
- **テスト容易性**: ロジックが分離されることで、各コンポーネントの単体テストが書きやすくなります。
- **メンテナンス性**: 将来的な機能追加（新しい最適化手法の導入など）が容易になります。
