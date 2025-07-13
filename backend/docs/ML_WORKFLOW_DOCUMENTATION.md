# オートストラテジー MLワークフロー解説

このドキュメントは、本システムにおける機械学習（ML）モデルを利用したオートストラテジーのワークフローを、設定値やデータ処理の詳細を含めて解説します。

## 1. 概要

本システムのMLワークフローは、大きく分けて「**モデル学習フェーズ**」と「**戦略利用フェーズ**」の2つから構成されます。さらに、これらのプロセスを最適化するための「**ベイジアン最適化フェーズ**」が存在します。

1.  **モデル学習フェーズ**: 過去の市場データから、未来の価格変動を予測するLightGBMモデルを学習させます。
2.  **戦略利用フェーズ**: GA（遺伝的アルゴリズム）において、学習済みモデルを指標として利用し、高度な取引戦略を探索します。
3.  **ベイジアン最適化フェーズ**: GA自体のパラメータや、MLモデルのハイパーパラメータを効率的に探索し、システム全体のパフォーマンスを向上させます。

## 2. 主要コンポーネントと役割

| コンポーネント | 役割 | 主要ファイル |
| --- | --- | --- |
| **API Endpoints** | ユーザーからのリクエストを受け付けるインターフェース。 | `backend/app/api/ml_training.py`<br>`backend/app/api/ml_management.py`<br>`backend/app/api/auto_strategy.py`<br>`backend/app/api/bayesian_optimization.py` |
| `MLTrainingService` | モデル学習のコアロジックを担う。 | `backend/app/core/services/ml/ml_training_service.py` |
| `FeatureEngineeringService` | 生データからモデルが使用する「特徴量」を計算する。 | `backend/app/core/services/ml/feature_engineering/feature_engineering_service.py` |
| `LightGBMTrainer` | LightGBMモデルの学習、評価、予測の具体的な実装を提供。 | `backend/app/core/services/ml/lightgbm_trainer.py` |
| `ModelManager` | 学習済みモデルの保存、読み込み、バージョン管理を行う。 | `backend/app/core/services/ml/model_manager.py` |
| `AutoStrategyService` | GAによる戦略生成のライフサイクル全体を管理する。 | `backend/app/core/services/auto_strategy/services/auto_strategy_service.py` |
| `StrategyFactory` | 戦略遺伝子を実行可能なPythonコードに変換する。 | `backend/app/core/services/auto_strategy/factories/strategy_factory.py` |
| `MLOrchestrator` | **学習と推論の橋渡し役**。ML指標をオンデマンドで提供する。 | `backend/app/core/services/auto_strategy/services/ml_orchestrator.py` |
| `MLSignalGenerator` | 学習済みモデルを保持し、実際の予測確率を計算する。 | `backend/app/core/services/ml/signal_generator.py` |
| `BacktestDataService` | 複数のデータソースをバックテスト用に結合・整形する。 | `backend/app/core/services/backtest_data_service.py` |
| `BayesianOptimizer` | `skopt`を使用し、GAやMLのパラメータをベイジアン最適化する。 | `backend/app/core/services/optimization/bayesian_optimizer.py` |
| **Config Files** | システム全体の動作を規定する設定値。 | `backend/app/core/config/ml_config.py`<br>`backend/app/core/services/auto_strategy/models/ga_config.py` |

## 3. ワークフロー詳細

### フェーズ1: モデル学習 (Model Training)

**フロー:**

1.  **APIトリガー**: `POST /api/ml/train`
2.  **データ収集と結合**: `BacktestDataService`がOHLCV, Funding Rate, Open Interestを`pd.merge_asof`で結合。
3.  **特徴量エンジニアリング**: `FeatureEngineeringService`が特徴量を計算。
4.  **ラベル付け**: `BaseMLTrainer`が、`ml_config.py`の設定（`prediction_horizon: 24`, `threshold_up: 0.02`, `threshold_down: -0.02`）に基づき、上昇(2)/レンジ(1)/下落(0)のラベルを付与。
5.  **学習と評価**: `LightGBMTrainer`がデータを標準化（`StandardScaler`）し、LightGBMモデルを学習。
6.  **モデルの保存**: `ModelManager`が学習済みモデル、スケーラー、特徴量リストを単一の`.joblib`ファイルとして`models/`ディレクトリに保存。

### フェーズ2: 戦略内でのモデル利用 (Inference)

**フロー:**

1.  **モデルの自動読み込み**: `AutoStrategyService`の初期化時に、`MLOrchestrator`が最新の学習済みモデルをロード。
2.  **戦略のコンパイル**: GAが`ML_UP_PROB`などを含む戦略を生成。`StrategyFactory`はこれを実行可能なコードに変換。
3.  **バックテスト実行と推論**: バックテスト中、戦略コードが`MLOrchestrator`にML指標を要求。
4.  **予測確率の計算**: `MLOrchestrator`が`FeatureEngineeringService`で特徴量を計算し、`MLSignalGenerator`がロード済みのスケーラーとモデルで予測確率（e.g., `{"up": 0.7, ...}`）を計算。
5.  **戦略条件の評価**: 予測確率が戦略の条件（例: `ML_UP_PROB > 0.6`）を満たすか評価し、取引のトリガーとする。

### フェーズ3: ベイジアン最適化 (Bayesian Optimization)

このフェーズは、GAやMLモデルの性能をさらに引き出すために行われます。

**目的**:

*   **GAパラメータ最適化**: より良い戦略を生成できるGAの設定（`population_size`など）を見つける。
*   **MLハイパーパラメータ最適化**: モデルの予測精度を向上させるためのLightGBMの設定（`num_leaves`など）を見つける。

**フロー:**

1.  **APIトリガー**:
    *   ユーザーがUIから `POST /api/bayesian-optimization/ga-parameters` または `ml-hyperparameters` を呼び出します。

2.  **目的関数の定義**:
    *   `BayesianOptimizer`内で、最適化の「良さ」を測るための目的関数が定義されます。
    *   **GA最適化の場合**: 目的関数は、試行するGAパラメータで**実際にGAとバックテストを実行**し、その結果のパフォーマンス（デフォルトではSQN）をスコアとして返します。
    *   **ML最適化の場合**: 目的関数は、試行するハイパーパラメータで**実際にモデルを学習・評価**し、その精度をスコアとして返します。(注: 現在の実装はダミーで、ランダム値を返します)

3.  **最適化の実行**:
    *   `BayesianOptimizer`サービスが`skopt.gp_minimize`を呼び出します。
    *   `gp_minimize`は、過去の評価結果を基に次に試すべき最も有望なパラメータをベイズ推定で効率的に探索し、指定回数（`n_calls`）だけ目的関数の評価を繰り返します。

4.  **結果の返却**:
    *   最適化が完了すると、最も高いスコアを記録したパラメータの組み合わせ（`best_params`）と、そのスコア（`best_score`）がAPIから返されます。