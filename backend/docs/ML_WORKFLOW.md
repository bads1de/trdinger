# ML ワークフローとアーキテクチャ (詳細版)

**最終更新日**: 2025-07-11
このドキュメントでは、本システムに実装されている機械学習（ML）のワークフローとアーキテクチャについて、実装レベルの詳細を含めて解説します。

## 1. 概要

本 ML システムは、過去の市場データから将来の価格変動（上昇・下落・レンジ）を予測し、その予測確率を自動戦略生成（オートストラテジー）のインプットとして利用することを目的としています。これにより、従来のテクニカル指標のみに依存するのではなく、より高度なデータ駆動型のアプローチで優位性の高い取引戦略を探索します。

システムは、**特徴量生成**、**モデル学習・予測**、**指標提供**、**運用・最適化**の 4 つの主要な責務に分かれたコンポーネント群で構成されています。

## 2. アーキテクチャ

```mermaid
graph TD
    subgraph データ層
        A[データソース (OHLCV, FR, OI)]
    end

    subgraph MLコア
        B(FeatureEngineeringService)
        C(MLSignalGenerator)
        D[学習済みモデル (.pkl)]
    end

    subgraph 運用・最適化
        G(AutoRetrainingScheduler)
        H(BayesianOptimizer)
    end

    subgraph 戦略生成層
        E(MLIndicatorService)
        F[オートストラテジーGA]
    end

    A --> B
    B --> C
    C -- 学習・予測 --> E
    C -- 保存・読込 --> D
    E --> F
    G -- 再学習トリガー --> C
    H -- 最適化 --> C
```

- **データフロー**: `データソース` → `FeatureEngineeringService` → `MLSignalGenerator` → `MLIndicatorService` → `オートストラテジーGA`
- **制御フロー**: `BayesianOptimizer`や`AutoRetrainingScheduler`が`MLSignalGenerator`の学習プロセスを制御・最適化します。

## 3. 主要コンポーネント詳細

### 3.1. `FeatureEngineeringService`

- **ファイル**: [`backend/app/core/services/feature_engineering/feature_engineering_service.py`](backend/app/core/services/feature_engineering/feature_engineering_service.py)
- **役割**: 生データから ML モデルの学習に有効な多角的な特徴量を生成する、データ前処理の心臓部です。
- **実装詳細**:
  - **`calculate_advanced_features`**: 全ての特徴量計算を統括するメインメソッド。
  - **豊富な特徴量群**:
    - **価格**: 移動平均(SMA)、価格と SMA の乖離、モメンタム
    - **ボラティリティ**: 実現ボラティリティ、ATR、ボラティリティスパイク
    - **出来高**: 出来高移動平均、出来高比率
    - **市場データ**: ファンディングレート(FR)、建玉残高(OI)から、それぞれの変化量や市場の過熱感を示す複合指標（`Price_FR_Divergence`, `Market_Heat_Index`など）を計算。
    - **高度な特徴**: 市場レジーム（トレンド/レンジ）、RSI や MACD などのモメンタム指標、ダイバージェンスなどのパターン認識特徴まで、幅広く生成します。
  - **効率化**:
    - **キャッシュ機構**: `_generate_cache_key`でデータとパラメータのハッシュからキーを生成し、計算結果をメモリ内にキャッシュすることで、同一条件での再計算を回避します。
    - **データ型最適化**: `_optimize_dtypes`メソッドで、`float64`を`float32`に、`int64`を`int32`に変換し、メモリ使用量を削減します。

### 3.2. `MLSignalGenerator`

- **ファイル**: [`backend/app/core/services/ml/signal_generator.py`](backend/app/core/services/ml/signal_generator.py)
- **役割**: ML モデルの学習、予測、管理のライフサイクル全体を担うコアコンポーネント。
- **実装詳細**:
  - **モデル**: `LightGBM`による 3 クラス分類（上昇:2, レンジ:1, 下落:0）モデルを採用。
  - **`prepare_training_data`**:
    - 未来の価格変動率 (`future_returns`) を計算し、`threshold_up` / `threshold_down` を閾値として 3 クラスのラベルを付与します。
    - これにより、モデルは「将来`prediction_horizon`期間後に価格がどうなっているか」を学習します。
  - **`train`**:
    - **時系列分割**: データの時間的順序を維持するため、`train_test_split`の`shuffle=False`の代わりに、インデックスで訓練データとテストデータを分割します。
    - **標準化**: `StandardScaler`を用いて特徴量のスケールを揃え、モデルの学習を安定させます。このスケーラーはモデルと共に保存されます。
    - **過学習抑制**: `lightgbm.early_stopping`コールバックを使用し、テストデータの性能が改善しなくなった時点で学習を打ち切ります。
  - **`predict`**:
    - 学習時に使用した`StandardScaler`を適用してから予測を実行し、学習時と予測時でデータの一貫性を保ちます。
    - 結果は各クラスの確率を持つ辞書 (`{"up": float, "down": float, "range": float}`) として返されます。
  - **`save_model` / `load_model`**:
    - `joblib`を使用し、学習済みモデルオブジェクト、`StandardScaler`オブジェクト、特徴量カラム名のリストを一つの`.pkl`ファイルにシリアライズして保存・読み込みします。これにより、モデルの再現性を担保します。

### 3.3. `MLIndicatorService`

- **ファイル**: [`backend/app/core/services/auto_strategy/services/ml_indicator_service.py`](backend/app/core/services/auto_strategy/services/ml_indicator_service.py)
- **役割**: ML コア機能（特徴量計算と予測）をカプセル化し、オートストラテジー(GA)エンジンに対してシンプルなインターフェースを提供するファサード（Facade）としての役割を果たします。
- **実装詳細**:
  - **`calculate_ml_indicators`**:
    - このサービスの主要な口。`FeatureEngineeringService`と`MLSignalGenerator`を順に呼び出し、最終的に GA が利用できる指標（`ML_UP_PROB`など）を`np.ndarray`形式で返します。
    - GA エンジンは、このメソッドを呼び出すだけで ML の恩恵を受けることができます。
  - **堅牢なエラーハンドリング**:
    - `_safe_ml_prediction`メソッドでは、モデルが未学習の場合や予測中にエラーが発生した場合でも、システムが停止しないようにデフォルト値や前回成功した予測値を返すフォールバック機構を備えています。
    - 入力データの検証、メモリ使用量チェック、計算のタイムアウト処理など、本番運用を想定した防御的プログラミングが施されています。
  - **`train_model`**: `MLSignalGenerator`の学習プロセスをラップし、サービス利用者（例: 再学習スケジューラ）に対して簡単な学習インターフェースを提供します。

### 3.4. `AutoRetrainingScheduler`

- **ファイル**: [`backend/app/core/services/auto_retraining/auto_retraining_scheduler.py`](backend/app/core/services/auto_retraining/auto_retraining_scheduler.py)
- **役割**: モデルの陳腐化を防ぎ、性能を維持するための自動再学習プロセスを管理します。
- **実装詳細**:
  - **非同期実行**: `threading`モジュールを使用し、スケジューラをバックグラウンドスレッドで実行することで、メインアプリケーションの動作を妨げません。
  - **ジョブ管理**: `RetrainingJob`データクラスで再学習タスクを管理。`pending`, `running`, `completed`, `failed`といったステータスを持ちます。
  - **柔軟なトリガー**: `RetrainingTrigger` Enum により、`SCHEDULED`（定期実行）、`PERFORMANCE_DEGRADATION`（性能劣化）、`MANUAL`（手動）など、様々なきっかけで再学習を開始できる設計になっています。
  - **コールバック方式**: `register_retraining_callback`でモデルごとの実際の学習処理を外部から注入（Dependency Injection）できます。これにより、スケジューラ自体は特定のモデルの実装に依存せず、汎用性を保っています。

### 3.5. `BayesianOptimizer`

- **ファイル**: [`backend/app/core/services/optimization/bayesian_optimizer.py`](backend/app/core/services/optimization/bayesian_optimizer.py)
- **役割**: ML モデルのハイパーパラメータや GA のパラメータを、ベイズ最適化を用いて効率的に探索・調整します。
- **実装詳細**:
  - **`scikit-optimize`の利用**: `gp_minimize`（ガウス過程回帰に基づく最適化）を主に使用し、少ない試行回数で有望なパラメータ空間を探索します。
  - **汎用的なインターフェース**: `optimize_ga_parameters`と`optimize_ml_hyperparameters`の 2 つのメソッドを提供。目的関数 (`objective_function`) とパラメータ空間 (`parameter_space`) を受け取る設計になっており、様々な対象の最適化に再利用可能です。
  - **フォールバック機構**: `scikit-optimize`がインストールされていない環境でも動作するよう、`_optimize_with_fallback`としてランダムサーチによる代替実装が用意されています。
  - **目的関数のラップ**: `use_named_args`デコレータを使い、最適化ライブラリが要求するリスト形式の引数と、人間が分かりやすい辞書形式の引数をブリッジしています。

## 5. 品質保証とテスト

本 ML システムは、品質を保証するために`backend/tests/ml/`ディレクトリに集約された多岐にわたるテストによってカバーされています。

- **単体テスト**: 各コンポーネント（`FeatureEngineeringService`, `MLSignalGenerator`など）が個別に正しく機能すること。
- **統合テスト**: 各コンポーネントが連携し、ML 指標がオートストラテジーに正しく統合されること。
- **堅牢性テスト**: 不正なデータ入力やエッジケースに対するエラーハンドリング。
- **パフォーマンス測定**: モデルの学習と予測にかかる時間を測定し、ボトルネックを特定します。

これらのテストは、CI/CD パイプラインに組み込むことで、コード変更時の自動的な品質検証を可能にします。
