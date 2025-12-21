# MLパイプライン全体最適化（CASH）実装計画 (Revised)

## 1. 概要
特徴量エンジニアリング、特徴量選択、モデル学習のハイパーパラメータを同時最適化（CASH）する機能を、**本番コードベース（OptimizationService等）に直接統合**する。
検証スクリプト `verify_feature_reduction.py` をアップグレードし、単なる検証だけでなく、最適化タスクの実行・評価も行える統合ツールへと進化させる。

## 2. 実装方針：本番コードへの統合

### A. `FeatureEngineeringService` の拡張
計算コスト削減のため、Optuna探索用の「スーパーセット（全パラメータパターンの特徴量）」を生成する機能を追加する。

*   **`create_feature_superset(ohlcv_df, ...)`**: 
    *   `FracDiff` の `d` を {0.3, 0.4, 0.5} で計算した列などを全て含む巨大なDataFrameを生成する。
    *   各列名にパラメータ情報を埋め込む（例: `FracDiff_Price_d0.4`）。

### B. `OptimizationService` の拡張 (`backend/app/services/ml/optimization/optimization_service.py`)
モデルパラメータだけでなく、前処理（特徴量選択など）のパラメータも探索空間に含めるように `optimize_parameters` メソッドを拡張する。

*   **探索空間の統合**: `model_params` + `feature_selection_params` + `feature_engineering_params`
*   **Pipeline的な評価**: `objective` 関数内で `Selector` -> `Trainer` の順に実行し、スコアを算出。

### C. `FeatureSelector` の改修 (`backend/app/services/ml/feature_selection/feature_selector.py`)
Optunaから渡されたパラメータ（例: `threshold`, `method`）を受け取り、動的に振る舞いを変えられるようにする（現状でほぼ対応済みだが、インターフェースを確認）。

## 3. 検証・最適化ツールのアップグレード
`backend/scripts/verify_feature_reduction.py` を **`backend/scripts/run_ml_pipeline.py`** (仮称) にリネームし、以下の機能を実装する。

### コマンドライン引数
*   `--mode`: 
    *   `verify`: 既存の検証モード（固定パラメータで1回実行）。
    *   `optimize`: Optunaによる最適化モード。
*   `--n_trials`: 最適化の試行回数。
*   `--data_limit`: 使用するデータ行数。

### 最適化モード (`optimize`) の挙動
1.  **データ準備**: DBからデータを取得し、`FeatureEngineeringService.create_feature_superset` で全特徴量を生成。
2.  **データ分割**: `Train/Val` (Optuna用) と `Test` (最終評価用) に分割。
3.  **最適化実行**: `OptimizationService` を呼び出し、ベストパラメータを探索。
    *   試行ごとに、スーパーセットから必要なカラムを選択（例: `d=0.4` が選ばれたら `*_d0.4` の列を採用）。
4.  **最終評価**: ベストパラメータで `Test` データを評価し、ベースライン（デフォルト設定）と比較。
5.  **Config出力**: ベストパラメータを JSON/YAML 形式で出力（`unified_config.py` への反映用）。

## 4. 探索空間の詳細（再定義）

| カテゴリ | パラメータ | 探索範囲 | 実装方法 |
| :--- | :--- | :--- | :--- |
| **FE** | `frac_diff_d` | {0.3, 0.4, 0.5, 0.6} | スーパーセットから列選択 |
| **Selection** | `method` | {`staged`, `rfecv`, `mutual_info`} | `FeatureSelector` 引数 |
| **Selection** | `correlation_threshold` | 0.85 ~ 0.99 | `FeatureSelector` 引数 |
| **Selection** | `min_features` | 5 ~ 30 | `FeatureSelector` 引数 |
| **Model** | `learning_rate` | 0.005 ~ 0.1 (log) | `LightGBM` 引数 |
| **Model** | `num_leaves` | 16 ~ 128 | `LightGBM` 引数 |

## 5. 実行ステップ

1.  `FeatureEngineeringService` にスーパーセット生成メソッドを追加。
2.  `OptimizationService` のロジック拡張（特徴量選択パラメータへの対応）。
3.  `verify_feature_reduction.py` をアップグレード・リネーム。
4.  動作確認（少量のデータ・試行回数で）。