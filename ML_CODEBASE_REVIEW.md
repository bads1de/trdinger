# ML Codebase Review Report

## 概要

`backend/app/services/ml` 以下のコードベースをレビューしました。
設計パターン（Service 層、Trainer、Strategy など）に従って構造化されていますが、コードベースの拡大に伴い、いくつかのパフォーマンス上のボトルネック、非効率な実装、および潜在的なバグが確認されました。

特に `StackingEnsemble` における計算コストの重複は、学習時間を不必要に増大させているため、優先的に対処する必要があります。

## 🚨 クリティカルな問題 (Critical Issues)

### 1. スタッキング学習の計算重複 (`StackingEnsemble.py`)

**問題**:
`StackingEnsemble.fit` メソッド内で以下の処理が行われています。

1. 自前のループで `cross_val_predict` を実行し、OOF (Out-of-Fold) 予測を生成（メタラベリング用）。
2. `sklearn.ensemble.StackingClassifier.fit` を呼び出し。

`StackingClassifier` はデフォルトで内部的に Cross-Validation を行い、ベースモデルの OOF 予測を生成してメタモデルを学習します。その結果、**ベースモデルの学習・予測プロセスが 2 回繰り返されており、学習時間が約 2 倍になっています。**

**推奨される修正**:
`sklearn.ensemble.StackingClassifier` の使用をやめ、自前実装（現在のステップ 1）で生成した OOF 予測を使用してメタモデル（`logistic_regression`など）を直接学習させる実装に切り替えることを推奨します。これにより、計算コストを半減させつつ、OOF 予測をメタラベリングに活用する柔軟性を維持できます。

### 2. ラベルの型不整合リスク (`presets.py` vs `MetaLabeling`)

**問題**:
`presets.py` で定義されたラベル生成関数はデフォルトで文字列ラベル (`"UP"`, `"DOWN"`, `"RANGE"`) を返します。一方、`MetaLabelingService` や `StackingEnsemble` は数値（バイナリ）ラベル (`0`, `1`) を期待しています。
現在は `LabelCache` 内で変換処理が入っていますが、将来的に `presets` を直接利用してパイプラインを構築する際、型エラーや予期せぬ挙動（文字列が数値として解釈されずエラー、あるいは SILENT FAILURE）が発生するリスクがあります。

**推奨される修正**:
`presets.py` の各関数の戻り値を統一するか、明確な型定義とバリデーションを追加することを推奨します。

## ⚡ パフォーマンス改善 (Performance Improvements)

### 1. DataFrame 結合の非効率性 (`FeatureEngineeringService.py`)

**問題**:
`calculate_advanced_features` メソッド内で、`pd.concat` が複数回（5 回以上）連続して呼び出されています。Pandas の `concat` は都度メモリコピーが発生するため、大きなデータセットではパフォーマンスの大きなボトルネックとなります。

**推奨される修正**:
各計算機クラスから返される DataFrame をリストに格納し、最後に一度だけ `pd.concat` を実行するように書き換えてください。

### 2. Trend Scanning の低速なループ処理 (`trend_scanning.py`)

**問題**:
`TrendScanning.get_labels` メソッド内で、Python のネイティブ `for` ループを使用して回帰分析を行っています。O(N^2) の計算量となり、長期間のデータや多数のパラメータ探索を行う際に極めて低速になります。

**推奨される修正**:
`numba` を使用して計算ロジックを JIT コンパイルするか、`numpy` のストライドトリックを使用してベクトル化することを推奨します。これにより、数十倍〜数百倍の高速化が見込めます。

## 🛠️ その他の改善提案 (Other Suggestions)

### 1. 欠損値処理の洗練 (`FeatureEngineeringService.py`)

**問題**:
現在、全ての特徴量に対して一律で `fillna(0.0)` が適用されています。価格データや特定のインジケーター（例: RSI の 50）において、`0` は極端な値や異常値を意味する場合があり、モデルの学習を阻害する可能性があります。

**推奨される修正**:
特徴量の性質に応じて、以下の欠損値処理を使い分けることを検討してください。

- **前方補完 (`ffill`)**: 価格データなど
- **中央値/平均値埋め**: オシレーター系
- **定数埋め (-1, -999)**: Tree 系モデルが欠損として扱いやすい値

### 2. リソースクリーンアップ (`StackingEnsemble.py`)

**問題**:
`StackingEnsemble` クラスに明示的なリソース解放（`cleanup`）メソッドがありません。大量のメモリを消費する ML モデルオブジェクト（特に多数のベースモデル）がメモリに残存し、長時間稼働するプロセスでメモリ圧迫の原因となる可能性があります。

**推奨される修正**:
`BaseMLTrainer` のクリーンアップメカニズムに準拠し、`StackingEnsemble` にも `cleanup` メソッドを実装して、不要になったモデルやデータを明示的に `None` に設定してください。

### 3. モジュール構造の整理

**問題**:
`optimization_service.py` が `backend/app/services/optimization` に配置されていますが、これは ML サービス (`backend/app/services/ml`) の一部である機能（ハイパーパラメータ最適化）を提供しています。

**推奨される修正**:
将来的には `backend/app/services/ml/optimization` に移動し、ML モジュール内で完結させることで凝集度を高めることを検討してください。

## 🔄 Round 2 Review: 追加の発見事項 (2025/12/07)

`backend/app/services/ml/feature_selection`, `orchestration`, `model_manager` などを中心に追加のレビューを行いました。以下のパフォーマンスとスケーラビリティに関する重要な問題が特定されました。

### 🚨 クリティカルな問題 (Critical Issues - Round 2)

### 3. `FeatureSelector` におけるモデル適合の重複

**問題**:
`FeatureSelector` クラスの `_lasso_selection`, `_random_forest_selection` などのメソッドにおいて、モデルの適合が重複して実行されています。
例えば `_lasso_selection` では以下のようになっています：

```python
lasso.fit(X, y)  # 1回目のfit
selector = SelectFromModel(lasso, threshold=...)
selector.fit(X, y)  # 2回目のfit (prefit=Trueが指定されていないため)
```

`SelectFromModel` は `prefit=True` が指定されない場合、内部で estimator をクローンして再度 `fit` を実行します。これにより、特徴量選択の計算コストが意図せず **2 倍** になっています。

**推奨される修正**:

- `fit` したモデルを `SelectFromModel` に渡す場合は、必ず `prefit=True` を指定する。
- または、単に `SelectFromModel` だけを使用し、事前の明示的な `fit` を削除する。

### 4. モデル一覧取得時のスケーラビリティ問題 (`ModelManager`)

**問題**:
`MLManagementOrchestrationService.get_formatted_models` は、全モデルに対して `model_manager.load_model` を呼び出します。現在の実装では `joblib.load` を使用して **モデルファイル全体（ピクル）をメモリにロード** します。
モデルサイズが大きくなる（数百 MB〜数 GB）と、モデル一覧を表示するだけでサーバーのメモリを食いつぶし、応答時間が極端に長くなる恐れがあります。

**推奨される修正**:

- メタデータをモデル本体とは別の軽量なファイル（JSON など）として保存し、一覧取得時はそのサイドカーファイルのみを読み込むように変更する。
- 既存の `.joblib` ファイルからメタデータのみを読み込むことは困難なため、保存ロジックの変更が必要です。

## ⚡ パフォーマンス改善 (Performance Improvements - Round 2)

### 3. `MetricsCalculator` の計算オーバーヘッド

**問題**:
`MetricsCalculator.calculate_comprehensive_metrics` は、呼び出されるたびに ROC-AUC, PR-AUC, 混同行列, 分類レポートなど、重い計算を全て実行しています。学習中のバリデーションステップなどで頻繁に呼び出されると、全体の処理速度を低下させる要因になります。

**推奨される修正**:

- 計算するメトリクスのレベルを指定できるようにする（例: `level="basic"` で Accuracy/F1 のみ、`level="full"` で全指標）。
- または、ROC-AUC や PR-AUC などの重い計算を必要な場合のみ実行するフラグを追加する。

### 4. デフォルト特徴量選択手法の計算コスト (`MUTUAL_INFO`)

**問題**:
`FeatureSelector` のデフォルト設定で `SelectionMethod.MUTUAL_INFO` が含まれています。相互情報量の計算（k-近傍法に基づく）は、サンプル数が多い時系列データに対して非常に計算コストが高いです。

**推奨される修正**:

- デフォルトの `ensemble_methods` から `MUTUAL_INFO` を外すか、サンプルリングを行ってから計算するように変更する。
- より高速な `f_classif` (ANOVA) などを優先的に使用する設定にする。

## ✅ Action Plan (Updated)

1. ~~**[High Priority]** `StackingEnsemble.py` の重複計算を解消するリファクタリング。~~ ✅ **完了 (2025/12/06)**
   - sklearn `StackingClassifier` への依存を削除
   - 自前実装: OOF 予測生成 → メタモデル直接学習 → ベースモデル最終 fit
   - 計算コスト約半分に削減
   - `cleanup()` メソッドも追加
2. ~~**[High Priority]** `FeatureSelector` の重複 `fit` 修正。~~ ✅ **完了 (2025/12/06)**
   - `_lasso_selection` と `_random_forest_selection` で `SelectFromModel` に `prefit=True` を追加
   - 重複 `fit` を削除し、計算コストを半減
   - ユニットテスト `test_feature_selector.py` を追加
3. ~~**[High Priority]** `ModelManager` のメタデータ管理方法の改善（サイドカー JSON の導入）。~~ ✅ **完了 (2025/12/06)**
   - `save_model` 時にサイドカー JSON ファイル（`.meta.json`）を自動生成
   - `load_metadata_only` メソッドを追加（モデル本体をロードせずメタデータのみ取得）
   - サイドカー JSON がない古いモデルは `joblib` へフォールバック
   - `orchestration_utils.load_model_metadata_safely` を更新してサイドカー優先読み込み
   - ユニットテスト `test_model_manager.py` を追加
4. **[Medium Priority]** `MetricsCalculator` に軽量モード ("lite") を導入。(Round 2)
5. **[Medium Priority]** `FeatureEngineeringService.py` の `pd.concat` 回数を削減。（継続）
6. **[Medium Priority]** `TrendScanning` の高速化（`numba` 導入検討）。（継続）
7. ~~**[Low Priority]** `Optimization` ディレクトリの移動検討。~~ ✅ **完了 (2025/12/06)**
   - `backend/app/services/optimization` を `backend/app/services/ml/optimization` に移動
   - 関連するインポートパスを更新
   - テストパスを確認
