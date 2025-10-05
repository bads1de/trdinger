# オートストラテジー強化計画

## 現状整理

- `GeneticAlgorithmEngine` は DEAP を用いた単一/多目的最適化、フィットネス共有、ハイブリッド GA+ML 評価 (`HybridIndividualEvaluator`) をサポート。
- ランダム遺伝子生成は `RandomGeneGenerator` を中心にスマート条件生成とポジションサイジング遺伝子を組み合わせ、TPSL を標準有効化。
- レジーム検知 (`RegimeDetector`) により HMM ベースの 3 状態分類を提供するが、活用は任意設定に留まる。
- バックテストは `BacktestService` に依存し、評価メトリクスは損益重視でリスク指標の重み付けは限定的。

## 課題と機会

1. **評価指標の多様化不足**: 現行の多目的最適化は目的関数の定義が限定され、ドローダウン・取引頻度などリスク重視指標の組み込みが弱い。
2. **データ表現と学習信号の拡張余地**: シンプルな OHLCV と閾値生成が中心で、イベントドリブンラベリングや市場センチメント情報が不足。
3. **ハイブリッド ML の汎用性**: 既存のハイブリッド評価は推定スコアの線形加算に留まり、特徴量変換やモデル選択の自動化が不十分。
4. **レジーム適応の実利用**: レジーム検知結果をフィットネスや遺伝子生成に反映する仕組みが薄く、活用度が低い。
5. **説明性と検証性**: GA の探索結果に対する解釈や、戦略の汎化性能検証手法が体系化されていない。

## 強化提案

### 1. リスク重視の多目的最適化強化

- **施策**: `EvolutionRunner` と GA 設定にカスタム目的関数を追加し、`max_drawdown`、`ulcer_index`、取引頻度ペナルティなどを組み込む。コラボレーティブ多目的進化 (CMEA) に倣い、世代ごとに目的空間を動的再重み付けする仕組みを検討。[^cmea]
- **実装ポイント**: `GAConfig` に目的関数テンプレートと重み設定を追加、`HybridIndividualEvaluator._calculate_multi_objective_fitness` に新指標を反映。リスク関連メトリクス算出はバックテスト結果整形層 (`app/services/auto_strategy/core/metrics`) を新設して集約。
- **検証**: 目的関数ごとに pytest ベースのフィットネス変換テストを作成し、`tests/test_ga_engine.py` を拡張して Pareto フロントが期待通り更新されることを確認。

### 2. イベントドリブンラベリングとデータ拡張

- **施策**: 2024 年のトリプルバリアラベリング研究[^triple_barrier]を参考に、`BacktestService`/データサービス層へ HRHP・LRLP ラベル生成を追加し、学習データの質を向上。レジーム毎に閾値を再定義して変動性適応を図る。
- **実装ポイント**: `RandomGeneGenerator` のスマートコンテキストにレジーム別閾値プロファイルを注入、`HybridFeatureAdapter` でラベル別特徴量 (OI、ファンディング率、センチメント API) を生成できるよう拡張。
- **検証**: ラベル生成の単体テスト、レジーム別データサンプリングの統計検証 (クラスバランス) を pytest で導入。

### 3. ハイブリッド ML/DRL 統合の高度化

- **施策**: Wavelet ベースの特徴抽出と DRL モデル活用事例[^wavelet_drl]を元に、`HybridPredictor` をモジュール化して PPO/A2C/DQN 等の DRL 政策をプラガブルに接続。遺伝的アルゴリズムで DRL ハイパーパラメータを共同最適化する GA+RL フレームを検討。
- **実装ポイント**: `hybrid_automl_config` にモデル種別・ウェーブレット種別を設定可能にし、`HybridIndividualEvaluator` で特徴量生成時にウェーブレット変換を適用。DRL 評価は async 実行で GA 世代とバッチ連携。
- **検証**: DRL 連携部分は統合テストが困難なため、モック済みポリシーでフィットネス重畳が安定することを pytest で検証し、実戦テストは専用バックテストシナリオで評価。

### 4. レジーム適応と遺伝子生成の連携強化

- **施策**: レジーム検知結果を `RandomGeneGenerator` の条件生成と突然変異率に反映する「レジームガイダンス」機構を導入。レジームごとの指標プリセット (トレンド/レンジ向け) を定義し、遺伝子生成時に重み付け。
- **実装ポイント**: `RegimeDetector` 出力をキャッシュし、`GAConfig` にレジーム別設定 (指標セット、TPSL プリセット) を追加。`create_deap_mutate_wrapper` にレジーム依存パラメータを渡して変異強度を動的調整。
- **検証**: レジーム別の戦略多様性を計測するメトリクス (ユニーク条件数、指標分布) を追加し、CI 上で閾値監視。

### 5. 説明性・検証性の向上

- **施策**: SHAP などの XAI 手法[^shap]を `HybridFeatureAdapter` に組み込み、ハイブリッド評価時に重要特徴を記録。進化ログ (`logbook`) へ指標別寄与度や探索経路を追跡できるメタデータを保存。
- **実装ポイント**: GA 実行結果 (`run_evolution` 戻り値) に説明情報を添付し、フロントエンドで戦略比較ビューを提供。`tests/test_ga_engine.py` に説明値が存在することを検証するスナップショットテストを追加。

### 6. パイプラインと開発フロー

- `BacktestService` に結果キャッシュとリトライ制御を追加し、GA 世代ごとの I/O レイテンシを削減。
- TDD を徹底するため、各提案機能ごとにユースケース駆動の pytest ケース (成功/失敗パス) を先行で用意し、遺伝子生成・評価ロジックの回帰を防止。
- CI で `mypy`, `flake8`, `pytest` を並列実行し、GA 構成変更時の静的解析と動作保証を自動化。

## 参考文献

- [^triple_barrier]: Enhanced Genetic-Algorithm-Driven Triple Barrier Labeling Method and Machine Learning Approach for Pair Trading Strategy in Cryptocurrency Markets (2024)
- [^wavelet_drl]: Enhancing algorithmic trading with wavelet-based deep reinforcement learning (2025)
- [^cmea]: Collaborative Multiobjective Evolutionary Algorithms in search of better Pareto Fronts. An application to trading systems (2022)
- [^shap]: SHAP: Explain Any Machine Learning Model in Python (2024)
