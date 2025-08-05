# ML トレーニング ワークフロー総覧

本ドキュメントは、本リポジトリにおける機械学習（ML）トレーニングのエンドツーエンドなワークフロー（データ収集/特徴量/学習/評価/オーケストレーション/モデル管理/本番運用）を、役割・責務・API 連携・再現性の観点で体系化したものです。開発・運用の標準リファレンスとして利用してください。

関連ソースの起点（代表例）

- API（学習開始/状態/停止）: [`backend/app/api/ml_training.py.start_ml_training()`](backend/app/api/ml_training.py:177), [`backend/app/api/ml_training.py.get_ml_training_status()`](backend/app/api/ml_training.py:231), [`backend/app/api/ml_training.py.stop_ml_training()`](backend/app/api/ml_training.py:256)
- モデル/設定管理 API: [`backend/app/api/ml_management.py.get_models()`](backend/app/api/ml_management.py:29), [`backend/app/api/ml_management.py.update_ml_config()`](backend/app/api/ml_management.py:194), [`backend/app/api/ml_management.py.reset_ml_config()`](backend/app/api/ml_management.py:227)
- AutoML 特徴量 API: [`backend/app/api/automl_features.py.generate_features()`](backend/app/api/automl_features.py:94), [`backend/app/api/automl_features.py.validate_config()`](backend/app/api/automl_features.py:151)
- 設定バリデーション: [`backend/app/config/validators.py.MLConfigValidator()`](backend/app/config/validators.py:97)
- 評価/アンサンブルのリファレンステスト: [`backend/tests/test_robust_multifaceted_evaluation.py.RobustMultifacetedEvaluation()`](backend/tests/test_robust_multifaceted_evaluation.py:30), [`backend/tests/test_advanced_accuracy_improvement.py.AdvancedAccuracyImprovementTest()`](backend/tests/test_advanced_accuracy_improvement.py:25)
- オーケストレーション（参照）: [`backend/tests/test_ml_orchestrator_dynamic_params.py.MLOrchestrator()`](backend/tests/test_ml_orchestrator_dynamic_params.py:13)

注意: orchestration/feature_engineering/models/evaluation の実装クラスは本リポジトリ内に複数配置されていますが、一部はテスト経由で参照されるため、詳細は各ディレクトリ配下を参照してください。

---

## 1. 全体像（ハイレベルアーキテクチャ）

1. データ層

- 市場データ（OHLCV）、オープンインタレスト、ファンディングレートなどをデータベースから取得し、バックテスト/特徴量生成に供給

1. 特徴量層

- AutoML 特徴量（TSFresh/AutoFeat 等の設定を抽象化）
- テクニカル指標、統計特徴、シリーズ変換など
- 設定バリデーションで上限や実行時間の制御

1. 学習層

- 単一モデル（LightGBM/XGBoost/RandomForest など）
- アンサンブル（バギング/スタッキング）
- ハイパラ最適化（Optuna 想定）
- 乱数固定/タイムスプリット等の再現性担保

1. 評価層

- 時系列交差検証、クラス不均衡ロバストネス、特徴量重要度安定性
- 取引シミュレーション指標（総リターン、シャープ、最大 DD 等）

1. オーケストレーション層

- 学習ジョブの開始/状態監視/停止
- モデル・設定管理
- 背景タスク・非同期実行・ログ集約

1. 提供/運用層

- 推論 API、現在のモデルロード/切替、モデルファイルのライフサイクル管理
- 設定の取得/更新/リセット、クリーンアップ

---

## 2. データパイプライン

役割

- シンボル/タイムフレームの正規化、上限行数やタイムアウトの制御
- 特徴量生成のためのデータ統合（OHLCV + OI + FR）

主要ポイント

- シンボル/タイムフレームの正規化:
  - [`backend/app/config/validators.py.MarketDataValidator.normalize_symbol()`](backend/app/config/validators.py:61)
- データ処理設定のバリデーション:
  - [`backend/app/config/validators.py.MLConfigValidator.validate_data_processing_config()`](backend/app/config/validators.py:165)
- 拡張データの統合はバックテスト用サービスに集約される想定（参照: 管理 API での依存関係）
  - [`backend/app/api/ml_management.py.get_data_service()`](backend/app/api/ml_management.py:275)

テストから読み解く拡張データ例

- FR/OI の統合や時間枠連携を通した強化データの作成:
  - [`backend/tests/test_ml_orchestrator_dynamic_params.py.test_get_enhanced_data_with_fr_oi_integration()`](backend/tests/test_ml_orchestrator_dynamic_params.py:180)

---

## 3. 特徴量エンジニアリング

3.1 AutoML 特徴量 API

- 生成: [`backend/app/api/automl_features.py.generate_features()`](backend/app/api/automl_features.py:94)
- 設定検証: [`backend/app/api/automl_features.py.validate_config()`](backend/app/api/automl_features.py:151)
- デフォルト設定取得: [`backend/app/api/automl_features.py.get_default_config()`](backend/app/api/automl_features.py:208)
- キャッシュクリア: [`backend/app/api/automl_features.py.clear_cache()`](backend/app/api/automl_features.py:226)

  3.2 AutoML 設定モデル

- TSFresh/AutoFeat 等の上位設定を Pydantic で受ける（詳細は API モデル参照）

  - [`backend/app/api/automl_features.py.AutoMLConfigModel()`](backend/app/api/automl_features.py:57)

  3.3 高度特徴量（テストの参照実装）

- 例: 高度特徴量生成/クリーニング/選択の流れ
  - [`backend/tests/test_advanced_accuracy_improvement.py.test_advanced_features_performance()`](backend/tests/test_advanced_accuracy_improvement.py:163)
  - [`backend/tests/test_robust_multifaceted_evaluation.py.RobustMultifacetedEvaluation.create_diverse_test_scenarios()`](backend/tests/test_robust_multifaceted_evaluation.py:47)

---

## 4. 学習（トレーニング）

4.1 エンドポイント

- 開始: [`backend/app/api/ml_training.py.start_ml_training()`](backend/app/api/ml_training.py:177)
- 状態取得: [`backend/app/api/ml_training.py.get_ml_training_status()`](backend/app/api/ml_training.py:231)
- 停止: [`backend/app/api/ml_training.py.stop_ml_training()`](backend/app/api/ml_training.py:256)

  4.2 トレーニング設定（Pydantic）

- 主要構成体

  - 最適化設定: [`backend/app/api/ml_training.py.OptimizationSettingsConfig()`](backend/app/api/ml_training.py:35)
  - アンサンブル設定: [`backend/app/api/ml_training.py.EnsembleConfig()`](backend/app/api/ml_training.py:75)
    - バギング: [`backend/app/api/ml_training.py.BaggingParamsConfig()`](backend/app/api/ml_training.py:46)
    - スタッキング: [`backend/app/api/ml_training.py.StackingParamsConfig()`](backend/app/api/ml_training.py:64)
  - 単一モデル設定: [`backend/app/api/ml_training.py.SingleModelConfig()`](backend/app/api/ml_training.py:90)
  - トレーニング本体設定: [`backend/app/api/ml_training.py.MLTrainingConfig()`](backend/app/api/ml_training.py:102)

  4.3 アンサンブル/単一モデル

- デフォルトはアンサンブル前提（バギング/スタッキング）
- 単一モデルのみの学習もフラグで可能

  4.4 再現性とベストプラクティス

- 乱数固定（random_state）
- 時系列に配慮した分割（リーク回避）
- 早期停止、適切な評価指標の利用
- データ/特徴量のスナップショット保存を推奨

---

## 5. 評価・検証（ロバスト性重視）

5.1 時系列安定性

- TimeSeriesSplit による安定性評価（平均/分散/最小/最大/CV）

  - [`backend/tests/test_robust_multifaceted_evaluation.py.RobustMultifacetedEvaluation.evaluate_time_series_stability()`](backend/tests/test_robust_multifaceted_evaluation.py:157)

  5.2 クラス不均衡ロバストネス

- balanced accuracy、macro 指標、混同行列、確率予測を含む多角評価

  - [`backend/tests/test_robust_multifaceted_evaluation.py.evaluate_class_imbalance_robustness()`](backend/tests/test_robust_multifaceted_evaluation.py:231)

  5.3 特徴量重要度の安定性

- ブートストラップ × 統計集約

  - [`backend/tests/test_robust_multifaceted_evaluation.py.evaluate_feature_importance_stability()`](backend/tests/test_robust_multifaceted_evaluation.py:300)

  5.4 取引パフォーマンス・シミュレーション

- 総リターン/シャープ/最大 DD/勝率/トレード数

  - [`backend/tests/test_robust_multifaceted_evaluation.py.simulate_trading_performance()`](backend/tests/test_robust_multifaceted_evaluation.py:349)

  5.5 段階的改善の検証（再現例）

- ベースライン → 高度特徴量 → アンサンブルの増分改善を可視化
  - [`backend/tests/test_advanced_accuracy_improvement.py.run_comprehensive_accuracy_test()`](backend/tests/test_advanced_accuracy_improvement.py:273)

---

## 6. オーケストレーション

6.1 Orchestration Service 呼び出し

- 学習開始/状態/停止は Orchestration Service 経由

  - [`backend/app/api/ml_training.py.MLTrainingOrchestrationService()`](backend/app/api/ml_training.py:15)

  6.2 ダイナミックパラメータ推定（参照）

- OHLCV のメタ/価格レンジ/列名/インデックス間隔からシンボル・時間軸を推定

  - [`backend/tests/test_ml_orchestrator_dynamic_params.py.TestMLOrchestratorDynamicParams`](backend/tests/test_ml_orchestrator_dynamic_params.py:16)

  6.3 背景タスクと安全実行

- FastAPI BackgroundTasks と共に、統一エラーハンドラでラップ

  - [`backend/app/api/ml_training.py.UnifiedErrorHandler.safe_execute_async()`](backend/app/api/ml_training.py:228)
  - [`backend/app/api/automl_features.py.UnifiedErrorHandler.safe_execute_async()`](backend/app/api/automl_features.py:146)

  6.4 ロギング

- エントリ時に受信設定を詳細ログ
  - [`backend/app/api/ml_training.py.start_ml_training()`](backend/app/api/ml_training.py:197)

---

## 7. モデル/設定 管理

7.1 モデル一覧/削除/読み込み/現在モデル/古いモデル掃除

- 一覧: [`backend/app/api/ml_management.py.get_models()`](backend/app/api/ml_management.py:29)
- 削除: [`backend/app/api/ml_management.py.delete_model()`](backend/app/api/ml_management.py:51)
- 読み込み: [`backend/app/api/ml_management.py.load_model()`](backend/app/api/ml_management.py:115)
- 現在モデル: [`backend/app/api/ml_management.py.get_current_model()`](backend/app/api/ml_management.py:135)
- クリーンアップ: [`backend/app/api/ml_management.py.cleanup_old_models()`](backend/app/api/ml_management.py:256)

  7.2 設定（取得/更新/リセット）

- 取得: [`backend/app/api/ml_management.py.get_ml_config()`](backend/app/api/ml_management.py:172)
- 更新: [`backend/app/api/ml_management.py.update_ml_config()`](backend/app/api/ml_management.py:194)
- リセット: [`backend/app/api/ml_management.py.reset_ml_config()`](backend/app/api/ml_management.py:227)
- 設定値の妥当性検証: [`backend/app/config/validators.py.MLConfigValidator()`](backend/app/config/validators.py:97)

  7.3 API レスポンス一貫化/エラー処理

- APIResponseHelper/UnifiedErrorHandler で標準化
  - 参照: 管理/学習/特徴量 API 内の呼び出し

---

## 8. 本番運用

8.1 推論・提供

- 現在モデルのロード/切替 API で運用

  - [`backend/app/api/ml_management.py.load_model()`](backend/app/api/ml_management.py:115)
  - [`backend/app/api/ml_management.py.get_current_model()`](backend/app/api/ml_management.py:135)

  8.2 監視/異常検知

- モデル状態 API での可視化・メトリクス集約
  - [`backend/app/api/ml_management.py.get_ml_status()`](backend/app/api/ml_management.py:72)
- ドリフト兆候: 分布変化、指標低下の監視を推奨

  8.3 再学習トリガ

- スケジュール/イベント（精度低下、データドリフト、期間経過など）を条件に再学習フローを起動（学習 API）

---

## 9. トラブルシューティング

- 学習が進まない/停止しない

  - OrchestrationService の状態 API で確認: [`backend/app/api/ml_training.py.get_ml_training_status()`](backend/app/api/ml_training.py:231)
  - 安全停止: [`backend/app/api/ml_training.py.stop_ml_training()`](backend/app/api/ml_training.py:256)

- 精度が出ない

  - データリーク防止（時系列分割/先見情報除去）
  - 特徴量過多による過学習 → 重要度で選択/正則化/ドロップ
  - クラス不均衡 → クラス重み/閾値最適化/再サンプリング

- 処理時間が長い/タイムアウト

  - AutoML/特徴量の上限・タイムアウトを設定
  - サンプリング/期間短縮/パラレル設定見直し

- モデルファイル肥大化/過多
  - クリーンアップ API の定期実行: [`backend/app/api/ml_management.py.cleanup_old_models()`](backend/app/api/ml_management.py:256)

---

## 10. セキュリティ/再現性/標準化

- 再現性: 乱数シード固定、データ/特徴量のスナップショット保存
- バリデーション: 入力/設定/範囲チェックを統一（Validators）
  - [`backend/app/config/validators.py.MLConfigValidator.validate_model_config()`](backend/app/config/validators.py:215)
- ログ/監査: API 入口で config を記録（機密情報は配慮）
- 命名規約/バージョニング: モデル名に日時/ハッシュ/設定要約を付与推奨

---

## 付録 A: 主要エンドポイント一覧（抜粋）

- 学習

  - POST /api/ml-training/train → [`backend/app/api/ml_training.py.start_ml_training()`](backend/app/api/ml_training.py:177)
  - GET /api/ml-training/training/status → [`backend/app/api/ml_training.py.get_ml_training_status()`](backend/app/api/ml_training.py:231)
  - POST /api/ml-training/stop → [`backend/app/api/ml_training.py.stop_ml_training()`](backend/app/api/ml_training.py:256)

- 特徴量（AutoML）

  - POST /api/automl-features/generate → [`backend/app/api/automl_features.py.generate_features()`](backend/app/api/automl_features.py:94)
  - POST /api/automl-features/validate-config → [`backend/app/api/automl_features.py.validate_config()`](backend/app/api/automl_features.py:151)
  - GET /api/automl-features/default-config → [`backend/app/api/automl_features.py.get_default_config()`](backend/app/api/automl_features.py:208)
  - POST /api/automl-features/clear-cache → [`backend/app/api/automl_features.py.clear_cache()`](backend/app/api/automl_features.py:226)

- 管理
  - GET /api/ml/models → [`backend/app/api/ml_management.py.get_models()`](backend/app/api/ml_management.py:29)
  - DELETE /api/ml/models/{model_id} → [`backend/app/api/ml_management.py.delete_model()`](backend/app/api/ml_management.py:51)
  - POST /api/ml/models/{model_name}/load → [`backend/app/api/ml_management.py.load_model()`](backend/app/api/ml_management.py:115)
  - GET /api/ml/models/current → [`backend/app/api/ml_management.py.get_current_model()`](backend/app/api/ml_management.py:135)
  - GET /api/ml/status → [`backend/app/api/ml_management.py.get_ml_status()`](backend/app/api/ml_management.py:72)
  - GET /api/ml/config → [`backend/app/api/ml_management.py.get_ml_config()`](backend/app/api/ml_management.py:172)
  - PUT /api/ml/config → [`backend/app/api/ml_management.py.update_ml_config()`](backend/app/api/ml_management.py:194)
  - POST /api/ml/config/reset → [`backend/app/api/ml_management.py.reset_ml_config()`](backend/app/api/ml_management.py:227)
  - POST /api/ml/models/cleanup → [`backend/app/api/ml_management.py.cleanup_old_models()`](backend/app/api/ml_management.py:256)

---

## 付録 B: 実行チェックリスト

- 前提

  - [ ] データ取得設定（シンボル/時間軸/期間）の妥当性（Validators）
  - [ ] AutoML 特徴量設定の検証済み（validate-config）
  - [ ] データ/特徴量の上限・タイムアウト設定が適切

- 事前検証

  - [ ] 特徴量生成（必要ならサンプルで）と統計確認
  - [ ] 時系列分割設計（リーク回避）

- 学習

  - [ ] 単一/アンサンブルの方針選定
  - [ ] ハイパラ最適化の有無と試行回数
  - [ ] 乱数シード固定
  - [ ] 早期停止/評価指標の合意

- 評価

  - [ ] 時系列 CV の平均/分散/最小/最大
  - [ ] クラス不均衡ロバストネス指標
  - [ ] 重要度安定性
  - [ ] 取引パフォーマンス（Sharpe/MaxDD/勝率）

- 運用
  - [ ] モデル保存/命名規約/メタデータ格納
  - [ ] 現行モデルのロード/切替
  - [ ] 監視メトリクスと閾値
  - [ ] 古いモデルのクリーンアップ計画

---

## 付録 C: 参考テストの読み方

- 包括評価の進め方

  - 多様シナリオ生成 → 高度特徴量 → 特徴量選択 → 学習 → 各種ロバスト評価 → 取引パフォーマンス
  - 参照: [`backend/tests/test_robust_multifaceted_evaluation.py.run_comprehensive_evaluation()`](backend/tests/test_robust_multifaceted_evaluation.py:449)

- 段階的改善（ベンチマーク）
  - ベースライン → 高度特徴量 → アンサンブルの比較
  - 参照: [`backend/tests/test_advanced_accuracy_improvement.py._analyze_comprehensive_results()`](backend/tests/test_advanced_accuracy_improvement.py:303)

---

## まとめ

本ワークフローは「データの妥当性 → 特徴量の体系化 → 再現可能な学習 → ロバストな評価 → API 駆動のオーケストレーション → 安全な運用管理」という流れで統合されています。上記の API/サービス/バリデーション/テストの役割分担に従うことで、拡張しやすく再現性の高い ML 基盤を維持できます。
