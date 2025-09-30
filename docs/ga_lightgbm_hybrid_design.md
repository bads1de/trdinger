# GA + ML ハイブリッドアプローチの設計書 (リビジョン 3: 複数学習機対応一般化)

## 概要

この設計書は、オートストラテジーの強化計画の 2 番目の案「GA + ML のハイブリッドアプローチ」を対象とします。serena ツールと提供された ml モジュール詳細 (BaseMLTrainer, MLTrainingService, ModelManager, ml_metadata.py, exceptions.py など) を基に、既存アセットを最大限活用した設計を提案します。目的は、GA の探索力と ML モデルの予測力を組み合わせ、よりロバストで強い取引戦略を生成することです。LightGBM 特化を避け、MLTrainingService の single_model_config で XGBoost, CatBoost, RandomForest などの複数モデルを選択・統合可能に一般化。

既存アセットの活用 (ML 詳細分析反映):

- **GA 側**: core/ga_engine.py の GeneticAlgorithmEngine (run_evolution で DEAP ループ, toolbox.evaluate で IndividualEvaluator 呼び出し) と EvolutionRunner。individual_evaluator.py の IndividualEvaluator (evaluate_individual でバックテストフィットネス, \_calculate_fitness で Sharpe/Max Drawdown/Win Rate/balance_score 計算, multi_objective 対応) を拡張。
- **ML 側**: ml/base_ml_trainer.py の BaseMLTrainer (train_model で特徴量生成/学習/評価統合,\_calculate_features で AutoMLFeatureGenerationService 委譲,\_time_series_cross_validate で検証, get_feature_importance で重要度取得)。ml/ml_training_service.py の MLTrainingService (train_model で最適化付き学習, trainer_type="single" で model_type 指定 (LightGBM/XGBoost/CatBoost/RandomForest など), \_train_with_optimization で Optuna 統合, generate_signals で予測信号生成)。ml/model_manager.py の ModelManager (save_model/load_model でバージョン管理/クリーンアップ, get_latest_model で最新モデル取得, _extract_algorithm_name でモデル抽出)。ml/ml_metadata.py の ModelMetadata (from_training_result でメタデータ構築, validate で妥当性チェック)。ml/exceptions.py の MLModelError/MLTrainingError でエラー処理。
- **特徴量/統合**: ml/feature_engineering/automl_feature_generation_service.py の AutoMLFeatureGenerationService (calculate_enhanced_features で tsfresh/autofeat, \_calculate_target_for_automl でラベル生成)。serializers/gene_serialization.py で StrategyGene シリアライズ。optimization/optuna_optimizer.py でハイパーパラメータ。複数モデル対応: MLTrainingService.get_available_single_models() で利用可能モデル一覧取得, hybrid_predictor で複数モデル平均予測 (e.g., ensemble_config["models"] = ["lightgbm", "xgboost"] で平均化)。
- **データフロー**: OHLCV (ohlcv_repository.py) を BaseMLTrainer.\_calculate_features で特徴量化 → MLModel (models/lightgbm.py/xgboost.py/catboost.py など, MLTrainingService.model_type で選択) で predict → IndividualEvaluator.\_calculate_fitness に prediction_score 注入 (fitness = sharpe \* prediction_weight + score, MLTrainingService.generate_signals で信号化, 複数モデル時は平均/投票で統合)。
- **TDD 対応**: pytest でテスト先行。既存テスト拡張 (test_individual_evaluator.py, ml/tests/test_single_model_trainer.py)。BaseMLTrainer の safe_ml_operation でエラー耐性確保。カバレッジ 95%目標。

分析洞察: BaseMLTrainer のテンプレートメソッド (_train_model_impl で ML fit/predict, model_type で LightGBM/XGBoost など動的) をハイブリッド評価にフック。MLTrainingService の_ create_objective_function で Optuna 目的関数に GA フィットネス統合可能, single_model_config でモデル選択。ModelManager の PerformanceMonitoringConfig で予測精度監視, _extract_algorithm_name で複数モデル対応。exceptions.py で MLTrainingError ハンドリング。新規コード最小 (アダプタ/拡張評価器), 既存の sklearn/LightGBM/XGBoost/CatBoost/Optuna/hmmlearn 活用率 95%。

この設計で、仮想通貨ボラティリティ対応、予測精度向上、過剰適合防止、複数モデル柔軟性を実現。

## システムアーキテクチャ

### 主要コンポーネント (ML 詳細反映)

1. **特徴量生成レイヤー** (既存: AutoMLFeatureGenerationService, BaseMLTrainer.\_calculate_features)

   - 入力: OHLCV/FR/OI (data_collection/bybit/)。
   - 出力: pd.DataFrame (tsfresh + technical_indicators/orchestrator.py,\_calculate_target_for_automl でラベル)。
   - 新規拡張: utils/hybrid_feature_adapter.py - StrategyGene (models/strategy_gene.py) を特徴に変換 (gene_utils.py + tsfresh_calculator.py, BaseMLTrainer.\_preprocess_data でスケーリング)。

2. **LightGBM 予測レイヤー** (既存: LightGBMModel, BaseMLTrainer.\_predict_single, MLTrainingService.generate_signals)

   - モデル: MLModel (models/lightgbm.py/xgboost.py/catboost.py/randomforest.py など, MLTrainingService.single_model_config["model_type"] で選択, BaseMLTrainer.\_train_single_model で fit, feature_importance="gain"/"weight" など)。
   - トレーニング: MLTrainingService.train_model (trainer_type="single", single_model_config={"model_type": "xgboost"}, BaseMLTrainer.train_model 委譲, \_time_series_cross_validate で CV, ml_metadata.ModelMetadata.from_training_result でメタデータ)。
   - 予測: BaseMLTrainer.predict (num_iteration=best_iteration, generate_signals で{"up"/"down"/"range"}確率, 複数モデル時は MLTrainingService で平均化)。

3. **GA 統合レイヤー** (既存: GeneticAlgorithmEngine, IndividualEvaluator)

   - 初期化: strategy_factory.py.generate_random_gene。
   - 評価: IndividualEvaluator を HybridIndividualEvaluator にサブクラス (evaluate_individual で backtest 後 MLTrainingService.predict 呼び出し,\_calculate_fitness 拡張: prediction_score = generate_signals()["up"] - generate_signals()["down"], fitness += config.prediction_weight \* score,\_calculate_long_short_balance 統合)。
   - 進化: ga_engine.py.setup_deap で toolbox.register("evaluate", hybrid_evaluate), genetic_operators.py で予測高い個体優先。

4. **オーケストレーション** (既存: auto_strategy_service.py, MLTrainingService, ModelManager)
   - フロー: GA 初期化 → BaseMLTrainer.\_calculate_features → LightGBM predict → ハイブリッド評価 → run_evolution ループ (ModelManager.load_model でモデルロード, save_model で保存)。
   - パラレル化: deap_setup.py.multiprocessing + MLTrainingService.\_train_with_optimization の background_task_manager.py。

### データフロー (Mermaid, ML 反映)

```mermaid
graph TD
    A[市場データ<br/>(OHLCV, FR, OI)] --> B[BaseMLTrainer._calculate_features<br/>(AutoMLFeatureGenerationService)]
    B --> C[StrategyGene生成<br/>(strategy_factory.py)]
    C --> D[Gene to 特徴量<br/>(hybrid_feature_adapter.py: from_list + tsfresh)]
    D --> E[ML予測<br/>(MLTrainingService.model_type選択, BaseMLTrainer.predict, generate_signals)]
    E --> F[ハイブリッド評価<br/>(HybridIndividualEvaluator: _calculate_fitness + prediction_score, 複数モデル平均)]
    F --> G[GA進化<br/>(GeneticAlgorithmEngine.run_evolution: toolbox)]
    G --> H[バックテスト<br/>(backtest_service.run_backtest)]
    H --> I[最適Gene出力<br/>(GeneSerializer.to_list, ModelMetadata.validate)]
    I --> J[ModelManager.save_model<br/>(metadata=ModelMetadata.to_dict, _extract_algorithm_name)]
```

## 実施ステップ (TDD, ML 反映)

TDD 厳守: テスト → 実装 → リファクタ。BaseMLTrainer.safe_ml_operation でエラー処理, ModelMetadata.validate でメタデータ検証。

### フェーズ 1: 準備/テスト (1 日)

1. [ ] **テスト**: tests/test_hybrid_feature_adapter.py - Gene → 特徴 (mock BaseMLTrainer.\_preprocess_data, assert df.shape, MLFeatureError ハンドル)。
   - 既存: test_strategy_factory.py。
2. [ ] **テスト**: tests/test_hybrid_predictor.py - LightGBM バッチ予測 (mock MLTrainingService.generate_signals, assert "up" > 0.5, MLPredictionError)。
   - 既存: ml/tests/test_single_model_trainer.py (BaseMLTrainer.\_predict_single)。
3. [ ] **テスト**: tests/test_hybrid_evaluator.py - fitness 計算 (mock backtest + predict, assert fitness > base, MLTrainingError)。

### フェーズ 2: 実装 (2 日)

1. [ ] **アダプタ**: utils/hybrid_feature_adapter.py - StrategyGene → tsfresh 入力 (BaseMLTrainer.\_calculate_features 委譲, automl_config=ml_config, MLTrainingService.single_model_config 対応)。
   - 既存: gene_utils.py + AutoMLFeatureGenerationService.calculate_enhanced_features。
2. [ ] **予測モジュール**: core/hybrid_predictor.py - MLTrainingService ラップ (trainer_type="single", model_type=config指定, load_model で ModelManager.load_model, generate_signals で確率, 複数モデル平均, \_time_series_cross_validate 統合)。
   - 既存: BaseMLTrainer.\_predict_single + optuna_optimizer.py。
3. [ ] **評価器**: core/hybrid_individual_evaluator.py - IndividualEvaluator 継承 (evaluate_individual で hybrid_predictor.predict, \_calculate_fitness に prediction 統合 (単一/複数モデル平均), balance_score 拡張)。
   - 既存: \_calculate_multi_objective_fitness + exceptions.MLModelError。
4. [ ] **GA 拡張**: ga_engine.py.setup_deap で toolbox.evaluate = hybrid_evaluate (IndividualEvaluator 置き換え, ModelMetadata.from_training_result でログ, MLTrainingService.get_available_single_models() でモデル選択)。

### フェーズ 3: 統合/最適化 (1 日)

1. [ ] **オーケストレーション**: auto_strategy_service.py に hybrid_mode (MLTrainingService.train_model (model_type=config), ModelManager.get_latest_model でロード)。
   - 既存: experiment_manager.py + ml_config。
2. [ ] **統合テスト**: tests/test_hybrid_ga.py - run_evolution with hybrid (assert Sharpe +20%, ModelMetadata.validate pass)。
   - 既存: test_backtest_orchestration.py + time_series_cv.py。
3. [ ] **最適化**: MLTrainingService.\_train_with_optimization 拡張 for hybrid params (prediction_weight, Optuna n_calls=50, single_model_config 統合)。
   - 既存: base_optimizer.py + PerformanceMonitoringConfig。
4. [ ] **フロント**: useAutoStrategy.ts に hybrid, GAConfigForm.tsx 更新 (API /ml/train with automl_config, model_type選択UI追加)。

### リスク/緩和 (ML 反映)

- **過剰適合**: BaseMLTrainer.\_time_series_cross_validate 必須 (walk-forward in evaluate_individual, MLValidationError, 複数モデルでロバスト性向上)。
- **計算コスト**: MLTrainingService.\_train_with_optimization の background + deap multiprocessing (<5min/population=100, 複数モデル時は並列化)。
- **互換性**: hybrid フラグ分岐 (従来 GA 保持, ModelManager.cleanup_expired_models で管理, model_type="lightgbm" で後方互換)。
- **エラー**: exceptions.MLTrainingError ハンドル (safe_ml_operation)。

## 期待効果

- 強度: generate_signals 統合 (複数モデル平均) で Sharpe +20% (IndividualEvaluator 向上)。
- 活用率: 95% (新規アダプタ/評価器, BaseMLTrainer/MLTrainingService (single_model_config)/ModelManager (_extract_algorithm_name) 活用)。
- スケーラビリティ: \_train_with_optimization 並列で時間半減, ModelMetadata で監視, 複数モデル動的選択。

このリビジョンで ML モジュール (BaseMLTrainer の学習フロー, MLTrainingService の最適化/single_model_config, ModelManager の管理/_extract_algorithm_name) を詳細反映、LightGBM特化を一般化。承認後、code モード TDD 開始。
