# Auto Strategy 強化実装計画（改訂版）

## 目的

ロバストで実践投入可能な多様的な戦略の構築を最大目的とする。**NSGA-IIとマルチフィットネス関数の基盤は既に実装済み**であるため、これを基盤にレジーム検出とRL統合を追加して多様性強化。TDD（Test-Driven Development）で開発し、pytest で各コンポーネントを検証。最終的に、異なる市場環境で安定した戦略ポートフォリオを生成可能にする。

追加ウェブ検索（Tavily/Exa 経由）で確認: DEAP NSGA-II のトレーディング例（multi-obj for portfolio, Sharpe/Risk）、HMM レジーム検出の regime-adaptive ML（QuantInsti ブログ: HMM ラベルで RF 訓練）、HMM+RL ハイブリッド（PPO-HMM for dynamic trading, IDS2025 論文: regime-aware PPO for risk control）。これを基に計画強化: HMM ラベルをフィットネスに統合、**既存NSGA-II + RL for ensemble** 動的調整、regime-specific backtest。

## 現在の実装レビューと弱点（実装状況反映版）

### 実装概要

- **コアコンポーネント**:
  - `AutoStrategyService`: GA 実験の開始/停止/一覧管理。バックグラウンドタスクで実行。
  - `ExperimentManager`: GA エンジン初期化と進化実行（`run_experiment`）。DEAP ベースの標準 GA。
  - `GeneticAlgorithmEngine`: 個体生成（`RandomGeneGenerator`）、進化ループ（`run_evolution`）、**マルチフィットネス評価対応済み**。
  - **フィットネス: マルチ指標対応済み（Sharpe Ratio, Max Drawdown, Win Rate, CVaRなど）**。戦略は取引条件（エントリー/エグジット）の遺伝子で表現。
  - DB 統合: ExperimentPersistenceService で結果保存（SQLite）。
- **フロー**: API コール → 実験作成 → GA 初期化 → 進化（個体生成 → バックテスト → 選択/変異） → 結果保存。
- **使用ライブラリ**: DEAP (GA), backtesting.py (バックテスト), pandas/numpy (データ処理)。

### 弱点（実装状況反映版）

- **多様性不足**: **NSGA-IIは実装済み**であるが、レジーム（トレンド/レンジ）非対応。戦略が 1 タイプに偏る。
- **ロバストネス**: 過剰適合リスク（インサンプル最適化のみ）。アウトオブサンプル検証なし。
- **実践投入可能性**: 取引コスト/スリッページ無視。リアルタイム適応なし。

**マルチフィットネス関数は実装済み**（単一フィットネスではなく複数指標対応済み）。

## 強化案 (実装状況反映版)

多様性優先で以下のアイデアを実装。検索結果から: NSGA-II の non-dominated sorting/crowding for diversity (Medium/Deb 論文), HMM regime for ML training (QuantInsti: regime labels for RF, backtest with regime split), HMM+RL for adaptive trading (PPO-HMM: hidden state for policy adjustment, risk reduction in crypto)。

1. **レジーム検出統合**: HMM (hmmlearn) で市場を分類（トレンド/レンジ/高ボラ）。各レジームで別 GA 実行。強化: 検索例から HMM 状態ラベルをデータに追加（regime-adaptive backtest, QuantInsti: model_training_data['regime'] = regimes）。状態遷移でペナルティフィットネス。
2. **マルチフィットネス関数**: DEAP の NSGA-II に拡張。フィットネスベクトル: [Sharpe Ratio, -Max Drawdown, Win Rate, -CVaR]（4 次元、Medium DEAP 例: creator.FitnessMulti weights=(1.0, -1.0, 1.0, -1.0), selNSGA2）。Pareto 最適戦略を多様生成。DEAP コード例 (Medium 準拠):

   ```
   from deap import creator, base, tools, algorithms
   creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0, 1.0, -1.0))
   creator.create("Individual", list, fitness=creator.FitnessMulti)
   toolbox = base.Toolbox()
   toolbox.register("evaluate", multi_fitness_eval)  # Sharpe, -Drawdown, Win Rate, -CVaR
   pop = toolbox.population(n=pop_size)
   algorithms.eaMuPlusLambda(pop, toolbox, mu=pop_size, lambda_=lambda_, cxpb=0.7, mutpb=0.3, ngen=ngen, sel=tools.selNSGA2)
   pareto_front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
   ```

   強化: Hypervolume 指標で Pareto 品質測定（DEAP.tools, GeeksforGeeks: constraint handling for feasible solutions）。

3. **戦略タイプ多様化**: 初期個体にテンプレート（トレンドフォロー/均値回帰/ブレイクアウト）を導入。クロスオーバーでハイブリッド。強化: NSGA-II の crowding distance for diversity (Deb 論文: better spread near Pareto front), niching 追加。
4. **アンサンブル戦略**: 複数レジームの Pareto 戦略を組み合わせ、動的スイッチング（レジーム検出ベース）。強化: HMM+RL ハイブリッド (IDS2025: PPO-HMM for regime-aware policy, dynamic adjustment based on hidden state)。Optuna で重み最適化 + RL (stable-baselines3 PPO) for live switching。
5. **ロバストネス/実践性補助**: Walk-Forward Analysis 追加。取引コストモデル（手数料 0.1%, スリッページ 0.05%）統合。強化: CVaR 計算 (scipy.stats, ResearchGate: multi-obj risk portfolio with NSGA-II), regime-specific backtest (QuantInsti: run_backtest with regime filter)。

### 期待効果

- 多様な戦略生成で市場変動耐性向上（NSGA-II Pareto for trade-off, HMM regime for adaptive）。
- 計算効率: NSGA-II O(MN^2) with elitism (Deb: fast elitist), RL for real-time (PPO stable learning)。
- TDD で信頼性確保。暗号通貨特化: 高ボラ対応で CVaR/PPO 重視 (論文: risk control in stochastic markets)。

## システムフロー視覚化 (追加強化版)

HMM+RL と NSGA-II 詳細を強調。

```mermaid
flowchart TD
    subgraph "現在の単一フィットネスフロー"
        A[データ収集] --> B[GA初期化: 単一フィットネス]
        B --> C[進化: 個体生成 → 単一バックテスト → 選択]
        C --> D[単一最適戦略出力]
    end

    subgraph "強化フロー: 多様性 + NSGA-II + HMM-RL"
        E[データ収集] --> F[レジーム検出: HMM分類 + 状態ラベル追加]
        F --> G[レジーム別GA: テンプレート初期個体 + Niching/Crowding]
        G --> H[NSGA-II進化: Non-dominated Sorting + Evaluate Multi (Sharpe, Drawdown, Win Rate, CVaR)]
        H --> I[Hypervolume評価 + アンサンブル: Pareto組み合わせ + PPO-HMM動的スイッチ + Walk-Forward/Regime Backtest]
        I --> J[多様ロバスト戦略出力: コスト/VaR考慮 + RLライブ適応]
    end
```

## 実装ステップ（TDD ベース、実装状況反映版）

TDD で進める: 各ステップでテスト作成 → 実装 → リファクタ。pytest でカバレッジ 90%以上目指す。変更ファイル: backend/app/services/auto*strategy/*, tests/\_。検索から DEAP/HMM/RL 例をコード参考に。

### 1. 準備（依存追加/初期テスト）

- pyproject.toml に hmmlearn, scipy, stable-baselines3 (PPO 用)追加（`hmmlearn>=0.3.0`, `scipy>=1.11.0`, `stable-baselines3>=2.0.0`）。
- 新規テストファイル: `backend/tests/test_regime_detection.py`, `test_nsga2_multi_fitness.py`, `test_ppo_hmm_ensemble.py`。
- TODO: [ ] 依存インストール確認（pytest でテスト実行）。 [ ] DEAP NSGA-II + stable-baselines3 サンプルテスト。

### 2. レジーム検出実装

- 新規クラス: `RegimeDetector` (backend/app/services/auto_strategy/core/regime_detector.py)。
  - メソッド: `detect_regimes(ohlcv_data) → List[Regime]` (HMM フィット/状態推定, QuantInsti: regimes_for_rf_training)。
  - 強化: 状態ラベルをデータフレームに追加 (model_training_data['regime'] = regimes)。
  - テスト: 合成データで分類精度/遷移検証（精度>80%, GitHub regime_detection_ml 準拠）。
- 統合: ExperimentManager に`detect_and_split_data`追加 (regime filter for backtest)。
- TODO: [ ] テスト駆動で HMM モデル実装。 [ ] 暗号通貨データ（BTC/USDT）で精度評価 + regime backtest。

### 3. 既存NSGA-IIの活用と拡張

- **GeneticAlgorithmEngine は既にNSGA-II対応済み**のため、設定拡張。
  - GAConfig で`enable_multi_objective: true`設定をデフォルト化。
  - Pareto 保存: 結果 DB に複数戦略 + hypervolume。
- テスト: **既存テスト拡張**でベクトル/ソート/crowding 確認。
- TODO: [ ] マルチフィットネス設定のデフォルト有効化。 [ ] Pareto フロント/Hypervolume 生成検証（spread/convergence 確認）。

### 4. 戦略タイプ多様化（Fitness Sharing活用）

- **FitnessSharingは実装済み**のため、設定調整。
  - StrategyFactory 拡張: `create_template_population(num_individuals, regime_type)` (テンプレート: MACD/RSI/Bollinger, regime-aware)。
  - 初期化: RandomGeneGenerator にテンプレート混在 + **既存のFitness Sharing活用**。
- テスト: **既存テスト拡張**で多様性指標（ハミング距離/hypervolume）。
- TODO: [ ] テンプレート定義/テスト。 [ ] Fitness Sharingパラメータ最適化。

### 5. アンサンブルとロバストネス

- 新規クラス: `EnsembleBuilder` (core/ensemble_builder.py)。
  - メソッド: `build_ensemble(pareto_strategies, regimes) → EnsembleStrategy` (重み付けスイッチング, Optuna + PPO for dynamic, IDS2025: PPO-HMM hidden state adjustment)。
- Walk-Forward: BacktestService に`walk_forward_optimize`追加（期間分割, regime split backtest, QuantInsti: run_backtest with regime）。
- コスト統合: BacktestConfig に`cost_model`追加（手数料/スリッページ + CVaR, ResearchGate: multi-obj risk）。
- テスト: `test_ensemble_ppo.py`, `test_walk_forward_regime.py` (stable learning, risk reduction)。
- TODO: [ ] アンサンブル構築テスト。 [ ] CVaR/コスト影響シミュレーション（crypto stochastic, PPO convergence）。

### 6. API/フロントエンド調整

- API 拡張: `/generate` に`regime_mode: bool`, `use_rl: bool`パラメータ追加。**既存のenable_multi_objectiveを活用**。
- フロント: **useAutoStrategy.ts はマルチフィットネス対応済み**のため、regime/RLオプション追加。
- テスト: `test_api_regime_rl.py` (FastAPI TestClient, regime/PPO reward 閾値検証)。
- TODO: [ ] API エンドツーエンドテスト。 [ ] フロント統合確認 (regime label display)。

### 7. 全体検証とデプロイ

- 統合テスト: `comprehensive_nsga2_hmm_test.py` でエンドツーエンド（データ →HMM→NSGA-II→PPO アンサンブル）。
- パフォーマンス: 実行時間監視、Optuna で GA/RL パラ最適化 (PPO: stable-baselines3, IDS2025: lower variance)。
- 強化: GA 終了条件に convergence (stagnation, Deb: elitist MOEA), RL reward with regime penalty。
- ドキュメント: この計画更新、README.md に NSGA-II/HMM-RL/CVaR 説明。
- TODO: [ ] 全テスト実行/カバレッジ確認。 [ ] 生産環境テスト（Bybit ライブデータ, CVaR<5%, PPO reward>0.8）。

## リスクと緩和 (追加強化)

- 計算コスト増: NSGA-II O(MN^2 log N) + PPO training 対応で pop_size=50, batch_size=32 制限, GPU (torch for PPO)。
- 複雑性: モジュール化/TDD + 検索例で段階実装 (Medium DEAP tutorial, QuantInsti HMM code)。
- 暗号通貨特化リスク: 高ノイズ →CVaR/PPO 重視 + regime-adaptive (論文: SAC/DDPG for stochastic, TD3 tuning)。
- ライブラリ互換: DEAP 1.4+, hmmlearn 0.3+, stable-baselines3 2.0+固定, pytest-cov でカバレッジ。

## 次アクション

- Code モードスイッチ後、上記ステップ順に TDD 実装。**レジーム検出から開始**（基盤拡張）。
- 進捗: Git コミットごとに TODO 更新。**NSGA-II基盤を活用**して効率化。
- 優先順位: 1. レジーム検出 → 2. PPO-HMM統合 → 3. ウォークフォワード → 4. テスト・調整

作成日: 2025-09-24
