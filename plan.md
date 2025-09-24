# オートストラテジー強化計画 (ML 代替中心)

## 目的

ロバストで実践投入可能な多様的な戦略構築。ML 上下予測の信頼低 (ボラ/ノイズ/オーバーフィット) を考慮し、ルールベース/GA を主力に。既存コード (DEAP GA, IndividualEvaluator フィットネス, StrategyFactory 生成, BacktestService 実行) を拡張。TDD (pytest) で検証、負けリスク低減 (MDD15%低減目標)。

## コンテキストまとめ

### 既存コード分析 (serena ツールから)

- **GA 構造**: GeneticAlgorithmEngine で DEAP 進化 (crossover/mutation)、RandomGeneGenerator で遺伝子生成 (条件/インジケーター/TP/SL/PositionSizing)。
- **フィットネス**: IndividualEvaluator で Sharpe/MDD/win_rate/balance 計算、多目的対応 (enable_multi_objective)。重み固定 (total_return 0.3, sharpe 0.4)、制約 (min_trades, max_dd)。
- **戦略生成**: StrategyFactory で動的 Strategy クラス (backtesting.py 互換)、ConditionEvaluator で条件評価。
- **バックテスト**: BacktestService で実行/DB 保存、performance_metrics (total_return, sharpe, MDD, calmar 既に一部)。
- **弱点**: regime 適応なし、多様性オプション (fitness_sharing オフ)、外部シグナル未活用、フィットネスリターン偏重。

### ML 信頼性 (ウェブ検索/論文総評)

- **限界**: 精度 50-80%だが、実践でオーバーフィット/変動期失敗 (Ren et al. 2022, <https://www.sciencedirect.com/science/article/pii/S027553192200129X>: "非線形捕捉優位だがランダム性で限界")。Khedr et al. 2021, <https://link.springer.com/article/10.1007/s12599-021-00710-6>: "LSTM 最高だがノイズで<70%"。Fang et al. 2020 survey, <https://link.springer.com/article/10.1186/s40854-020-00217-x>: "方向予測 60-80%だが drawdown 悪化、regime shift で一般化失敗"。
- **有識者意見**: Khosravi & Ghazani 2023, <https://www.sciencedirect.com/science/article/pii/S1057521923005719>: "SVM/XGBoost 短期優位もスケール難、ルール併用必須"。Schnaubelt 2022, <https://www.sciencedirect.com/science/article/pii/S0957417422001168>: "ML 不安定、TA/GA で補完"。総評: ML 補助 (特徴量生成)、主力は GA/ルール (解釈性/耐性高)。

### GA 改善知見 (追加ウェブ検索)

- **ロバスト強化**: GA でペアトレーディング最適化 (Enhanced GA Triple Barrier, MDPI 2024, <https://www.mdpi.com/2227-7390/12/5/780>: "GA でシグナル多様化、HRHP/LRLP でリスク/利益バランス")。MOEA/D 変種で多目的 (uniform mutation で探索向上, Trading Strategies Optimization, Essex 2024, <https://repository.essex.ac.uk/38969/1/PHD_THESIS_SALMAN.pdf)。>
- **Regime 検知**: GA に regime 統合 (HMM/MA で状態分類, Optimized pairs-trading, RePEc 2024, <https://ideas.repec.org/p/tut/cremwp/2024-11.html>: "コインテグレーション+GA で仮想通貨ペア耐性向上")。
- **Optuna チューニング**: GA ハイパーパラメータ最適化 (population/mutation, Optuna docs 2024, <https://optuna.org/>: "Bayesian/evolutionary sampler で効率、finance で Sharpe 最大化")。Hyperparameter Tuning (Analytics Vidhya 2020, <https://www.analyticsvidhya.com/blog/2020/11/hyperparameter-tuning-using-optuna/>: "Optuna で grid/random 超え、ML/GA 統合")。
- **代替戦略**: GA で TA ルール最適化 (Using Genetic Algorithms To Forecast, Investopedia, <https://www.investopedia.com/articles/financial-theory/11/using-genetic-algorithms-forecast-financial-markets.asp>: "パラメータ探索で解釈性高、ANN 代替")。GA+コインテグレーションで仮想通貨 (CREM WP 2024, <https://www.mdpi.com/2504-2289/7/4/174>: "threshold/weight 最適化で性能向上")。

## 設計詳細 (ML 代替中心)

### 1. フィットネス計算強化 (IndividualEvaluator 拡張)

- Calmar ratio = total_return / MDD 追加 (安定重視)。
- 重み動的 (GAConfig.fitness_weights = {"calmar": 0.3, "sharpe": 0.3, "mdd": 0.2, "win_rate": 0.1, "balance": 0.1})。
- 制約強化: min_trades=10, max_dd=0.2 → fitness=0。
- 影響: オーバーフィット低減 (検索知見反映)。

### 2. Regime 検知 (utils/label_generation 拡張)

- HMM (hmmlearn) で 4 状態分類 (bull/bear/sideways/volatile)、入力: OHLCV+OI/Funding。
- 関数: detect_regime(data) → state。
- GeneGenerator 統合: regime テンプレート (bull: SMA buy, bear: RSI sell)。
- 代替: シンプル MA クロス (ボラ計算)。
- 影響: 多様戦略 (regime スイッチ, RePEc 2024 知見)。

### 3. 外部シグナル統合 (indicators/technical_indicators 拡張)

- FundingRateIndicator: rate > 0.01 → short シグナル。
- OpenInterestIndicator: OI 変動 > 10% → ボラ警戒 (position_size 低減)。
- IndicatorGene.type="funding"/"oi"追加、StrategyFactory で評価。
- 影響: ML 代替 (解釈性高, MDPI 2024)。

### 4. Optuna チューニング (optimization/optuna_optimizer 拡張)

- 対象: population_size (50-200), mutation_rate (0.1-0.3), generations (50-100), crossover_rate (0.5-0.9)。
- 目的: Sharpe 最大化 (study = optuna.create_study(direction="maximize"))。
- ExperimentManager で実行前チューニング (sampler="TPE")。
- 影響: GA 効率向上 (Optuna docs/Analytics Vidhya 知見)。

### 5. 検証/実践 (BacktestService 拡張)

- ウォークフォワード: データ分割 (80%最適/20%検証, run_backtest に param 追加)。
- TDD: pytest で Calmar/regime/Optuna テスト (カバレッジ 80%)。
- 影響: 実践耐性 (オーバーフィット避け, Essex 2024)。

## 実装計画 (Todo リスト)

- [ ] Calmar ratio 追加 (IndividualEvaluator)。
- [ ] HMM regime 検知 (label_generation)。
- [ ] 外部シグナル統合 (indicators)。
- [ ] Optuna チューニング (optimization)。
- [ ] pytest テスト追加 (TDD)。
- [ ] ウォークフォワード (BacktestService, オプション)。

## リスク/評価

- コスト: 低 (既存拡張)。
- 評価: バックテストで MDD/Sharpe 比較 (前後 15%改善目標)。
- レビュー: 変更提案歓迎 (例: HMM 簡略)。

この計画でレビューお願いします。承認後 code モード実装。
