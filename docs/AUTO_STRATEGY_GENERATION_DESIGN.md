# 自動ストラテジー生成機能 設計書

## 📋 プロジェクト概要

### 目的

遺伝的アルゴリズム（GA）を活用して、複数のテクニカル指標を組み合わせた取引戦略を自動生成し、既存のバックテストシステムと統合する機能を実装する。

### 背景

- 既存の 24 種類のテクニカル指標（TA-Lib 実装）を活用
- 手動での戦略作成の効率化
- データドリブンな戦略開発の実現
- 市場環境に適応する戦略の自動発見

### 技術スタック

- **バックエンド**: Python, FastAPI, backtesting.py, TA-Lib, DEAP（GA）
- **フロントエンド**: Next.js, TypeScript, Tailwind CSS
- **データベース**: SQLite
- **既存システム**: TALibAdapter, TechnicalIndicatorService, BacktestService

## 🎯 要件定義

### 機能要件

#### 1. 戦略遺伝子システム

- テクニカル指標の組み合わせをエンコード
- エントリー・イグジット条件の表現
- パラメータ範囲の定義
- 制約条件の設定

#### 2. 遺伝的アルゴリズムエンジン

- 戦略個体の生成・交叉・突然変異
- フィットネス評価（バックテスト結果ベース）
- エリート保存戦略
- 多様性維持機構

#### 3. 戦略ファクトリー

- 遺伝子から実行可能な戦略クラス生成
- backtesting.py 互換の Strategy 継承クラス作成
- 動的なテクニカル指標組み合わせ
- パラメータ検証機能

#### 4. 評価・最適化システム

- 複数の評価指標（シャープレシオ、最大ドローダウン等）
- 多目的最適化対応
- ロバストネス評価
- オーバーフィッティング検出

### 非機能要件

#### パフォーマンス

- 並列処理による高速化（マルチプロセッシング）
- 大規模戦略候補の効率的処理（1000 個体以上）
- リアルタイム進捗表示
- メモリ効率的な実装

#### 拡張性

- 新しいテクニカル指標の追加容易性
- カスタム評価指標の実装
- 戦略テンプレートの拡張
- プラグイン機構

#### 互換性

- 既存バックテストシステムとの完全統合
- TALibAdapter との互換性維持
- 既存 API 仕様の保持

## 🏗️ システム設計

### アーキテクチャ概要

```
┌─────────────────────────────────────────────────────────────┐
│                    フロントエンド                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ 戦略生成モーダル  │  │ 進捗表示        │  │ 結果比較        │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                      API Layer                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ 戦略生成API     │  │ 進捗監視API     │  │ 結果取得API     │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                   ビジネスロジック層                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ GAエンジン      │  │ 戦略ファクトリー  │  │ 評価システム    │ │
│  │ - 個体生成      │  │ - 遺伝子デコード  │  │ - フィットネス  │ │
│  │ - 交叉・突然変異 │  │ - 戦略クラス生成  │  │ - 多目的最適化  │ │
│  │ - 選択・淘汰    │  │ - パラメータ検証  │  │ - ロバストネス  │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                    既存システム統合                           │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ TALibAdapter    │  │ BacktestService │  │ データベース    │ │
│  │ (24指標)        │  │ (backtesting.py)│  │ (結果保存)      │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### コンポーネント設計

#### 1. 戦略遺伝子（StrategyGene）

```python
class StrategyGene:
    indicators: List[IndicatorGene]     # 使用する指標
    entry_conditions: List[Condition]   # エントリー条件
    exit_conditions: List[Condition]    # イグジット条件
    risk_management: RiskManagement     # リスク管理
    parameters: Dict[str, float]        # パラメータ
```

#### 2. GA エンジン（GeneticAlgorithmEngine）

```python
class GeneticAlgorithmEngine:
    population_size: int = 100
    generations: int = 50
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    elite_size: int = 10
```

#### 3. 戦略ファクトリー（StrategyFactory）

```python
class StrategyFactory:
    def create_strategy(gene: StrategyGene) -> Type[Strategy]
    def validate_gene(gene: StrategyGene) -> bool
    def encode_strategy(strategy: Strategy) -> StrategyGene
```

## 📊 データベース設計

### 新規テーブル

#### 1. generated_strategies

```sql
CREATE TABLE generated_strategies (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    gene_data JSONB NOT NULL,           -- 戦略遺伝子
    generation INTEGER NOT NULL,        -- 世代数
    fitness_score FLOAT,               -- フィットネススコア
    parent_ids INTEGER[],              -- 親戦略のID
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

#### 2. ga_experiments

```sql
CREATE TABLE ga_experiments (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    config JSONB NOT NULL,             -- GA設定
    status VARCHAR(50) DEFAULT 'running',
    progress FLOAT DEFAULT 0.0,
    best_fitness FLOAT,
    total_generations INTEGER,
    current_generation INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP
);
```

#### 3. strategy_evaluations

```sql
CREATE TABLE strategy_evaluations (
    id SERIAL PRIMARY KEY,
    strategy_id INTEGER REFERENCES generated_strategies(id),
    experiment_id INTEGER REFERENCES ga_experiments(id),
    backtest_result_id INTEGER REFERENCES backtest_results(id),
    fitness_metrics JSONB,             -- 評価指標
    evaluation_time FLOAT,             -- 評価時間
    created_at TIMESTAMP DEFAULT NOW()
);
```

## 🎨 フロントエンド設計

### 1. 戦略生成モーダル（StrategyGenerationModal）

#### 基本設定セクション

- 実験名
- 対象通貨ペア（既存設定から自動引き継ぎ）
- 時間軸（既存設定から自動引き継ぎ）
- バックテスト期間（全 OHLCV データの最初の日付〜今日を自動設定）
- 初期資金・手数料（既存設定から自動引き継ぎ）

#### GA 設定セクション

- 個体数（デフォルト: 100）
- 世代数（デフォルト: 50）
- 交叉率（デフォルト: 0.8）
- 突然変異率（デフォルト: 0.1）
- エリート保存数（デフォルト: 10）

#### 戦略制約セクション

- 使用可能テクニカル指標の選択
- 最大指標数制限
- パラメータ範囲設定
- 評価指標の重み設定

### 2. 進捗表示コンポーネント（GAProgressDisplay）

- リアルタイム進捗バー
- 現在の世代数・最高フィットネス
- 実行時間・推定残り時間
- 最優秀戦略のプレビュー

### 3. 結果比較コンポーネント（StrategyComparisonView）

- 生成された戦略一覧（フィットネス順）
- 戦略詳細表示（使用指標、条件、パラメータ）
- バックテスト結果比較
- 戦略のエクスポート・保存機能

## 🚀 実装計画

### フェーズ 1: 基盤実装（4 週間）

#### 成果物

- StrategyGene クラス実装
- GeneticAlgorithmEngine 基本機能
- StrategyFactory プロトタイプ
- 基本的なテスト実装

#### 技術的マイルストーン

- 遺伝子エンコード・デコード機能
- 基本的な GA 操作（選択、交叉、突然変異）
- 簡単な戦略生成・実行

### フェーズ 2: バックテスト統合（3 週間）

#### 成果物

- BacktestService との統合
- 並列処理実装
- フィットネス評価システム
- データベーススキーマ実装

#### 技術的マイルストーン

- 大規模戦略評価の高速化
- 既存バックテストシステムとの完全互換
- 結果の永続化

### フェーズ 3: フロントエンド実装（3 週間）

#### 成果物

- 戦略生成モーダル
- 進捗表示機能
- 結果表示・比較機能
- API 統合

#### 技術的マイルストーン

- モーダルベース UI（ユーザー好み準拠）
- リアルタイム進捗表示
- 既存設定の自動引き継ぎ

### フェーズ 4: 高度機能実装（2 週間）

#### 成果物

- 多目的最適化
- ロバストネス評価
- カスタム制約条件
- パフォーマンス最適化

#### 技術的マイルストーン

- NSGA-II 実装
- オーバーフィッティング検出
- 高速化・メモリ最適化

## 🔧 技術的考慮事項

### パフォーマンス最適化

- **並列処理**: multiprocessing による戦略評価の並列化
- **メモリ管理**: 大量個体の効率的なメモリ使用
- **キャッシュ**: 重複計算の回避
- **データベース最適化**: インデックス設計、クエリ最適化

### セキュリティ

- **入力検証**: 戦略パラメータの範囲チェック
- **リソース制限**: CPU・メモリ使用量の制限
- **実行時間制限**: 無限ループ防止

### 監視・ログ

- **進捗監視**: リアルタイム実行状況の追跡
- **エラーハンドリング**: 戦略実行エラーの適切な処理
- **パフォーマンス監視**: 実行時間・リソース使用量の記録

### 拡張性設計

- **プラグイン機構**: 新しい指標・条件の追加
- **設定駆動**: YAML/JSON による設定管理
- **モジュラー設計**: 独立したコンポーネント設計

## 📝 テスト戦略

### TDD アプローチ

1. **単体テスト**: 各コンポーネントの基本機能
2. **統合テスト**: システム間連携の確認
3. **パフォーマンステスト**: 大規模データでの性能確認
4. **エンドツーエンドテスト**: フロントエンドからバックエンドまでの完全なフロー

### テスト構成

```
backend/tests/
├── auto_strategy/
│   ├── test_strategy_gene.py
│   ├── test_ga_engine.py
│   ├── test_strategy_factory.py
│   └── test_integration.py
└── performance/
    └── test_ga_performance.py
```

## 🎯 成功指標

### 機能的指標

- 戦略生成成功率: 95%以上
- バックテスト実行成功率: 99%以上
- 生成戦略の多様性: 指標組み合わせの重複率 30%以下

### パフォーマンス指標

- 100 個体 ×50 世代の実行時間: 30 分以内
- メモリ使用量: 8GB 以内
- 並列処理効率: 70%以上

### ユーザビリティ指標

- モーダル操作の直感性
- 進捗表示の分かりやすさ
- 結果比較の有用性

## 📋 詳細実装仕様

### ディレクトリ構造

```
backend/app/core/services/
├── auto_strategy/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── strategy_gene.py          # 戦略遺伝子モデル
│   │   ├── ga_config.py              # GA設定モデル
│   │   └── fitness_metrics.py        # フィットネス評価モデル
│   ├── engines/
│   │   ├── __init__.py
│   │   ├── ga_engine.py              # GAエンジン本体
│   │   ├── selection.py              # 選択アルゴリズム
│   │   ├── crossover.py              # 交叉アルゴリズム
│   │   └── mutation.py               # 突然変異アルゴリズム
│   ├── factories/
│   │   ├── __init__.py
│   │   ├── strategy_factory.py       # 戦略ファクトリー
│   │   ├── indicator_factory.py      # 指標ファクトリー
│   │   └── condition_factory.py      # 条件ファクトリー
│   ├── evaluators/
│   │   ├── __init__.py
│   │   ├── fitness_evaluator.py      # フィットネス評価器
│   │   ├── multi_objective.py        # 多目的最適化
│   │   └── robustness.py             # ロバストネス評価
│   └── services/
│       ├── __init__.py
│       ├── auto_strategy_service.py  # メインサービス
│       ├── experiment_service.py     # 実験管理サービス
│       └── progress_service.py       # 進捗管理サービス
```

### API 設計

#### 1. 戦略生成実行 API

```python
POST /api/auto-strategy/generate
{
    "experiment_name": "BTC_Strategy_Gen_001",
    "base_config": {
        "symbol": "BTC/USDT",
        "timeframe": "1h",
        "start_date": "2023-01-01",
        "end_date": "2024-12-19",
        "initial_capital": 100000,
        "commission_rate": 0.00055
    },
    "ga_config": {
        "population_size": 100,
        "generations": 50,
        "crossover_rate": 0.8,
        "mutation_rate": 0.1,
        "elite_size": 10
    },
    "constraints": {
        "max_indicators": 5,
        "allowed_indicators": ["SMA", "EMA", "RSI", "MACD", "BB"],
        "parameter_ranges": {
            "SMA_period": [5, 200],
            "RSI_period": [10, 30],
            "RSI_overbought": [70, 90],
            "RSI_oversold": [10, 30]
        }
    },
    "fitness_config": {
        "primary_metric": "sharpe_ratio",
        "weights": {
            "total_return": 0.3,
            "sharpe_ratio": 0.4,
            "max_drawdown": 0.2,
            "win_rate": 0.1
        },
        "constraints": {
            "min_trades": 10,
            "max_drawdown_limit": 0.3
        }
    }
}
```

#### 2. 進捗監視 API

```python
GET /api/auto-strategy/experiments/{experiment_id}/progress
{
    "experiment_id": "exp_001",
    "status": "running",
    "progress": 0.65,
    "current_generation": 33,
    "total_generations": 50,
    "best_fitness": 1.85,
    "average_fitness": 0.92,
    "execution_time": 1800,
    "estimated_remaining": 900,
    "best_strategy_preview": {
        "indicators": ["SMA(20)", "RSI(14)", "MACD(12,26,9)"],
        "entry_condition": "SMA_cross_up AND RSI < 30",
        "exit_condition": "SMA_cross_down OR RSI > 70"
    }
}
```

#### 3. 結果取得 API

```python
GET /api/auto-strategy/experiments/{experiment_id}/results
{
    "experiment_id": "exp_001",
    "status": "completed",
    "total_strategies": 5000,
    "best_strategies": [
        {
            "id": "strategy_001",
            "fitness_score": 2.15,
            "gene": {...},
            "backtest_result": {...},
            "performance_metrics": {...}
        }
    ],
    "diversity_metrics": {
        "unique_indicator_combinations": 85,
        "parameter_distribution": {...}
    }
}
```

### 戦略遺伝子詳細設計

#### IndicatorGene

```python
@dataclass
class IndicatorGene:
    type: str                    # "SMA", "EMA", "RSI", etc.
    parameters: Dict[str, float] # {"period": 20, "source": "close"}
    weight: float               # 条件での重み (0.0-1.0)
    enabled: bool               # 使用するかどうか
```

#### Condition

```python
@dataclass
class Condition:
    left_operand: str           # "SMA_20", "RSI_14", "price"
    operator: str               # ">", "<", "cross_above", "cross_below"
    right_operand: Union[str, float]  # "SMA_50", 70, etc.
    logic_operator: str         # "AND", "OR", "NOT"
```

#### StrategyGene

```python
@dataclass
class StrategyGene:
    id: str
    indicators: List[IndicatorGene]
    entry_conditions: List[Condition]
    exit_conditions: List[Condition]
    risk_management: Dict[str, float]  # {"stop_loss": 0.02, "take_profit": 0.05}
    position_sizing: Dict[str, float]  # {"method": "fixed", "size": 1.0}
    metadata: Dict[str, Any]           # 追加情報

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換（データベース保存用）"""
        pass

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategyGene':
        """辞書から復元"""
        pass

    def validate(self) -> bool:
        """遺伝子の妥当性検証"""
        pass
```

### GA エンジン詳細設計

#### 選択アルゴリズム

```python
class SelectionMethods:
    @staticmethod
    def tournament_selection(population: List[StrategyGene],
                           fitness_scores: List[float],
                           tournament_size: int = 3) -> StrategyGene:
        """トーナメント選択"""
        pass

    @staticmethod
    def roulette_wheel_selection(population: List[StrategyGene],
                               fitness_scores: List[float]) -> StrategyGene:
        """ルーレット選択"""
        pass

    @staticmethod
    def rank_selection(population: List[StrategyGene],
                      fitness_scores: List[float]) -> StrategyGene:
        """ランク選択"""
        pass
```

#### 交叉アルゴリズム

```python
class CrossoverMethods:
    @staticmethod
    def uniform_crossover(parent1: StrategyGene,
                         parent2: StrategyGene,
                         crossover_rate: float = 0.5) -> Tuple[StrategyGene, StrategyGene]:
        """一様交叉"""
        pass

    @staticmethod
    def single_point_crossover(parent1: StrategyGene,
                              parent2: StrategyGene) -> Tuple[StrategyGene, StrategyGene]:
        """一点交叉"""
        pass

    @staticmethod
    def semantic_crossover(parent1: StrategyGene,
                          parent2: StrategyGene) -> Tuple[StrategyGene, StrategyGene]:
        """意味的交叉（指標の意味を考慮）"""
        pass
```

#### 突然変異アルゴリズム

```python
class MutationMethods:
    @staticmethod
    def parameter_mutation(gene: StrategyGene,
                          mutation_rate: float,
                          parameter_ranges: Dict[str, Tuple[float, float]]) -> StrategyGene:
        """パラメータ突然変異"""
        pass

    @staticmethod
    def indicator_mutation(gene: StrategyGene,
                          available_indicators: List[str]) -> StrategyGene:
        """指標突然変異（指標の追加・削除・変更）"""
        pass

    @staticmethod
    def condition_mutation(gene: StrategyGene) -> StrategyGene:
        """条件突然変異（条件の変更）"""
        pass
```

### フィットネス評価詳細設計

#### 単一目的評価

```python
class SingleObjectiveFitness:
    def __init__(self, primary_metric: str, weights: Dict[str, float]):
        self.primary_metric = primary_metric
        self.weights = weights

    def evaluate(self, backtest_result: Dict[str, Any]) -> float:
        """加重平均によるフィットネス計算"""
        metrics = backtest_result['performance_metrics']

        # 正規化された指標値を計算
        normalized_metrics = self._normalize_metrics(metrics)

        # 加重平均を計算
        fitness = sum(
            self.weights.get(metric, 0) * value
            for metric, value in normalized_metrics.items()
        )

        # ペナルティ適用
        fitness = self._apply_penalties(fitness, metrics)

        return fitness

    def _normalize_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """指標の正規化（0-1範囲）"""
        pass

    def _apply_penalties(self, fitness: float, metrics: Dict[str, float]) -> float:
        """制約違反に対するペナルティ"""
        pass
```

#### 多目的評価（NSGA-II）

```python
class MultiObjectiveFitness:
    def __init__(self, objectives: List[str]):
        self.objectives = objectives

    def evaluate_population(self, population: List[StrategyGene],
                          backtest_results: List[Dict[str, Any]]) -> List[Dict[str, float]]:
        """個体群の多目的評価"""
        pass

    def non_dominated_sort(self, fitness_values: List[Dict[str, float]]) -> List[List[int]]:
        """非劣解ソート"""
        pass

    def crowding_distance(self, front: List[int],
                         fitness_values: List[Dict[str, float]]) -> List[float]:
        """混雑距離計算"""
        pass
```

### 戦略ファクトリー詳細設計

```python
class StrategyFactory:
    def __init__(self, talib_adapter: TALibAdapter):
        self.talib_adapter = talib_adapter
        self.indicator_cache = {}

    def create_strategy_class(self, gene: StrategyGene) -> Type[Strategy]:
        """遺伝子から動的にStrategy継承クラスを生成"""

        class GeneratedStrategy(Strategy):
            def __init__(self):
                super().__init__()
                self.gene = gene
                self.indicators = {}

            def init(self):
                """指標の初期化"""
                for indicator_gene in gene.indicators:
                    if indicator_gene.enabled:
                        indicator = self._create_indicator(indicator_gene)
                        self.indicators[indicator_gene.type] = indicator

            def next(self):
                """売買ロジック"""
                # エントリー条件チェック
                if self._check_entry_conditions() and not self.position:
                    self.buy()

                # イグジット条件チェック
                elif self._check_exit_conditions() and self.position:
                    self.sell()

                # リスク管理
                self._apply_risk_management()

            def _create_indicator(self, indicator_gene: IndicatorGene):
                """指標インスタンス作成"""
                pass

            def _check_entry_conditions(self) -> bool:
                """エントリー条件評価"""
                pass

            def _check_exit_conditions(self) -> bool:
                """イグジット条件評価"""
                pass

            def _apply_risk_management(self):
                """リスク管理適用"""
                pass

        return GeneratedStrategy

    def validate_gene(self, gene: StrategyGene) -> Tuple[bool, List[str]]:
        """遺伝子の妥当性検証"""
        errors = []

        # 指標の妥当性チェック
        for indicator_gene in gene.indicators:
            if not self._validate_indicator(indicator_gene):
                errors.append(f"Invalid indicator: {indicator_gene.type}")

        # 条件の妥当性チェック
        if not self._validate_conditions(gene.entry_conditions):
            errors.append("Invalid entry conditions")

        if not self._validate_conditions(gene.exit_conditions):
            errors.append("Invalid exit conditions")

        return len(errors) == 0, errors
```

---

**📅 実装開始予定**: 設計承認後即座
**🎯 完成予定**: 12 週間後
**👥 開発体制**: TDD 重視、段階的実装、継続的テスト

## 📋 コードベース分析結果と設計書修正

### 🔍 重要な発見事項

#### 1. 既存ストラテジー実装の詳細分析

**発見された追加ストラテジー**:

- **MACDStrategy**: 基本 MACD 戦略 + 3 つの派生戦略
  - MACDDivergenceStrategy（ダイバージェンス検出）
  - MACDTrendStrategy（トレンドフィルター付き）
  - MACDScalpingStrategy（短期取引用）
- **RSIStrategy**: RSI 逆張り戦略 + 派生戦略
- **BaseStrategy**: 抽象基底クラス（共通インターフェース提供）

**パラメータ構造の統一性**:

```python
# 全戦略で統一されたパラメータ定義パターン
class StrategyExample(Strategy):
    # クラス変数としてパラメータ定義
    param1 = default_value
    param2 = default_value

    def validate_parameters(self) -> bool:
        # パラメータ検証の統一インターフェース
        pass

    def get_current_signals(self) -> dict:
        # シグナル状況取得の統一インターフェース
        pass
```

#### 2. EnhancedBacktestService の高度機能

**SAMBO 最適化の詳細実装**:

- backtesting.py 内蔵の SAMBO（Sequential Model-Based Optimization）
- 制約条件の事前定義済み関数（sma_cross, rsi, macd, risk_management）
- マルチプロセシング対応の並列処理
- ヒートマップ生成・保存機能

**制約条件システム**:

```python
def sma_cross_constraint(params):
    return params.n1 < params.n2

def rsi_constraint(params):
    return (params.rsi_lower < params.rsi_upper and
            10 <= params.rsi_lower <= 90)
```

#### 3. データベーススキーマの互換性

**既存テーブル構造**:

- **backtest_results**: JSON 形式で performance_metrics, equity_curve, trade_history 保存
- **technical_indicator_data**: 複合インデックス最適化済み
- **SQLite 使用**: PostgreSQL/TimescaleDB ではなく SQLite

**新規テーブル設計の修正が必要**:

```sql
-- SQLite対応の修正版
CREATE TABLE generated_strategies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,  -- SERIAL → INTEGER
    name TEXT NOT NULL,                     -- VARCHAR → TEXT
    gene_data TEXT NOT NULL,                -- JSONB → TEXT (JSON)
    generation INTEGER NOT NULL,
    fitness_score REAL,                     -- FLOAT → REAL
    parent_ids TEXT,                        -- INTEGER[] → TEXT (JSON)
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

#### 4. フロントエンドコンポーネントの再利用可能性

**OptimizationModal の拡張可能性**:

- タブベース設計（enhanced, multi, robustness）
- 設定の自動引き継ぎ機能実装済み
- Modal 共通コンポーネント（サイズ、動作カスタマイズ可能）

**useApiCall フックの活用**:

```typescript
const { execute, loading, error } = useApiCall<ResponseType>();
// 統一されたAPI呼び出しパターン
// ローディング状態、エラーハンドリング、成功コールバック
```

#### 5. テクニカル指標システムとの統合

**TALibAdapter の具体的使用方法**:

```python
# 機能別アダプター使用例
from app.core.services.indicators.adapters import TrendAdapter
result = TrendAdapter.sma(data, period)

# 統合サービス使用例
from app.core.services.indicators import TechnicalIndicatorService
service = TechnicalIndicatorService()
result = await service.calculate_technical_indicator(...)
```

#### 6. テスト基盤の活用

**既存テスト構造**:

- **conftest.py**: 共通フィクスチャ、パフォーマンス閾値定義
- **performance/**: 実データでのパフォーマンステスト
- **unit/**: モック使用の単体テスト
- **integration/**: 実システム統合テスト

### 🔧 設計書への重要な修正・追加

#### 1. 戦略遺伝子設計の拡張

```python
@dataclass
class StrategyGene:
    # 既存設計に追加
    strategy_template: str  # "SMA_CROSS", "MACD", "RSI", "HYBRID"
    risk_management: Dict[str, float]  # 既存戦略のリスク管理パターン活用
    signal_filters: List[Dict[str, Any]]  # 出来高フィルター等

    # 既存戦略との互換性
    def to_backtesting_strategy(self) -> Type[Strategy]:
        """既存のStrategy継承クラス形式に変換"""
        pass
```

#### 2. 制約条件システムの活用

```python
class GAConstraintManager:
    def __init__(self):
        # EnhancedBacktestServiceの制約条件を活用
        self.predefined_constraints = {
            "sma_cross": self._sma_cross_constraint,
            "rsi": self._rsi_constraint,
            "macd": self._macd_constraint,
            "risk_management": self._risk_management_constraint
        }
```

#### 3. データベース設計の修正

```sql
-- SQLite対応版
CREATE TABLE ga_experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    config TEXT NOT NULL,  -- JSON形式
    status TEXT DEFAULT 'running',
    progress REAL DEFAULT 0.0,
    best_fitness REAL,
    total_generations INTEGER,
    current_generation INTEGER DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    completed_at DATETIME
);
```

#### 4. フロントエンド設計の具体化

```typescript
// 既存OptimizationModalの拡張
interface StrategyGenerationModalProps extends OptimizationModalProps {
  onStrategyGeneration: (config: GAConfig) => void;
  gaExperimentId?: string;
  realTimeProgress?: GAProgress;
}

// useApiCallパターンの活用
const { execute: runGA, loading: gaLoading } = useApiCall<GAResult>();
```

#### 5. テスト戦略の具体化

```python
# backend/tests/auto_strategy/ 構造
├── unit/
│   ├── test_strategy_gene.py
│   ├── test_ga_engine.py
│   └── test_strategy_factory.py
├── integration/
│   ├── test_ga_backtest_integration.py
│   └── test_strategy_generation_api.py
└── performance/
    └── test_ga_performance.py
```

## 🔄 次のステップ

1. **設計書レビュー**: 修正された設計内容の確認・承認
2. **環境準備**: DEAP、scikit-optimize 等のライブラリインストール
3. **フェーズ 1 開始**: 修正された設計に基づく基盤実装
4. **継続的テスト**: 既存テストパターンを活用した TDD アプローチ

### 🎯 実装優先度の調整

**高優先度**:

1. 既存戦略パターンとの互換性確保
2. SQLite データベース対応
3. EnhancedBacktestService との統合

**中優先度**:

1. フロントエンドモーダル拡張
2. リアルタイム進捗表示
3. 制約条件システム活用

**低優先度**:

1. 高度な GA 機能（NSGA-II 等）
2. カスタム制約条件
3. パフォーマンス最適化

この修正された設計書に基づいて、既存システムとの完全な互換性を保ちながら、効率的な自動ストラテジー生成機能を実装していきます。
