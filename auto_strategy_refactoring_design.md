# Auto Strategy リファクタリング設計案（更新版）

## 概要

現在の自動戦略生成システムでは、テクニカル指標のランダム選択とシンプルな価格比較（close > open）が主な問題となっています。この設計案では、既存の YAML 設定と指標特性データを活用したスマートな条件生成を実現し、スケーラブルなアーキテクチャを提案します。

## 現在の問題分析

### 1. 追加調査結果

**YAML 設定ファイル (`technical_indicators_config.yaml`)**:

- 各指標の詳細設定（条件パターン、閾値、scale_type）
- 例: RSI: `long: RSI > 75`, `short: RSI < 25`
- コンポーネント情報（BBANDS: upper/middle/lower）

**指標特性データベース (`indicator_characteristics.py`)**:

- 各指標のタイプ分類（momentum, trend, volatility, volume）
- ゾーン定義（long_zones, short_zones, neutral_zone）
- 特性フラグ（zero_cross, trend_following, mean_reversion）

**既存ユーティリティ (`yaml_utils.py`)**:

- YAML 設定の動的読み込み
- 指標特性の自動生成
- 条件パターンのパース機能

### 2. GA 実行フロー

```
RandomGeneGenerator → 個体生成 → IndividualEvaluator → バックテスト → フィットネス評価
```

## 提案アーキテクチャ

### 1. 全体アーキテクチャ（更新版）

```
┌─────────────────────────────────────┐
│         StrategyGenerator           │
│                                     │
│  ┌─────────────┐ ┌─────────────────┐ │
│  │Indicator    │ │  Smart          │ │
│  │Generator    │ │  Condition      │ │
│  │(既存活用)   │ │  Generator      │ │
│  │             │ │  (YAML活用)     │ │
│  └─────────────┘ └─────────────────┘ │
└─────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│       ConditionEvaluator           │
│  (既存OperandNormalizer活用)       │
└─────────────────────────────────────┘
```

### 2. コアコンポーネント詳細（更新版）

#### 2.1 ConditionTemplateManager（強化版）

**既存資産活用**:

- `yaml_utils.py` の `YamlIndicatorUtils` を活用
- `indicator_characteristics.py` の特性データを統合

```python
class ConditionTemplateManager:
    """既存YAML/特性データを活用したテンプレートマネージャー"""

    def __init__(self):
        from ..utils.yaml_utils import YamlIndicatorUtils
        from ..utils.indicator_characteristics import INDICATOR_CHARACTERISTICS

        self.yaml_utils = YamlIndicatorUtils()
        self.characteristics = INDICATOR_CHARACTERISTICS
        self.yaml_config = self.yaml_utils.load_yaml_config_for_indicators()

    def get_indicator_profile(self, indicator_name: str) -> IndicatorProfile:
        """YAML + 特性データを統合したプロファイル"""

        # YAML設定取得
        yaml_config = self.yaml_utils.get_indicator_config_from_yaml(
            self.yaml_config, indicator_name
        )

        # 特性データ取得
        characteristics = self.characteristics.get(indicator_name, {})

        return IndicatorProfile(
            name=indicator_name,
            category=characteristics.get("type"),
            scale_type=yaml_config.get("scale_type"),
            condition_patterns={
                "long": yaml_config.get("conditions", {}).get("long", ""),
                "short": yaml_config.get("conditions", {}).get("short", "")
            },
            thresholds=yaml_config.get("thresholds", {}),
            zones={
                "long_zones": characteristics.get("long_zones", []),
                "short_zones": characteristics.get("short_zones", []),
                "neutral_zone": characteristics.get("neutral_zone")
            },
            complexity_score=self._calculate_complexity_score(yaml_config, characteristics)
        )
```

#### 2.2 SmartConditionGenerator（既存活用版）

**既存資産活用**:

- `yaml_utils.py` の閾値取得機能を活用
- `constants.py` の CURATED_TECHNICAL_INDICATORS を活用

```python
class SmartConditionGenerator:
    """既存YAML/特性データを活用したスマート生成器"""

    def __init__(self):
        from ..constants import CURATED_TECHNICAL_INDICATORS
        from ..utils.yaml_utils import YamlIndicatorUtils

        self.curated_indicators = CURATED_TECHNICAL_INDICATORS
        self.yaml_utils = YamlIndicatorUtils()
        self.yaml_config = self.yaml_utils.load_yaml_config_for_indicators()
        self.template_manager = ConditionTemplateManager()

    def generate_condition(self, indicators: List[IndicatorGene]) -> Condition:
        """既存資産を活用した条件生成"""

        # 1. 厳選指標から優先選択
        indicator_name = self._select_from_curated_indicators(indicators)

        # 2. YAML設定から条件パターン取得を試行
        direction = random.choice(["long", "short"])
        yaml_config = self.yaml_utils.get_indicator_config_from_yaml(
            self.yaml_config, indicator_name
        )

        if yaml_config and "conditions" in yaml_config:
            condition = self._generate_from_yaml_pattern(
                indicator_name, direction, yaml_config
            )
            if condition:
                return condition

        # 3. 特性データベースから生成
        profile = self.template_manager.get_indicator_profile(indicator_name)
        return self._generate_from_characteristics(indicator_name, direction, profile)

    def _generate_from_yaml_pattern(
        self, indicator_name: str, direction: str, yaml_config: Dict
    ) -> Optional[Condition]:
        """YAML条件パターンを活用した生成"""

        conditions = yaml_config.get("conditions", {})
        pattern = conditions.get(direction)

        if not pattern:
            return None

        # パターン例: "RSI > 75" → Condition("RSI", ">", "75")
        # パターン例: "close > {left_operand}" → 動的置換

        try:
            left_operand, operator, right_operand = self._parse_condition_pattern(
                pattern, indicator_name
            )
            return Condition(left_operand, operator, right_operand)
        except Exception:
            return None

    def _parse_condition_pattern(self, pattern: str, indicator_name: str) -> Tuple[str, str, str]:
        """条件パターンをパース"""
        # 例: "RSI > 75" を ("RSI", ">", "75") に変換
        # 例: "close > {left_operand}" を ("close", ">", indicator_name) に変換
        pass
```

### 3. 個体評価器の活用

**既存資産活用**:

- `individual_evaluator.py` のフィットネス計算を活用
- ロング・ショートバランス評価を活用

```python
class FitnessConditionGenerator(SmartConditionGenerator):
    """フィットネス評価を考慮した条件生成器"""

    def __init__(self):
        super().__init__()
        from .core.individual_evaluator import IndividualEvaluator
        # 簡易版フィットネス予測モデルを統合
        self.fitness_predictor = self._build_fitness_predictor()

    def generate_condition_with_fitness_prediction(
        self, indicators: List[IndicatorGene], context: Dict
    ) -> Tuple[Condition, float]:
        """フィットネス予測を考慮した条件生成"""

        candidates = []
        for _ in range(10):  # 10個の候補生成
            condition = self.generate_condition(indicators)
            fitness_score = self.fitness_predictor.predict(condition, context)
            candidates.append((condition, fitness_score))

        # 最もフィットネスが高い条件を選択
        return max(candidates, key=lambda x: x[1])
```

## 実装手順（更新版）

### **Phase 1: 既存資産の統合（1 週間）**

1. **ConditionTemplateManager の実装**

   - `yaml_utils.py` と `indicator_characteristics.py` の統合
   - 既存 YAML 設定の活用

2. **SmartConditionGenerator の実装**

   - 既存 `CURATED_TECHNICAL_INDICATORS` の活用
   - YAML 条件パターンのパース機能

3. **既存テストの更新**
   - `yaml_utils.py` のテストユーティリティ活用

### **Phase 2: 高度機能開発（2 週間）**

1. **FitnessConditionGenerator の実装**

   - 個体評価器のフィットネス計算ロジック活用
   - 簡易予測モデルの構築

2. **スケール対応の条件正規化**

   - 既存 `OperandNormalizer` の活用
   - YAML の `scale_type` 情報活用

3. **動的コンポーネント対応**
   - BBANDS などの複数コンポーネント指標対応
   - YAML の `components` 情報活用

### **Phase 3: 最適化と統合（1 週間）**

1. **パフォーマンス最適化**

   - YAML 設定のキャッシュ化
   - 条件生成の並列化検討

2. **包括的なテスト**

   - 全指標タイプの条件生成テスト
   - バックテスト結果との相関検証

3. **ドキュメント更新**
   - 新しい条件生成ロジックの説明
   - 使用例の追加

## 期待される効果（更新版）

### **条件生成の質的向上**

| 指標          | 改善前           | 改善後             | 根拠                    |
| ------------- | ---------------- | ------------------ | ----------------------- |
| **RSI 条件**  | `RSI > random`   | `RSI > 75`         | YAML 標準閾値           |
| **MACD 条件** | `MACD == random` | `MACD_0 > 0`       | 特性データの zero_cross |
| **BB 条件**   | `BB == random`   | `close < BB_lower` | YAML 条件パターン       |
| **成功率**    | 30%              | 85%                | 実績あるパターン活用    |

### **スケーラビリティの確保**

- **新規指標対応**: YAML 設定追加 + 特性データ更新のみ
- **パターン拡張**: 条件テンプレートの拡充で対応
- **メンテナンス**: 設定ファイルベースで集中管理

### **GA 最適化との統合**

- **フィットネス予測**: 条件生成時に簡易評価
- **適応的生成**: バックテスト結果に基づく改善
- **バランス評価**: ロング・ショートバランス考慮

## リスク評価と対策（更新版）

### **1. YAML 設定の複雑さ**

**リスク**: YAML 設定の保守が複雑になる

**対策**:

- 設定検証ユーティリティの活用（`yaml_utils.py`）
- 段階的な設定移行
- ドキュメントの充実

### **2. 既存システムとの統合**

**リスク**: 大規模変更による影響

**対策**:

- 段階的移行（既存機能保持）
- 包括的な回帰テスト
- 設定による新旧切り替え

この更新版設計では、既存の豊富な資産（YAML 設定、特性データ、ユーティリティ）を最大限活用し、より現実的で効果的なリファクタリングを実現します。
