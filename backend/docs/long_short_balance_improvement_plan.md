# Auto-Strategy ロング・ショートバランス改善計画

## 概要

現在のauto-strategy機能において「ロング・ショート両対応戦略が0%」という深刻な問題が発生している。本文書では、この問題の根本原因を分析し、包括的な解決策を提示する。

## 調査結果

### 現在の実装状況

#### 1. GeneDecoder の条件生成ロジック

**`_generate_balanced_long_short_conditions` メソッド（482-606行）**
- 主要指標（indicators[0]）のみを使用して条件生成
- 同一指標に対して相反する条件を生成
- 例：RSI指標で `RSI < 30` (ロング) と `RSI > 70` (ショート)

**問題点：**
- 同じ時点で両方の条件が満たされることは物理的に不可能
- 結果として戦略は常にロングオンリーまたはショートオンリーになる

#### 2. RandomGeneGenerator のロング・ショート条件生成

**`_generate_long_short_conditions` メソッド（516-596行）**
- 各指標に対して個別に条件を生成
- `_create_long_short_conditions_for_indicator` を呼び出し
- ADX指標では同じ条件（`ADX > 25`）をロング・ショート両方に設定

**問題点：**
- ADXのような指標で同一条件を使用
- 指標データが不足している場合の条件評価失敗

#### 3. StrategyFactory の条件評価ロジック

**条件評価メソッド（242-281行）**
- `_check_long_entry_conditions` と `_check_short_entry_conditions`
- 修正により、空条件時のフォールバック処理は改善済み

### 利用可能な指標タイプ

#### モメンタム系
- **RSI**: 0-100スケール、売られすぎ/買われすぎ判定
- **MACD**: ゼロ中心、トレンド転換点検出
- **STOCH**: 0-100スケール、オシレーター
- **CCI**: ±100スケール、サイクル検出

#### トレンド系
- **SMA/EMA**: 価格比率、トレンド方向判定
- **ADX**: トレンド強度（方向性なし）

#### ボラティリティ系
- **BB**: 複合結果（Upper, Middle, Lower Band）
- **ATR**: 絶対価格、ボラティリティ測定

## 問題分析

### 根本原因

1. **同一指標での相反条件**
   ```
   例：BB指標
   ロング条件: BB < 45
   ショート条件: BB > 55
   → 同時に満たされることは不可能
   ```

2. **指標データ不足**
   - テストで多くの指標（MACD、STOCH、BB、CCI、ADX、ATR）のデータが計算されていない
   - 「未対応のオペランド」エラーが大量発生

3. **ADXの誤用**
   - ADXはトレンド強度を示すが、方向性は示さない
   - 同じ条件をロング・ショート両方に設定している

4. **ボリンジャーバンドの簡略化**
   - BBは本来3つの値（Upper, Middle, Lower）を持つ
   - 現在は単一値として扱われている

### 影響範囲

- **戦略生成**: 70%の戦略で条件が満たされない
- **バックテスト**: 指標データ不足によるエラー
- **ユーザー体験**: 期待される多様な戦略が生成されない
### コードベースの追加分析

- **ロジックの散在と責務の曖昧さ**:
  - `GeneDecoder`内の`_generate_balanced_long_short_conditions`と`RandomGeneGenerator`内の`_create_long_short_conditions_for_indicator`に、ロング・ショート条件を生成するロジックが散在している。
  - `GeneDecoder`はランダムな組み合わせに依存し、`RandomGeneGenerator`は固定的で多様性に欠けるロジックとなっている。
  - この責務の曖昧さが、メンテナンス性と拡張性を著しく低下させている。

- **既存機能の活用不足**:
  - `RandomGeneGenerator`には、スケールの異なる指標同士の比較を避けるための`operand_grouping_system`が実装されているが、現在の条件生成ロジックでは十分に活用されていない。


## 設計計画

### アプローチ1: 異なる指標の組み合わせ戦略

#### 基本コンセプト
- ロング条件とショート条件で**異なる指標**を使用
- 指標の特性を活かした組み合わせ

#### 実装例
```python
# トレンド + モメンタム組み合わせ
long_conditions = [
    Condition("close", ">", "SMA_20"),      # トレンド：上昇
    Condition("RSI_14", "<", 70)            # モメンタム：過熱していない
]

short_conditions = [
    Condition("close", "<", "EMA_20"),      # トレンド：下降
    Condition("CCI_14", ">", 100)           # モメンタム：買われすぎ
]
```

### アプローチ2: 時間軸分離戦略

#### 基本コンセプト
- 同じ指標でも異なる期間を使用
- 短期と長期の組み合わせ

#### 実装例
```python
# 短期・長期RSI組み合わせ
long_conditions = [
    Condition("RSI_7", "<", 30),            # 短期：売られすぎ
    Condition("RSI_21", ">", 50)            # 長期：上昇トレンド
]

short_conditions = [
    Condition("RSI_7", ">", 70),            # 短期：買われすぎ
    Condition("RSI_21", "<", 50)            # 長期：下降トレンド
]
```

### アプローチ3: 複合条件戦略

#### 基本コンセプト
- 複数の条件を組み合わせて確率を高める
- AND/OR条件の活用

#### 実装例
```python
# ボリンジャーバンド + ADX組み合わせ
long_conditions = [
    Condition("close", "<", "BB_Lower"),    # 価格がバンド下限以下
    Condition("ADX_14", ">", 25),           # 強いトレンド
    Condition("close", ">", "open")         # 上昇方向
]

short_conditions = [
    Condition("close", ">", "BB_Upper"),    # 価格がバンド上限以上
    Condition("ADX_14", ">", 25),           # 強いトレンド
    Condition("close", "<", "open")         # 下降方向
]
```

### アプローチ4: 指標特性活用戦略

#### ボリンジャーバンドの正しい実装
```python
# BBの3つの値を活用
long_conditions = [
    Condition("close", "<", "BB_Lower"),    # 下限突破（逆張り）
    Condition("close", ">", "BB_Middle")    # 中央線回復
]

short_conditions = [
    Condition("close", ">", "BB_Upper"),    # 上限突破（逆張り）
    Condition("close", "<", "BB_Middle")    # 中央線割れ
]
```

#### ADXの正しい活用
```python
# ADX + 方向性指標の組み合わせ
long_conditions = [
    Condition("ADX_14", ">", 25),           # 強いトレンド
    Condition("DI_Plus", ">", "DI_Minus"),  # 上昇方向
    Condition("close", ">", "SMA_20")       # 価格確認
]

short_conditions = [
    Condition("ADX_14", ">", 25),           # 強いトレンド
    Condition("DI_Minus", ">", "DI_Plus"),  # 下降方向
    Condition("close", "<", "SMA_20")       # 価格確認
]
```

## 実装ロードマップ

### フェーズ1: 基盤整備（1-2週間）

#### 1.1 指標データ計算の改善
- [ ] IndicatorCalculator の拡張
- [ ] 複合指標（BB、MACD）の適切な処理
- [ ] 指標データ不足時のエラーハンドリング改善

#### 1.2 条件生成エンジンの設計
- [ ] `SmartConditionGenerator` クラスの作成（`ConditionGenerationEngine`から改名）
- [ ] 指標特性データベースの構築（指標のタイプ、スケール、有効な戦略などを定義）
- [ ] 組み合わせルールの定義（例：トレンド系＋オシレーター系）
- [ ] **責務の集約**: ロング・ショート条件生成ロジックを`SmartConditionGenerator`に完全に集約する。

### フェーズ2: 新条件生成ロジック実装（2-3週間）

#### 2.1 `SmartConditionGenerator` の実装と責務集約
```python
class SmartConditionGenerator:
    def generate_balanced_conditions(self, indicators: List[IndicatorGene]):
        """バランスの取れたロング・ショート条件を生成"""
        
    def _select_indicator_combinations(self, indicators):
        """指標の組み合わせを選択"""
        
    def _generate_complementary_conditions(self, long_indicators, short_indicators):
        """補完的な条件を生成"""
```

#### 2.2 指標特性マッピング
```python
INDICATOR_CHARACTERISTICS = {
    "RSI": {
        "type": "oscillator",
        "range": (0, 100),
        "long_zones": [(0, 30), (40, 60)],
        "short_zones": [(40, 60), (70, 100)],
        "neutral_zone": (40, 60)
    },
    "BB": {
        "type": "volatility_bands",
        "components": ["upper", "middle", "lower"],
        "long_strategy": "mean_reversion",
        "short_strategy": "mean_reversion"
    }
}
```

#### 2.3 組み合わせルール
```python
COMBINATION_RULES = {
    "trend_momentum": {
        "long": ["SMA", "EMA"] + ["RSI", "STOCH"],
        "short": ["SMA", "EMA"] + ["CCI", "RSI"]
    },
    "volatility_trend": {
        "long": ["BB", "ATR"] + ["SMA", "EMA"],
        "short": ["BB", "ATR"] + ["SMA", "EMA"]
    }
}
```

### フェーズ3: 統合とテスト（1-2週間）

#### 3.1 既存システムとの統合
- [ ] **リファクタリング**: `GeneDecoder` と `RandomGeneGenerator` から既存の条件生成ロジックを削除し、`SmartConditionGenerator`の呼び出しに置き換える。
- [ ] **既存機能連携**: `RandomGeneGenerator`の`operand_grouping_system`を`SmartConditionGenerator`で活用し、論理的に整合性の取れた条件を生成する。
- [ ] 後方互換性の確保（設定フラグによる新旧ロジックの切り替え機能など）

#### 3.2 包括的テスト
- [ ] **単体テスト**: `SmartConditionGenerator`が、定義されたルールに基づき、多様かつ論理的な条件の組み合わせを生成することを検証する。
- [ ] **統合テスト**: 戦略生成からバックテストまでの一連のフローが、新しい条件生成ロジックで正常に動作することを確認する。
- [ ] **バランステスト**: 生成された戦略のロング・ショート比率が、目標値（60%以上）に達していることを確認する。
- [ ] **有効性テスト**: `backend/tests/test_long_short_balance.py`に、生成された戦略が実際にロングとショート両方のトレードを実行することを検証するテストケースを追加する。
- [ ] **パフォーマンステスト**: 新しいロジックによる戦略生成速度が、許容範囲内であることを確認する。

### フェーズ4: 最適化と監視（継続）

#### 4.1 パフォーマンス監視
- [ ] 戦略バランス監視ダッシュボード
- [ ] 条件満足率の追跡
- [ ] 指標データ可用性の監視

#### 4.2 継続的改善
- [ ] 新しい指標組み合わせの追加
- [ ] 機械学習による最適化
- [ ] ユーザーフィードバックの反映

## 後方互換性の維持

### 既存APIの保持
- 既存の `generate_random_gene()` メソッドは維持
- 新しい条件生成は内部実装として追加
- 設定フラグによる新旧切り替え機能

### 段階的移行
1. **デフォルト無効**: 新機能はオプトイン
2. **A/Bテスト**: 一部ユーザーで新機能テスト
3. **段階的展開**: 問題なければ全ユーザーに展開
4. **旧機能廃止**: 十分な検証後に旧機能を削除

## 期待される効果

### 定量的改善
- ロング・ショート両対応戦略: 0% → 60%以上
- 条件満足率: 30% → 80%以上
- 指標データエラー: 大幅減少

### 定性的改善
- より多様で実用的な戦略の生成
- ユーザー満足度の向上
- システムの信頼性向上

## リスク管理

### 技術的リスク
- **複雑性増加**: 段階的実装で管理
- **パフォーマンス低下**: プロファイリングと最適化
- **バグ混入**: 包括的テストで防止

### 運用リスク
- **既存戦略への影響**: 後方互換性で対応
- **ユーザー混乱**: 段階的展開で最小化
- **データ品質**: 監視とアラートで対応

## 結論

本計画により、auto-strategy機能のロング・ショートバランス問題を根本的に解決し、より実用的で多様な戦略生成を実現する。段階的な実装により、リスクを最小限に抑えながら確実な改善を図る。
