# 資金管理（ポジションサイジング）メソッド実装計画

## 1. 概要

### 1.1 実装目的

トレーディングシステムのオートストラテジー機能に、完全自動化された資金管理（ポジションサイジング）メソッドを追加する。現在の TP/SL 自動化システムと同様に、遺伝的アルゴリズム（GA）による最適化対象として実装し、ユーザーによる手動設定を最小限に抑制する。

### 1.2 実装対象の資金管理手法

1. **ハーフオプティマル F（Half Optimal F）** - オプティマル F の半分の値を使用する保守的手法
2. **ボラティリティベース（Volatility-based position sizing）** - ATR やボラティリティに基づく動的サイジング
3. **固定比率ベース（Fixed ratio position sizing）** - 口座残高に対する固定比率
4. **枚数ベース（Fixed quantity position sizing）** - 常に固定の枚数/単位数

### 1.3 実装要件

- 完全自動化（ユーザー設定最小限）
- GA 最適化対象として統合
- ロング・ショート両ポジション対応
- 既存の TP/SL 自動化システムとの整合性維持

## 2. 現在のシステム分析

### 2.1 既存の TP/SL 自動化システム

- **TPSLGene**: 5 つの方式（固定パーセンテージ、リスクリワード比、ボラティリティベース、統計的、適応的）
- **StrategyGene**: 戦略遺伝子に`tpsl_gene`として統合済み
- **GA 最適化**: 交叉・突然変異操作が実装済み
- **GAConfig**: パラメータ範囲設定が定義済み

### 2.2 現在のリスク管理実装

- `risk_management`パラメータは主に`position_size`（0.1-0.5 の範囲）のみ
- バックテスト時は固定サイズ（1.0 単位）で実行
- 実際のポジションサイジング計算は非常にシンプル

### 2.3 GA 最適化システム

- **GAConfig**: パラメータ範囲定義、制約設定
- **GeneEncoder**: 遺伝子エンコード/デコード機能
- **交叉・突然変異**: 戦略遺伝子レベルで実装済み

## 3. 資金管理手法の理論的背景

### 3.1 ハーフオプティマル F（Half Optimal F）

**理論**:

- オプティマル F（Kelly Criterion）の半分の値を使用
- オプティマル F = (勝率 × 平均利益 - 敗率 × 平均損失) / 平均利益
- ハーフオプティマル F = オプティマル F / 2

**計算方法**:

```python
def calculate_half_optimal_f(win_rate, avg_win, avg_loss):
    optimal_f = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
    return max(0, optimal_f / 2)  # 負の値は0にクリップ
```

**パラメータ**:

- `lookback_period`: 過去データ参照期間（50-200 日）
- `optimal_f_multiplier`: オプティマル F の倍率（0.25-0.75）

### 3.2 ボラティリティベース（Volatility-based position sizing）

**理論**:

- ATR（Average True Range）やボラティリティ指標を使用
- 市場のボラティリティに応じて動的にサイズ調整
- 高ボラティリティ時は小さく、低ボラティリティ時は大きく

**計算方法**:

```python
def calculate_volatility_based_size(account_balance, risk_per_trade, atr, atr_multiplier):
    risk_amount = account_balance * risk_per_trade
    position_size = risk_amount / (atr * atr_multiplier)
    return position_size
```

**パラメータ**:

- `atr_period`: ATR 計算期間（10-30 日）
- `atr_multiplier`: ATR に対する倍率（1.0-4.0）
- `risk_per_trade`: 1 取引あたりのリスク（1%-5%）

### 3.3 固定比率ベース（Fixed ratio position sizing）

**理論**:

- 口座残高に対する固定比率でポジションサイズを決定
- シンプルで理解しやすく、複利効果を活用

**計算方法**:

```python
def calculate_fixed_ratio_size(account_balance, fixed_ratio):
    return account_balance * fixed_ratio
```

**パラメータ**:

- `fixed_ratio`: 口座残高に対する比率（5%-30%）

### 3.4 枚数ベース（Fixed quantity position sizing）

**理論**:

- 常に固定の枚数/単位数で取引
- 最もシンプルな手法
- 口座残高の変動に関係なく一定

**計算方法**:

```python
def calculate_fixed_quantity_size(fixed_quantity):
    return fixed_quantity
```

**パラメータ**:

- `fixed_quantity`: 固定枚数（0.1-5.0 単位）

## 4. 設計仕様

### 4.1 PositionSizingGene 設計

```python
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any

class PositionSizingMethod(Enum):
    HALF_OPTIMAL_F = "half_optimal_f"
    VOLATILITY_BASED = "volatility_based"
    FIXED_RATIO = "fixed_ratio"
    FIXED_QUANTITY = "fixed_quantity"

@dataclass
class PositionSizingGene:
    # ポジションサイジング方式
    method: PositionSizingMethod = PositionSizingMethod.FIXED_RATIO

    # ハーフオプティマルF方式のパラメータ
    lookback_period: int = 100
    optimal_f_multiplier: float = 0.5

    # ボラティリティベース方式のパラメータ
    atr_period: int = 14
    atr_multiplier: float = 2.0
    risk_per_trade: float = 0.02

    # 固定比率ベース方式のパラメータ
    fixed_ratio: float = 0.1

    # 枚数ベース方式のパラメータ
    fixed_quantity: float = 1.0

    # 共通パラメータ
    min_position_size: float = 0.01
    max_position_size: float = 1.0
    enabled: bool = True
    priority: float = 1.0
```

### 4.2 StrategyGene への統合

```python
@dataclass
class StrategyGene:
    # 既存フィールド...
    tpsl_gene: Optional[TPSLGene] = None
    position_sizing_gene: Optional[PositionSizingGene] = None  # 新規追加
```

### 4.3 GAConfig への追加パラメータ

```python
# ポジションサイジングGA最適化範囲設定
position_sizing_method_constraints: List[str] = [
    "half_optimal_f", "volatility_based", "fixed_ratio", "fixed_quantity"
]
position_sizing_lookback_range: List[int] = [50, 200]
position_sizing_optimal_f_multiplier_range: List[float] = [0.25, 0.75]
position_sizing_atr_period_range: List[int] = [10, 30]
position_sizing_atr_multiplier_range: List[float] = [1.0, 4.0]
position_sizing_risk_per_trade_range: List[float] = [0.01, 0.05]
position_sizing_fixed_ratio_range: List[float] = [0.05, 0.3]
position_sizing_fixed_quantity_range: List[float] = [0.1, 5.0]
```

## 5. 実装計画

### 5.1 フェーズ 1: 基盤実装

**目標**: PositionSizingGene の基本構造を実装

**実装内容**:

1. `backend/app/core/services/auto_strategy/models/position_sizing_gene.py`作成

   - `PositionSizingMethod`列挙型定義
   - `PositionSizingGene`データクラス実装
   - 基本的なバリデーション機能
   - 辞書変換機能（`to_dict()`, `from_dict()`）

2. 基本的な計算ロジック実装
   - 各手法の基本計算関数
   - エラーハンドリング
   - フォールバック機能

**成果物**:

- `PositionSizingGene`クラス
- 基本計算関数群
- 単体テスト

### 5.2 フェーズ 2: 計算サービス実装

**目標**: 実際のポジションサイズ計算機能を実装

**実装内容**:

1. `backend/app/core/services/auto_strategy/calculators/position_sizing_calculator.py`作成

   - `PositionSizingCalculatorService`クラス
   - 各手法の詳細計算ロジック
   - 市場データ統合機能
   - パフォーマンス最適化

2. 市場データ統合
   - ATR 計算機能
   - 過去データ取得・分析
   - キャッシュ機能

**成果物**:

- `PositionSizingCalculatorService`
- 市場データ統合機能
- 計算結果検証機能

### 5.3 フェーズ 3: GA 統合

**目標**: 遺伝的アルゴリズムによる最適化機能を実装

**実装内容**:

1. `GAConfig`への新パラメータ追加

   - ポジションサイジング関連の制約設定
   - パラメータ範囲定義

2. `GeneEncoder`への統合

   - `PositionSizingGene`のエンコード/デコード機能
   - 既存の`TPSLGene`と同様の実装パターン

3. 交叉・突然変異操作実装
   - `crossover_position_sizing_genes()`関数
   - `mutate_position_sizing_gene()`関数

**成果物**:

- GA 統合機能
- 交叉・突然変異操作
- パラメータ最適化機能

### 5.4 フェーズ 4: StrategyGene 統合

**目標**: 戦略遺伝子レベルでの統合を完了

**実装内容**:

1. `StrategyGene`への`position_sizing_gene`追加
2. 戦略遺伝子レベルの操作更新
   - 交叉操作での`position_sizing_gene`処理
   - 突然変異操作での処理
3. バリデーション機能追加
   - 遺伝子整合性チェック
   - パラメータ妥当性検証

**成果物**:

- 統合された`StrategyGene`
- 完全な GA 操作機能
- バリデーション機能

### 5.5 フェーズ 5: バックテスト統合

**目標**: 実際のバックテストでの動的ポジションサイズ計算

**実装内容**:

1. `StrategyFactory`での実装

   - 動的ポジションサイズ計算
   - 既存の固定サイズからの移行
   - `backtesting.py`ライブラリとの互換性確保

2. パフォーマンス最適化
   - 計算効率の改善
   - メモリ使用量の最適化
   - 並列処理対応

**成果物**:

- 動的ポジションサイズ機能
- 最適化されたバックテスト
- パフォーマンス改善

### 5.6 フェーズ 6: テスト・検証

**目標**: 包括的なテストと検証を実施

**実装内容**:

1. 単体テスト作成

   - 各計算手法のテスト
   - エラーケースのテスト
   - パフォーマンステスト

2. 統合テスト実装

   - GA 最適化テスト
   - バックテスト統合テスト
   - エンドツーエンドテスト

3. 検証・調整
   - 実際のデータでの検証
   - パラメータ調整
   - パフォーマンス最適化

**成果物**:

- 完全なテストスイート
- 検証済みの実装
- パフォーマンス最適化

## 6. 技術的考慮事項

### 6.1 パフォーマンス最適化

- **過去データ計算の効率化**: ハーフオプティマル F 計算時の大量データ処理
- **キャッシュ機能**: ATR やボラティリティ計算結果のキャッシュ
- **並列処理**: GA 実行時の並列計算対応
- **メモリ管理**: 大量の履歴データ処理時のメモリ効率

### 6.2 エラーハンドリング

- **計算エラー時のフォールバック**: 無効なパラメータや計算エラー時の対応
- **データ不足時の処理**: 過去データが不十分な場合の対応
- **パラメータ検証**: 無効な範囲のパラメータの検出と修正
- **ログ出力**: デバッグ用の詳細ログ機能

### 6.3 既存システムとの整合性

- **TP/SL との相互作用**: ポジションサイズと TP/SL 設定の整合性
- **リスク管理統合**: 既存の`risk_management`パラメータとの統合
- **バックテスト互換性**: `backtesting.py`ライブラリとの互換性維持
- **フロントエンド統合**: 既存 UI との整合性

### 6.4 データ要件

- **市場データ**: ATR 計算用の価格データ
- **履歴データ**: ハーフオプティマル F 計算用の取引履歴
- **リアルタイムデータ**: 動的計算用の現在価格
- **データ品質**: 欠損データや異常値の処理

## 7. テスト戦略

### 7.1 単体テスト

- **計算ロジックテスト**: 各手法の計算精度検証
- **エラーケーステスト**: 異常入力時の動作検証
- **パフォーマンステスト**: 計算速度とメモリ使用量測定
- **境界値テスト**: パラメータ範囲の境界値での動作確認

### 7.2 統合テスト

- **GA 統合テスト**: 遺伝的アルゴリズムでの最適化動作確認
- **バックテスト統合テスト**: 実際のバックテストでの動作確認
- **エンドツーエンドテスト**: フロントエンドからバックエンドまでの完全な動作確認
- **パフォーマンス統合テスト**: システム全体でのパフォーマンス測定

### 7.3 検証テスト

- **実データ検証**: 実際の市場データでの動作確認
- **比較検証**: 既存手法との結果比較
- **ストレステスト**: 極端な市場条件での動作確認
- **長期間テスト**: 長期間のバックテストでの安定性確認

## 8. リスク評価と対策

### 8.1 実装リスク

**リスク**: 既存システムとの互換性問題
**対策**: 段階的実装、既存機能の保持、十分なテスト

**リスク**: パフォーマンス劣化
**対策**: プロファイリング、最適化、並列処理

**リスク**: 計算精度の問題
**対策**: 数値計算の検証、エラーハンドリング強化

### 8.2 運用リスク

**リスク**: 過度なポジションサイズによる損失拡大
**対策**: 最大ポジションサイズ制限、リスク管理機能強化

**リスク**: 市場データ依存による計算エラー
**対策**: データ品質チェック、フォールバック機能

**リスク**: GA 最適化による過学習
**対策**: 検証データでの評価、制約条件の設定

### 8.3 保守性リスク

**リスク**: コードの複雑化
**対策**: 適切な設計パターン、ドキュメント整備

**リスク**: テスト保守の困難
**対策**: 自動テスト、継続的インテグレーション

## 9. 今後の拡張可能性（ここはまだ実装しない予定です）

### 9.1 追加手法の実装

- **マーチンゲール法**: 損失後のポジション増加
- **アンチマーチンゲール法**: 利益後のポジション増加
- **フラクショナルケリー**: ケリー基準の分数版

### 9.2 高度な機能とデータ駆動型アプローチ

- **統計的リスクモデルの導入 (VaR, CVaR)**

  - **Value at Risk (VaR)**: ポートフォリオ全体のリスクを統計的に評価し、特定の信頼区間で想定される最大損失額に基づいてサイズを決定する。
  - **Conditional Value at Risk (CVaR)**: VaR を超えるような、深刻な（テールリスク）事態における平均損失を考慮し、より保守的なリスク管理を実現する。

- **機械学習による動的ポジションサイジング**

  - **強化学習 (Reinforcement Learning)**: AI エージェントがシミュレーション環境内で試行錯誤を繰り返し、長期的なリターンを最大化する最適なサイジング戦略を自律的に学習する。
  - **教師あり学習 (Supervised Learning)**: 過去の市場データから、特定のパターンと最適なポジションサイズの間の関係性を学習し、未来のサイズを予測する。

- **動的リバランスとポートフォリオ管理**
  - **動的リバランス**: 市場の状況（ボラティリティ、流動性など）に応じて、最適な資金管理手法そのものを動的に切り替える。
  - **ポートフォリオレベル管理**: 複数の戦略や資産を横断して、全体のリスクが最適になるようにポジションサイズを調整する。

### 9.3 ユーザーインターフェース強化

- **視覚化機能**: ポジションサイズの推移グラフ
- **シミュレーション機能**: 手法比較シミュレーション
- **レポート機能**: 詳細な分析レポート

---

## 実装開始準備

この実装計画に基づき、フェーズ 1 から順次実装を開始します。各フェーズの完了後に検証を行い、次のフェーズに進む前に品質を確保します。

実装時は既存の TP/SL 自動化システムの実装パターンを参考にし、一貫性のある設計を維持します。また、ユーザーによる手動設定を最小限に抑制し、GA 最適化による完全自動化を実現します。
