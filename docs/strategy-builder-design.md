# ストラテジービルダー機能 設計計画書

## 概要

ユーザーが既存の 58 種類のテクニカル指標を組み合わせて、手動で独自の投資戦略を作成できる「ストラテジービルダー」機能の設計計画書です。

## 現状分析

### 既存システム

- **テクニカル指標**: 58 種類実装済み（trend、momentum、volatility、volume、price_transform）
- **指標管理**: IndicatorConfig クラスと TechnicalIndicatorService による統一管理
- **戦略システム**: BaseStrategy、SMACrossStrategy、RSIStrategy 等の実装
- **バックテスト**: BacktestService による backtesting.py 統合
- **オートストラテジー**: GA 使用の自動戦略生成（10 戦略限定）
- **フロントエンド**: Next.js/TypeScript/React によるモダンな実装

### 技術スタック

- **バックエンド**: Python/FastAPI
- **フロントエンド**: Next.js/TypeScript/React
- **データベース**: SQLAlchemy ORM
- **バックテスト**: backtesting.py
- **テクニカル指標**: TA-Lib

## 機能要件

### 1. 基本機能

- **指標選択**: 58 種類のテクニカル指標から複数選択
- **パラメータ設定**: 各指標のパラメータを GUI で設定
- **条件設定**: 買い/売りの条件をビジュアルに設定
- **戦略保存**: 作成した戦略を保存・管理
- **バックテスト実行**: 作成した戦略でバックテスト実行
- **プレビュー機能**: 戦略ロジックの可視化

### 2. 対象指標カテゴリ

#### トレンド系指標 (14 種類)

- SMA, EMA, MACD, KAMA, T3, TEMA, DEMA, WMA, HMA, VWMA, ZLEMA, MIDPOINT, MIDPRICE, TRIMA

#### モメンタム系指標 (19 種類)

- RSI, Stochastic, CCI, Williams%R, Momentum, ROC, ADX, Aroon, MFI, StochasticRSI, UltimateOscillator, BOP, PPO, PLUSDI, MINUSDI, ROCP, ROCR, STOCHF, CMO

#### ボラティリティ系指標 (8 種類)

- ATR, NATR, TRANGE, BB (Bollinger Bands), STDDEV, VAR, BETA, CORREL

#### ボリューム系指標 (3 種類)

- OBV, AD, ADOSC

#### 価格変換系指標 (14 種類)

- AVGPRICE, MEDPRICE, TYPPRICE, WCLPRICE, HT_DCPERIOD, HT_DCPHASE, HT_PHASOR, HT_SINE, HT_TRENDMODE, MAMA, FAMA, SAREXT, SAR, APO

## アーキテクチャ設計

### 1. システム構成

```text
Frontend (Next.js/React)
├── /strategy-builder (新規ページ)
├── IndicatorSelector (指標選択コンポーネント)
├── ParameterEditor (パラメータ設定)
├── ConditionBuilder (条件設定)
├── StrategyPreview (プレビュー)
└── SavedStrategies (保存済み戦略管理)

Backend (FastAPI)
├── /api/strategy-builder/* (新規APIエンドポイント)
├── StrategyBuilderService (新規: ユーザー戦略管理)
├── UserStrategyRepository (新規: 戦略保存)
├── StrategyFactory (既存活用: 動的戦略生成)
├── TechnicalIndicatorService (既存活用: 指標管理)
└── BacktestService (拡張: USER_CUSTOM戦略タイプ追加)

Database
└── user_strategies (ユーザー戦略テーブル)
```

### 2. データモデル設計

#### UserStrategy テーブル

```sql
CREATE TABLE user_strategies (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    strategy_config JSONB NOT NULL,  -- StrategyGene形式
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE
);
```

#### 戦略設定 JSON 構造（StrategyGene 準拠）

```json
{
  "id": "user_strategy_001",
  "indicators": [
    {
      "type": "SMA",
      "parameters": { "period": 20 },
      "enabled": true,
      "json_config": {
        "indicator_name": "SMA",
        "parameters": { "period": 20 }
      }
    },
    {
      "type": "SMA",
      "parameters": { "period": 50 },
      "enabled": true,
      "json_config": {
        "indicator_name": "SMA",
        "parameters": { "period": 50 }
      }
    },
    {
      "type": "RSI",
      "parameters": { "period": 14 },
      "enabled": true,
      "json_config": {
        "indicator_name": "RSI",
        "parameters": { "period": 14 }
      }
    }
  ],
  "entry_conditions": [
    {
      "type": "crossover",
      "indicator1": "SMA_0",
      "indicator2": "SMA_1",
      "operator": "above"
    },
    {
      "type": "threshold",
      "indicator": "RSI_0",
      "operator": "less_than",
      "value": 70
    }
  ],
  "exit_conditions": [
    {
      "type": "crossover",
      "indicator1": "SMA_0",
      "indicator2": "SMA_1",
      "operator": "below"
    }
  ],
  "risk_management": {
    "stop_loss_pct": 0.02,
    "take_profit_pct": 0.05,
    "position_sizing": "fixed"
  },
  "metadata": {
    "created_by": "strategy_builder",
    "version": "1.0"
  }
}
```

## API 設計

### 1. エンドポイント一覧

```http
GET    /api/strategy-builder/indicators          # 利用可能指標一覧
POST   /api/strategy-builder/validate           # 戦略検証
POST   /api/strategy-builder/preview            # 戦略プレビュー
POST   /api/strategy-builder/save               # 戦略保存
GET    /api/strategy-builder/strategies         # 保存済み戦略一覧
GET    /api/strategy-builder/strategies/{id}    # 戦略詳細取得
PUT    /api/strategy-builder/strategies/{id}    # 戦略更新
DELETE /api/strategy-builder/strategies/{id}    # 戦略削除

# 既存エンドポイント拡張
POST   /api/backtest/run                        # USER_CUSTOM戦略タイプ対応
```

### 2. レスポンス例

#### 指標一覧取得

```json
{
  "success": true,
  "data": {
    "categories": {
      "trend": [
        {
          "type": "SMA",
          "name": "Simple Moving Average",
          "description": "単純移動平均",
          "parameters": [
            {
              "name": "period",
              "type": "integer",
              "default": 20,
              "min": 2,
              "max": 200,
              "description": "移動平均期間"
            }
          ],
          "data_sources": ["close", "open", "high", "low"]
        }
      ]
    }
  }
}
```

## UI/UX 設計

### 1. ページ構成

#### メインページ: `/strategy-builder`

```text
┌─────────────────────────────────────────────┐
│ ストラテジービルダー                          │
├─────────────────────────────────────────────┤
│ [新規作成] [保存済み戦略] [テンプレート]      │
├─────────────────────────────────────────────┤
│ ステップ1: 指標選択                          │
│ ┌─────────────────────────────────────────┐ │
│ │ [トレンド] [モメンタム] [ボラティリティ] │ │
│ │ [ボリューム] [価格変換]                 │ │
│ │                                       │ │
│ │ ○ SMA (Simple Moving Average)         │ │
│ │ ○ EMA (Exponential Moving Average)    │ │
│ │ ○ RSI (Relative Strength Index)      │ │
│ └─────────────────────────────────────────┘ │
├─────────────────────────────────────────────┤
│ ステップ2: パラメータ設定                    │
│ ┌─────────────────────────────────────────┐ │
│ │ SMA_SHORT: 期間 [20] データソース[Close]│ │
│ │ SMA_LONG:  期間 [50] データソース[Close]│ │
│ │ RSI:       期間 [14] データソース[Close]│ │
│ └─────────────────────────────────────────┘ │
├─────────────────────────────────────────────┤
│ ステップ3: 条件設定                          │
│ ┌─────────────────────────────────────────┐ │
│ │ 買い条件:                              │ │
│ │ - SMA_SHORT が SMA_LONG を上抜け        │ │
│ │ - RSI < 70                            │ │
│ │                                       │ │
│ │ 売り条件:                              │ │
│ │ - SMA_SHORT が SMA_LONG を下抜け        │ │
│ └─────────────────────────────────────────┘ │
├─────────────────────────────────────────────┤
│ [プレビュー] [保存] [バックテスト実行]        │
└─────────────────────────────────────────────┘
```

### 2. コンポーネント設計（既存 UI パターン準拠）

#### IndicatorSelector

- `TabButton` によるカテゴリ別タブ表示
- `InputField` による検索・フィルタ機能
- 選択済み指標の表示（`enterprise-card` スタイル）

#### ParameterEditor

- `InputField`, `SelectField` による統一フォーム
- リアルタイムバリデーション
- デフォルト値の自動設定

#### ConditionBuilder

- `SelectField` による条件タイプ選択
- 論理演算子（AND/OR）の設定
- 条件の可視化（`enterprise-card` レイアウト）

#### StrategyPreview

- `Modal` による戦略プレビュー表示
- 生成される StrategyGene 構造のプレビュー
- 設定サマリー（`TabButton` による切り替え）

#### SavedStrategies

- 既存の戦略一覧表示パターンを活用
- `ApiButton` による操作ボタン
- `Modal` による詳細表示・編集

## 実装計画

### Phase 1: MVP 開発 (4 週間)

#### Week 1: バックエンド基盤

- [ ] データベーススキーマ設計・作成
- [ ] 基本 API エンドポイント実装
- [ ] 動的戦略生成エンジン開発

#### Week 2: フロントエンド基盤

- [ ] ストラテジービルダーページ作成
- [ ] 基本 UI コンポーネント実装
- [ ] 指標選択機能

#### Week 3: 機能統合

- [ ] パラメータ設定機能
- [ ] 条件設定機能
- [ ] 戦略保存機能

#### Week 4: テスト・最適化

- [ ] バックテスト統合
- [ ] エラーハンドリング
- [ ] パフォーマンス最適化

### Phase 2: 拡張機能 (4 週間)

#### Week 5-6: 高度な条件設定

- [ ] 複雑な論理演算
- [ ] カスタム条件式
- [ ] 条件の可視化強化

#### Week 7-8: UI/UX 改善

- [ ] ドラッグ&ドロップインターフェース
- [ ] リアルタイムプレビュー
- [ ] 戦略テンプレート

## 技術的考慮事項

### 1. 既存システムとの統合

- `StrategyFactory` による動的戦略生成の活用
- `TechnicalIndicatorService` による指標管理の統合
- `BacktestService` への新戦略タイプ追加
- 既存の `StrategyGene` 形式との互換性確保

### 2. パフォーマンス

- 指標計算の最適化（既存の TA-Lib アダプター活用）
- フロントエンドでの状態管理（React hooks 活用）
- API レスポンス時間の最適化

### 3. セキュリティ

- 入力値のバリデーション（既存の `APIErrorHandler` パターン活用）
- SQL インジェクション対策
- 戦略設定の検証（既存の `GeneValidator` 活用）

### 4. 拡張性

- 新しい指標の追加容易性（既存の指標登録システム活用）
- 条件タイプの拡張性
- 他システムとの連携

### 5. ユーザビリティ

- 既存 UI パターンとの統一性
- エラーメッセージの分かりやすさ
- ヘルプ・ドキュメント

## 成功指標

### 1. 機能指標

- 戦略作成完了率: 80%以上
- バックテスト実行成功率: 95%以上
- 戦略保存・読み込み成功率: 99%以上

### 2. パフォーマンス指標

- ページ読み込み時間: 3 秒以内
- 戦略生成時間: 5 秒以内
- バックテスト実行時間: 30 秒以内

### 3. ユーザビリティ指標

- 戦略作成時間: 平均 10 分以内
- エラー発生率: 5%以下
- ユーザー満足度: 4.0/5.0 以上

## リスクと対策

### 1. 技術的リスク

- **複雑な戦略の処理性能**: キャッシュ機能とバックグラウンド処理
- **指標計算エラー**: 堅牢なエラーハンドリングと検証機能
- **データベース負荷**: インデックス最適化とクエリ改善

### 2. ユーザビリティリスク

- **操作の複雑さ**: ステップバイステップのガイダンス
- **設定ミス**: リアルタイムバリデーションと警告表示
- **学習コスト**: チュートリアルとヘルプ機能

### 3. ビジネスリスク

- **開発期間の延長**: 段階的リリースと MVP 優先
- **既存機能への影響**: 十分なテストと段階的統合
- **ユーザー採用率**: ベータテストとフィードバック収集

## 次のステップ

1. **要件確認**: ステークホルダーとの詳細要件確認
2. **技術検証**: プロトタイプ開発による技術的実現性確認
3. **UI/UX デザイン**: 詳細なワイヤーフレームとデザイン作成
4. **開発開始**: Phase 1 の MVP 開発着手

---

## 修正履歴

### v1.1 (2025-01-27)

- 既存システムとの統合を重視した設計に修正
- `StrategyFactory` と `StrategyGene` 形式の活用
- データベース設計の簡素化（`user_strategies` テーブルのみ）
- 既存 UI パターン（`TabButton`, `Modal`, `InputField`等）の準拠
- `BacktestService` への `USER_CUSTOM` 戦略タイプ追加

---

_この設計計画書は、既存システムとの整合性を保ちながら、ユーザーフレンドリーなストラテジービルダー機能を実現するための包括的な計画を提供します。既存の 58 種類のテクニカル指標と動的戦略生成システムを最大限活用し、効率的な実装を可能にします。_
