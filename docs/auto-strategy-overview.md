# オートストラテジー機能 概要

## 目次
1. [概要](#概要)
2. [システム構成](#システム構成)
3. [主要コンポーネント](#主要コンポーネント)
4. [データフロー](#データフロー)
5. [API エンドポイント](#api-エンドポイント)
6. [フロントエンド実装](#フロントエンド実装)
7. [データベース構造](#データベース構造)
8. [技術的詳細](#技術的詳細)

## 概要

オートストラテジー機能は、**遺伝的アルゴリズム（GA）**を使用して取引戦略を自動生成するシステムです。ユーザーが設定したパラメータに基づいて、複数のテクニカル指標を組み合わせた最適な取引戦略を進化的に生成します。

### 主な特徴
- **遺伝的アルゴリズム**による戦略最適化
- **50種類以上のテクニカル指標**をサポート
- **バックグラウンド実行**による非同期処理
- **リアルタイム進捗監視**
- **戦略ショーケース**機能
- **詳細なパフォーマンス分析**

## システム構成

```
オートストラテジー機能
├── バックエンド（Python/FastAPI）
│   ├── API エンドポイント
│   ├── GA エンジン
│   ├── 戦略ファクトリー
│   ├── バックテストサービス
│   └── データベース管理
├── フロントエンド（Next.js/React）
│   ├── 戦略ショーケースページ
│   ├── GA 設定フォーム
│   ├── 進捗監視UI
│   └── 結果表示コンポーネント
└── データベース（PostgreSQL/TimescaleDB）
    ├── GA実験テーブル
    ├── 生成戦略テーブル
    ├── バックテスト結果テーブル
    └── 戦略ショーケーステーブル
```

## 主要コンポーネント

### 1. GA エンジン (`ga_engine.py`)
- **DEAP ライブラリ**を使用した遺伝的アルゴリズム実装
- 個体群の生成、評価、選択、交叉、突然変異を管理
- 並列処理による高速化対応

### 2. 戦略ファクトリー (`strategy_factory.py`)
- 戦略遺伝子から実際の取引戦略クラスを動的生成
- **backtesting.py** 互換のStrategy継承クラスを作成
- 50種類以上のテクニカル指標をサポート

### 3. 戦略遺伝子 (`strategy_gene.py`)
- 戦略の遺伝的表現を定義
- 指標、エントリー/エグジット条件、リスク管理設定を含む
- JSON形式でのシリアライゼーション対応

### 4. バックテストサービス
- 生成された戦略のパフォーマンス評価
- OHLCV、オープンインタレスト、ファンディングレートデータを使用
- 詳細なメトリクス計算（シャープレシオ、最大ドローダウンなど）

## データフロー

### 戦略生成プロセス
```
1. ユーザー設定入力
   ↓
2. GA設定の検証
   ↓
3. 初期個体群生成（ランダム戦略遺伝子）
   ↓
4. 各個体の評価（バックテスト実行）
   ↓
5. フィットネス計算
   ↓
6. 選択・交叉・突然変異
   ↓
7. 新世代の生成
   ↓
8. 収束条件まで4-7を繰り返し
   ↓
9. 最適戦略の保存
```

### フィットネス評価
```python
fitness = (
    total_return * 0.3 +
    sharpe_ratio * 0.4 +
    (1 - max_drawdown) * 0.2 +
    win_rate * 0.1
)
```

## API エンドポイント

### 1. 戦略生成 API (`/api/auto-strategy`)

#### `POST /api/auto-strategy/generate`
- GA戦略生成を開始
- バックグラウンド実行
- 実験IDを返却

#### `GET /api/auto-strategy/progress/{experiment_id}`
- 実験の進捗状況を取得
- リアルタイム監視用

#### `GET /api/auto-strategy/result/{experiment_id}`
- 実験結果の取得
- 最適戦略と詳細メトリクス

#### `POST /api/auto-strategy/test-strategy`
- 単一戦略のテスト実行
- GA実行前の検証用

### 2. 戦略ショーケース API (`/api/strategies/showcase`)

#### `POST /api/strategies/showcase/generate`
- ショーケース用戦略30個を自動生成
- 多様な戦略カテゴリを作成

#### `GET /api/strategies/showcase/list`
- 生成済み戦略の一覧取得
- フィルタリング・ソート機能

## フロントエンド実装

### 1. 戦略ショーケースページ (`/strategies`)
- **戦略カード表示**: 各戦略の要約情報
- **フィルタリング機能**: カテゴリ、リスクレベル、パフォーマンス
- **詳細モーダル**: 戦略の詳細情報表示
- **戦略生成ボタン**: 新しい戦略セットの生成

### 2. GA設定フォーム (`GAConfigForm.tsx`)
- **基本設定**: 個体数、世代数、交叉率、突然変異率
- **指標選択**: 使用する技術指標の選択
- **フィットネス重み**: 評価指標の重み付け設定
- **制約条件**: 最小取引数、最大ドローダウン制限

### 3. 主要コンポーネント
- `StrategyCard`: 戦略の要約表示
- `StrategyFilters`: フィルタリング・ソート機能
- `StrategyModal`: 戦略詳細表示
- `GAConfigForm`: GA設定入力フォーム

## データベース構造

### 1. GA実験テーブル (`ga_experiments`)
```sql
- id: 実験ID
- name: 実験名
- config: GA設定（JSON）
- status: 実行状態
- progress: 進捗率
- best_fitness: 最高フィットネス
- total_generations: 総世代数
- current_generation: 現在の世代数
- created_at, completed_at: タイムスタンプ
```

### 2. 生成戦略テーブル (`generated_strategies`)
```sql
- id: 戦略ID
- experiment_id: 実験ID（外部キー）
- gene_data: 戦略遺伝子データ（JSON）
- generation: 世代数
- fitness_score: フィットネススコア
- parent_ids: 親戦略のID（JSON配列）
- backtest_result_id: バックテスト結果ID
- created_at: 作成日時
```

### 3. 戦略ショーケーステーブル (`strategy_showcase`)
```sql
- id: ショーケースID
- name: 戦略名
- category: 戦略カテゴリ
- description: 説明
- expected_return: 期待リターン
- sharpe_ratio: シャープレシオ
- max_drawdown: 最大ドローダウン
- win_rate: 勝率
- gene_data: 戦略遺伝子データ（JSON）
- risk_level: リスクレベル
- is_active: アクティブフラグ
```

## 技術的詳細

### 使用可能なテクニカル指標（50種類以上）

#### トレンド系
- SMA, EMA, WMA, HMA, KAMA, TEMA, DEMA, ZLEMA, TRIMA
- MIDPOINT, MIDPRICE, T3

#### モメンタム系
- RSI, MACD, STOCH, STOCHF, STOCHRSI
- MOMENTUM, ROC, ROCP, ROCR
- CCI, WILLIAMS, MFI, CMO, TRIX
- AROON, AROONOSC, PPO, ULTOSC

#### ボラティリティ系
- BB (Bollinger Bands), ATR, NATR, TRANGE
- STDDEV, DONCHIAN, KELTNER

#### ボリューム系
- VWMA, VWAP, AD, ADOSC, OBV, EMV

#### その他
- ADX, ADXR, DX, PLUS_DI, MINUS_DI
- BOP, SAR, HT_TRENDLINE

### 戦略カテゴリ
1. **トレンドフォロー**: トレンドに追従する戦略
2. **逆張り**: 平均回帰を狙う戦略
3. **ブレイクアウト**: 価格突破を狙う戦略
4. **レンジ取引**: レンジ相場での取引戦略
5. **モメンタム**: 勢いに基づく戦略

### パフォーマンス最適化
- **並列処理**: 複数プロセスでの個体評価
- **バックグラウンド実行**: 非同期タスク処理
- **データベース最適化**: インデックス設定
- **メモリ効率**: 大量データの効率的処理

### エラーハンドリング
- **戦略検証**: 無効な戦略遺伝子の検出
- **バックテスト失敗**: エラー時のフォールバック処理
- **リソース管理**: メモリ・CPU使用量の監視
- **ログ記録**: 詳細なデバッグ情報

## 今後の拡張予定
- **機械学習統合**: より高度な最適化手法
- **リアルタイム取引**: 生成戦略の自動実行
- **マルチアセット対応**: 複数銘柄での戦略生成
- **カスタム指標**: ユーザー定義指標の追加
