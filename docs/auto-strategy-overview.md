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
- **50 種類以上のテクニカル指標**をサポート
- **バックグラウンド実行**による非同期処理
- **リアルタイム進捗監視**
- **戦略ショーケース**機能
- **詳細なパフォーマンス分析**
- **高度なリスク管理機能**（ケリー基準、ATR ベース SL/TP など）

## システム構成

```
オートストラテジー機能
├── バックエンド（Python/FastAPI）
│   ├── API エンドポイント
│   ├── GA エンジン
│   ├── 戦略ファクトリー
│   ├── バックテストサービス
│   ├── リスク管理モジュール
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
    ├── 戦略ショーケーステーブル
    ├── ファンディングレートテーブル
    └── オープンインタレストテーブル
```

## 主要コンポーネント

### 1. GA エンジン (`ga_engine.py`)

- **DEAP ライブラリ**を使用した遺伝的アルゴリズム実装
- 個体群の生成、評価、選択、交叉、突然変異を管理
- 並列処理による高速化対応

### 2. 戦略ファクトリー (`strategy_factory.py`)

- 戦略遺伝子から実際の取引戦略クラスを動的生成
- **backtesting.py** 互換の Strategy 継承クラスを作成
- 50 種類以上のテクニカル指標をサポート

### 3. 戦略遺伝子 (`strategy_gene.py`)

- 戦略の遺伝的表現を定義
- 指標、エントリー/エグジット条件、リスク管理設定を含む
- JSON 形式でのシリアライゼーション対応

### 4. バックテストサービス

- 生成された戦略のパフォーマンス評価
- OHLCV、オープンインタレスト、ファンディングレートデータを使用
- 詳細なメトリクス計算（シャープレシオ、最大ドローダウンなど）

### 5. 高度なリスク管理モジュール (`risk_management.py`)

- **ケリー基準**、**リスクリワード比率**に基づくポジションサイジング
- **ATR（Average True Range）**ベースの動的ストップロス・テイクプロフィット
- 取引履歴を考慮した適応的リスク調整

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

### 1. オートストラテジー API (`/api/auto-strategy`)

#### `POST /generate`

- GA 戦略生成を開始
- バックグラウンド実行
- 実験 ID を返却

#### `GET /experiments/{experiment_id}/progress`

- 実験の進捗状況を取得
- リアルタイム監視用

#### `GET /experiments/{experiment_id}/results`

- 実験結果の取得
- 最適戦略と詳細メトリクス

#### `POST /test-strategy`

- 単一戦略のテスト実行
- GA 実行前の検証用

### 2. 戦略ショーケース API (`/api/strategies/showcase`)

#### `POST /generate`

- ショーケース用戦略 30 個を自動生成
- 多様な戦略カテゴリを作成

#### `GET /list`

- 生成済み戦略の一覧取得
- フィルタリング・ソート機能

### 3. 高度なバックテスト API (`/api/backtest`)

#### `POST /run`

- 単一のバックテストを実行

#### `POST /optimize`

- 戦略パラメータの最適化

#### `POST /optimize-enhanced`

- 制約条件付きの高度な最適化

#### `POST /multi-objective-optimization`

- 複数目的（リターン、リスクなど）の同時最適化

#### `POST /robustness-test`

- 戦略の堅牢性（ロバストネス）を複数期間でテスト

## フロントエンド実装

### 1. 戦略ショーケースページ (`/strategies`)

- **戦略カード表示**: 各戦略の要約情報
- **フィルタリング機能**: カテゴリ、リスクレベル、パフォーマンス
- **詳細モーダル**: 戦略の詳細情報表示
- **戦略生成ボタン**: 新しい戦略セットの生成

### 2. GA 設定フォーム (`GAConfigForm.tsx`)

- **基本設定**: 個体数、世代数、交叉率、突然変異率
- **指標選択**: 使用する技術指標の選択
- **フィットネス重み**: 評価指標の重み付け設定
- **制約条件**: 最小取引数、最大ドローダウン制限

### 3. 主要コンポーネント

- `StrategyCard`: 戦略の要約表示
- `StrategyFilters`: フィルタリング・ソート機能
- `StrategyModal`: 戦略詳細表示
- `GAConfigForm`: GA 設定入力フォーム

## データベース構造

### 1. GA 実験テーブル (`ga_experiments`)

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

### 3. バックテスト結果テーブル (`backtest_results`)

```sql
- id: バックテスト結果ID
- strategy_name: 戦略名
- symbol, timeframe: 取引ペア、時間軸
- start_date, end_date: バックテスト期間
- initial_capital, commission_rate: 初期資金、手数料
- config_json: 戦略設定（JSON）
- performance_metrics: パフォーマンス指標（JSON）
- equity_curve: 資産曲線データ（JSON）
- trade_history: 全取引履歴（JSON）
- created_at: 作成日時
```

### 4. 戦略ショーケーステーブル (`strategy_showcase`)

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

### 5. ファンディングレートテーブル (`funding_rate_data`)

```sql
- symbol: 取引ペア
- funding_rate: ファンディングレート
- funding_timestamp: ファンディング時刻
```

### 6. オープンインタレストテーブル (`open_interest_data`)

```sql
- symbol: 取引ペア
- open_interest_value: 建玉残高（USD）
- data_timestamp: データ時刻
```

## 技術的詳細

### 使用可能なテクニカル指標（50 種類以上）

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
- **リソース管理**: メモリ・CPU 使用量の監視
- **ログ記録**: 詳細なデバッグ情報

### 高度なリスク管理

- **ポジションサイジング**: ケリー基準、固定リスク比率
- **リスクリワード比率**: エントリー条件のフィルタリング
- **動的 SL/TP**: ATR ベースのストップロス・テイクプロフィット
- **適応的リスク調整**: 過去の取引成績に基づくリスク調整

## 今後の拡張予定

- **機械学習統合**: より高度な最適化手法
- **リアルタイム取引**: 生成戦略の自動実行
- **マルチアセット対応**: 複数銘柄での戦略生成
- **カスタム指標**: ユーザー定義指標の追加
