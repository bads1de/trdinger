# 📡 API仕様書

Trdinger バックテストサービスのAPI仕様書です。

## 📋 目次

- [概要](#概要)
- [認証](#認証)
- [エラーハンドリング](#エラーハンドリング)
- [戦略API](#戦略api)
- [バックテストAPI](#バックテストapi)
- [データAPI](#データapi)
- [型定義](#型定義)

## 🌟 概要

### ベースURL
```
http://localhost:3000/api
```

### レスポンス形式
全てのAPIレスポンスはJSON形式です。

```json
{
  "success": true,
  "data": {},
  "message": "成功メッセージ",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

### HTTPステータスコード

| コード | 説明 |
|--------|------|
| 200 | 成功 |
| 201 | 作成成功 |
| 400 | リクエストエラー |
| 404 | リソースが見つからない |
| 500 | サーバーエラー |

## 🔐 認証

現在のバージョンでは認証は実装されていません。将来のバージョンでJWT認証を実装予定です。

## ❌ エラーハンドリング

### エラーレスポンス形式

```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "入力値が無効です",
    "details": {
      "field": "strategy_name",
      "reason": "必須フィールドです"
    }
  },
  "timestamp": "2024-01-01T00:00:00Z"
}
```

### エラーコード

| コード | 説明 |
|--------|------|
| `VALIDATION_ERROR` | 入力値検証エラー |
| `STRATEGY_NOT_FOUND` | 戦略が見つからない |
| `BACKTEST_FAILED` | バックテスト実行エラー |
| `DATA_NOT_AVAILABLE` | データが利用できない |
| `INTERNAL_ERROR` | 内部サーバーエラー |

## 🎯 戦略API

### 戦略一覧取得

```http
GET /api/strategies
```

**レスポンス:**
```json
{
  "success": true,
  "data": [
    {
      "id": "strategy-1",
      "strategy_name": "SMA クロス戦略",
      "target_pair": "BTC/USD",
      "timeframe": "1h",
      "indicators": [
        {
          "name": "SMA",
          "params": { "period": 20 }
        }
      ],
      "entry_rules": [
        {
          "condition": "close > SMA(close, 20)",
          "description": "終値が20期間移動平均を上回る"
        }
      ],
      "exit_rules": [
        {
          "condition": "close < SMA(close, 20)",
          "description": "終値が20期間移動平均を下回る"
        }
      ],
      "created_at": "2024-01-01T00:00:00Z",
      "updated_at": "2024-01-01T00:00:00Z"
    }
  ]
}
```

### 戦略詳細取得

```http
GET /api/strategies/{id}
```

**パラメータ:**
- `id` (string): 戦略ID

**レスポンス:**
```json
{
  "success": true,
  "data": {
    "id": "strategy-1",
    "strategy_name": "SMA クロス戦略",
    // ... 戦略の詳細情報
  }
}
```

### 戦略作成

```http
POST /api/strategies
```

**リクエストボディ:**
```json
{
  "strategy_name": "新しい戦略",
  "target_pair": "BTC/USD",
  "timeframe": "1h",
  "indicators": [
    {
      "name": "SMA",
      "params": { "period": 20 }
    }
  ],
  "entry_rules": [
    {
      "condition": "close > SMA(close, 20)",
      "description": "エントリー条件"
    }
  ],
  "exit_rules": [
    {
      "condition": "close < SMA(close, 20)",
      "description": "エグジット条件"
    }
  ]
}
```

**レスポンス:**
```json
{
  "success": true,
  "data": {
    "id": "strategy-2",
    "strategy_name": "新しい戦略",
    // ... 作成された戦略の詳細
  },
  "message": "戦略が正常に作成されました"
}
```

### 戦略更新

```http
PUT /api/strategies/{id}
```

**パラメータ:**
- `id` (string): 戦略ID

**リクエストボディ:** 戦略作成と同じ形式

### 戦略削除

```http
DELETE /api/strategies/{id}
```

**パラメータ:**
- `id` (string): 戦略ID

**レスポンス:**
```json
{
  "success": true,
  "message": "戦略が正常に削除されました"
}
```

## 📊 バックテストAPI

### バックテスト実行

```http
POST /api/backtest
```

**リクエストボディ:**
```json
{
  "strategy": {
    "id": "strategy-1",
    "strategy_name": "SMA クロス戦略",
    // ... 戦略の詳細
  },
  "start_date": "2023-01-01T00:00:00Z",
  "end_date": "2023-12-31T23:59:59Z",
  "initial_capital": 100000,
  "commission_rate": 0.001,
  "slippage_rate": 0.0005
}
```

**レスポンス:**
```json
{
  "success": true,
  "data": {
    "id": "backtest-1",
    "strategy_id": "strategy-1",
    "config": {
      // ... バックテスト設定
    },
    "total_return": 0.15,
    "sharpe_ratio": 1.2,
    "max_drawdown": -0.08,
    "win_rate": 0.65,
    "profit_factor": 1.8,
    "total_trades": 45,
    "winning_trades": 29,
    "losing_trades": 16,
    "avg_win": 1250.50,
    "avg_loss": -680.25,
    "equity_curve": [
      {
        "timestamp": "2023-01-01T00:00:00Z",
        "equity": 100000,
        "drawdown": 0
      }
      // ... 更多数据点
    ],
    "trade_history": [
      {
        "id": "trade-1",
        "timestamp": "2023-01-02T10:30:00Z",
        "type": "buy",
        "price": 45000,
        "quantity": 2.0,
        "commission": 90.0
      }
      // ... 更多交易记录
    ],
    "created_at": "2024-01-01T00:00:00Z"
  }
}
```

### バックテスト結果一覧取得

```http
GET /api/backtest/results
```

**クエリパラメータ:**
- `strategy_id` (string, optional): 戦略IDでフィルタ
- `limit` (number, optional): 取得件数制限 (デフォルト: 20)
- `offset` (number, optional): オフセット (デフォルト: 0)

### バックテスト結果詳細取得

```http
GET /api/backtest/results/{id}
```

**パラメータ:**
- `id` (string): バックテスト結果ID

## 📈 データAPI

### 価格データ取得

```http
GET /api/data/prices
```

**クエリパラメータ:**
- `symbol` (string): 通貨ペア (例: "BTC/USD")
- `timeframe` (string): 時間足 (例: "1h", "1d")
- `start_date` (string): 開始日時 (ISO形式)
- `end_date` (string): 終了日時 (ISO形式)
- `limit` (number, optional): 取得件数制限

**レスポンス:**
```json
{
  "success": true,
  "data": {
    "symbol": "BTC/USD",
    "timeframe": "1h",
    "data": [
      {
        "timestamp": "2023-01-01T00:00:00Z",
        "open": 45000,
        "high": 45500,
        "low": 44800,
        "close": 45200,
        "volume": 1250000
      }
      // ... 更多价格数据
    ]
  }
}
```

### 利用可能な通貨ペア取得

```http
GET /api/data/symbols
```

**レスポンス:**
```json
{
  "success": true,
  "data": [
    {
      "symbol": "BTC/USD",
      "name": "Bitcoin / US Dollar",
      "base": "BTC",
      "quote": "USD"
    },
    {
      "symbol": "ETH/USD",
      "name": "Ethereum / US Dollar",
      "base": "ETH",
      "quote": "USD"
    }
  ]
}
```

### 利用可能な時間足取得

```http
GET /api/data/timeframes
```

**レスポンス:**
```json
{
  "success": true,
  "data": [
    {
      "value": "1m",
      "label": "1分足"
    },
    {
      "value": "5m",
      "label": "5分足"
    },
    {
      "value": "1h",
      "label": "1時間足"
    },
    {
      "value": "1d",
      "label": "日足"
    }
  ]
}
```

## 📝 型定義

### TechnicalIndicator

```typescript
interface TechnicalIndicator {
  name: string;                           // 指標名
  params: Record<string, number | string>; // パラメータ
}
```

### TradingCondition

```typescript
interface TradingCondition {
  condition: string;    // 条件式
  description?: string; // 説明
}
```

### TradingStrategy

```typescript
interface TradingStrategy {
  id?: string;                      // 戦略ID
  strategy_name: string;            // 戦略名
  target_pair: string;              // 対象通貨ペア
  timeframe: string;                // 時間足
  indicators: TechnicalIndicator[]; // 指標リスト
  entry_rules: TradingCondition[];  // エントリー条件
  exit_rules: TradingCondition[];   // エグジット条件
  created_at?: Date;                // 作成日時
  updated_at?: Date;                // 更新日時
}
```

### BacktestConfig

```typescript
interface BacktestConfig {
  strategy: TradingStrategy; // 戦略
  start_date: string;        // 開始日
  end_date: string;          // 終了日
  initial_capital: number;   // 初期資金
  commission_rate?: number;  // 手数料率
  slippage_rate?: number;    // スリッパージ率
}
```

### BacktestResult

```typescript
interface BacktestResult {
  id?: string;                    // 結果ID
  strategy_id: string;            // 戦略ID
  config: BacktestConfig;         // 設定
  total_return: number;           // 総リターン
  sharpe_ratio: number;           // シャープレシオ
  max_drawdown: number;           // 最大ドローダウン
  win_rate: number;               // 勝率
  profit_factor: number;          // プロフィットファクター
  total_trades: number;           // 総取引数
  winning_trades: number;         // 勝ち取引数
  losing_trades: number;          // 負け取引数
  avg_win: number;                // 平均利益
  avg_loss: number;               // 平均損失
  equity_curve: EquityPoint[];    // 損益曲線
  trade_history: Trade[];         // 取引履歴
  created_at: Date;               // 作成日時
}
```

### EquityPoint

```typescript
interface EquityPoint {
  timestamp: string; // タイムスタンプ
  equity: number;    // 資産価値
  drawdown: number;  // ドローダウン
}
```

### Trade

```typescript
interface Trade {
  id: string;        // 取引ID
  timestamp: string; // 実行時刻
  type: 'buy' | 'sell'; // 取引タイプ
  price: number;     // 価格
  quantity: number;  // 数量
  commission: number; // 手数料
  pnl?: number;      // 損益
}
```

### PriceData

```typescript
interface PriceData {
  timestamp: string; // タイムスタンプ
  open: number;      // 始値
  high: number;      // 高値
  low: number;       // 安値
  close: number;     // 終値
  volume: number;    // 出来高
}
```

## 🔄 レート制限

現在のバージョンではレート制限は実装されていませんが、将来のバージョンで以下の制限を実装予定です：

- **一般API**: 100リクエスト/分
- **バックテストAPI**: 10リクエスト/分
- **データAPI**: 1000リクエスト/時

## 📚 使用例

### JavaScript/TypeScript

```typescript
// 戦略作成
const strategy = await fetch('/api/strategies', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    strategy_name: 'SMAクロス戦略',
    target_pair: 'BTC/USD',
    timeframe: '1h',
    indicators: [
      { name: 'SMA', params: { period: 20 } },
      { name: 'SMA', params: { period: 50 } }
    ],
    entry_rules: [
      { condition: 'SMA(close, 20) > SMA(close, 50)' }
    ],
    exit_rules: [
      { condition: 'SMA(close, 20) < SMA(close, 50)' }
    ]
  })
});

// バックテスト実行
const backtest = await fetch('/api/backtest', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    strategy: strategyData,
    start_date: '2023-01-01T00:00:00Z',
    end_date: '2023-12-31T23:59:59Z',
    initial_capital: 100000,
    commission_rate: 0.001
  })
});
```

### Python

```python
import requests

# 戦略作成
strategy_data = {
    "strategy_name": "SMAクロス戦略",
    "target_pair": "BTC/USD",
    "timeframe": "1h",
    "indicators": [
        {"name": "SMA", "params": {"period": 20}},
        {"name": "SMA", "params": {"period": 50}}
    ],
    "entry_rules": [
        {"condition": "SMA(close, 20) > SMA(close, 50)"}
    ],
    "exit_rules": [
        {"condition": "SMA(close, 20) < SMA(close, 50)"}
    ]
}

response = requests.post(
    'http://localhost:3000/api/strategies',
    json=strategy_data
)

# バックテスト実行
backtest_config = {
    "strategy": strategy_data,
    "start_date": "2023-01-01T00:00:00Z",
    "end_date": "2023-12-31T23:59:59Z",
    "initial_capital": 100000,
    "commission_rate": 0.001
}

backtest_response = requests.post(
    'http://localhost:3000/api/backtest',
    json=backtest_config
)
```

## 📞 サポート

API に関する質問や問題がある場合：

- **GitHub Issues**: バグ報告や機能リクエスト
- **GitHub Discussions**: 一般的な質問や使用方法

---

**最終更新**: 2024年1月1日  
**APIバージョン**: v1.0.0
