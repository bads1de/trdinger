# オートストラテジー・バックテスト統合実装計画

## 概要

オートストラテジー機能で生成された戦略を実際にバックテストし、フロントエンドの`/strategies`ページにカード形式で表示する機能を実装します。

## 実装目標

1. **実際のオートストラテジー戦略の生成**
   - GAアルゴリズムによる戦略生成の実行
   - 生成された戦略のバックテスト実行
   - 結果のデータベース保存

2. **データ統合サービスの実装**
   - `generated_strategies`と`backtest_results`の結合
   - `StrategyShowcase`形式への変換
   - 既存の`strategy_showcase`データとの統合

3. **フロントエンド表示機能**
   - 統合されたデータの取得API
   - カード形式での戦略表示
   - フィルタリング・ソート機能

## 実装ファイル一覧

### 新規作成ファイル

1. **`backend/app/api/strategies.py`**
   - 統合戦略API
   - `/api/strategies/unified` エンドポイント

2. **`backend/app/core/services/strategy_integration_service.py`**
   - データ統合サービス
   - 形式変換ロジック

3. **`frontend/app/api/strategies/unified/route.ts`**
   - フロントエンド統合API

4. **`frontend/types/auto-strategy.ts`**
   - オートストラテジー関連型定義

### 修正ファイル

1. **`backend/app/api/strategy_showcase.py`**
   - 統合エンドポイント追加

2. **`frontend/app/strategies/page.tsx`**
   - 統合データ表示対応

3. **`frontend/components/strategies/StrategyCard.tsx`**
   - オートストラテジー情報表示

## データフロー

```
1. オートストラテジー実行
   ↓
2. generated_strategies テーブルに保存
   ↓
3. バックテスト実行
   ↓
4. backtest_results テーブルに保存
   ↓
5. StrategyIntegrationService で統合
   ↓
6. 統合API経由でフロントエンドに配信
   ↓
7. /strategies ページでカード表示
```

## 実装順序

1. **バックエンドサービス層実装**
2. **バックエンドAPI実装**
3. **オートストラテジー実行・データ生成**
4. **フロントエンドAPI実装**
5. **フロントエンド表示機能実装**
6. **統合テスト**

## データ変換仕様

### generated_strategies → StrategyShowcase 変換

| generated_strategies | StrategyShowcase | 変換ロジック |
|---------------------|------------------|-------------|
| id | id | 直接マッピング |
| gene_data.strategy_name | name | 戦略名抽出 |
| gene_data | description | 遺伝子情報から説明生成 |
| gene_data.indicators | indicators | 使用指標リスト |
| gene_data | parameters | パラメータ抽出 |
| backtest_result.performance_metrics | expected_return, sharpe_ratio, etc. | パフォーマンス指標 |
| fitness_score | - | 内部評価用 |

## API仕様

### GET /api/strategies/unified

**レスポンス:**
```json
{
  "success": true,
  "strategies": [
    {
      "id": "auto_1",
      "name": "GA生成戦略_SMA_RSI",
      "description": "遺伝的アルゴリズムで生成されたSMA+RSI戦略",
      "category": "auto_generated",
      "indicators": ["SMA", "RSI"],
      "parameters": {...},
      "expected_return": 0.15,
      "sharpe_ratio": 1.2,
      "max_drawdown": 0.08,
      "win_rate": 0.65,
      "source": "auto_strategy",
      "experiment_id": 1,
      "generation": 10
    }
  ]
}
```

## 技術仕様

- **バックエンド**: FastAPI, SQLAlchemy
- **フロントエンド**: Next.js, TypeScript
- **データベース**: SQLite (trdinger.db)
- **戦略生成**: 遺伝的アルゴリズム
- **バックテスト**: backtesting.py

## 成功基準

1. ✅ オートストラテジーで実際の戦略が生成される
2. ✅ 生成された戦略のバックテストが実行される
3. ✅ フロントエンドで戦略がカード形式で表示される
4. ✅ 既存のshowcase戦略と統合表示される
5. ✅ フィルタリング・ソート機能が動作する
