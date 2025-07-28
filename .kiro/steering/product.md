---
inclusion: always
---

---

## inclusion: always

# Trdinger トレーディングプラットフォーム

ML を活用したアルゴリズム的仮想通貨トレーディングプラットフォーム

## 重要な財務ルール

- **NEVER use `float`** - すべての価格、金額、パーセンテージには `Decimal` 型を使用
- **8 decimal precision** - 仮想通貨ペアの精度は小数点以下 8 桁
- **ROUND_HALF_UP** - すべての財務計算で適切な丸め処理を実装
- **Unit test all financial calculations** - 既知の期待される結果で検証
- **Millisecond precision** - 市場データのタイムスタンプ、ユーザー操作は秒精度

```python
from decimal import Decimal, ROUND_HALF_UP
price = Decimal('0.12345678')  # Never float
amount = price.quantize(Decimal('0.00000001'), rounding=ROUND_HALF_UP)
```

## コアドメインエンティティ

- **Strategy**: パラメータ付きトレーディングアルゴリズム（デプロイ前バックテスト必須）
- **Backtest**: シャープレシオ、最大ドローダウン、勝率による過去検証
- **Portfolio**: リアルタイム P&L 追跡付きポジション集合
- **Signal**: ML 生成の売買推奨（信頼度スコア 0-1）
- **Position**: エントリー/エグジットポイント付きアクティブトレード

## アーキテクチャ要件

### データフロー

- 市場データ: `コレクター → DB → 戦略エンジン → ポートフォリオ`
- ユーザー操作: `フロントエンド → API → ビジネスロジック → DB`
- ML パイプライン: `生データ → 特徴量 → モデル → シグナル`

### 必須パターン

- **Dependency Injection** - トレーディングロジックとインフラを分離
- **Strategy Pattern** - トレーディングアルゴリズム実装
- **Repository Pattern** - すべてのデータアクセス
- **async/await** - すべての I/O 操作
- **ML separation** - トレーニングコードと推論コードを分離

## セキュリティ & パフォーマンス

### セキュリティ

- **NEVER log API keys** - レスポンス、ログ、エラーメッセージに記録禁止
- **Separate API keys** - 取引所ごと、環境ごとに分離
- **Encrypt sensitive data** - トレーディングデータの保管時暗号化
- **Audit all portfolio changes** - トランザクションレベルロギング
- **Validate all inputs** - トレーディング操作前の検証

### パフォーマンス目標

- 市場データ処理: **< 100ms**
- 戦略シグナル生成: **< 500ms**
- ポートフォリオ更新: **< 1 秒**
- **Connection pooling** - DB および API 接続
- **Caching** - 頻繁アクセス市場データ

## 開発ガイドライン

### エラーハンドリング

- **Circuit breakers** - API レート制限対応
- **Structured logging** - 相関 ID でトレーシング
- **Graceful degradation** - 外部サービス障害時
- **Distinguish error types** - 回復可能 vs 致命的エラー

### UI/UX 規約

- **Color coding**: 緑（利益/買い）、赤（損失/売り）、青（中立/ホールド）
- **Loading states** - すべてのトレーディング操作
- **Confirmation dialogs** - 破壊的アクション
- **Confidence intervals** - ML 予測表示
- **Real-time updates** - ポートフォリオ価値とポジション

### テスト要件

- **Mock external APIs** - 単体テストで取引所 API をモック
- **Deterministic seeds** - 再現可能バックテスト
- **Test edge cases** - ゼロバランス、API 障害、極端市場状況
- **Integration tests** - 重要トレーディングフロー
- **Test concurrency** - 競合状態の確認
