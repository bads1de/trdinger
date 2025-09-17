# AGENTS.md

このファイルは、このリポジトリのコードを扱う際のエージェントへのガイドを提供します。

## Unified Configuration System

- すべての設定アクセスに `unified_config` シングルトンインスタンスを使用
- 環境変数はネストされたデリミタをサポート: `env_nested_delimiter = "__"`
- 設定階層: app, database, logging, security, market, data_collection, backtest, auto_strategy, ga, ml
- デフォルト取引シンボル: "BTC/USDT:USDT" (Bybit 形式)

## Error Handling Decorator

- `@safe_operation(context="...", is_api_call=True/False, default_return=...)` で堅牢なエラーハンドリング
- より良いエラーログのために `context` パラメータを使用して操作コンテキストを指定
- API エンドポイントには `is_api_call=True` を設定、内部操作には `False`
- クラッシュを防ぐために適切な `default_return` 値を指定

## Logging Infrastructure

- `DuplicateFilter(capacity=200, interval=1.0)` でログスパムを防ぐ
- ログ形式: `"%(asctime)s - %(name)s - %(levelname)s - %(message)s"`
- オートストラテジーロガーは別途設定: `logging.getLogger("app.services.auto_strategy")`

## Import Order Standards

- isort 設定に従う: profile="black", multi_line_output=3
- 既知のファーストパーティ: ["app", "backtest", "scripts"]
- 行長: 88 文字 (Black 規約)

## Type Checking Rigor

- **Backend**: MyPy strict モード有効:
  - `disallow_untyped_defs = true`
  - `no_implicit_optional = true`
  - `warn_return_any = true`
  - `strict_equality = true`
- **Frontend**: TypeScript `strict: true` モード
- すべてのサードパーティライブラリ (ccxt, pandas, numpy) は missing imports を無視

## Test Setup Specialties

- **Backend conftest.py**: インポートのためにバックエンドディレクトリを自動的に sys.path に追加
- **Frontend Jest setup**: 広範な DOM モッキングを含む:
  - `matchMedia`, `ResizeObserver`, `IntersectionObserver`
  - `TextEncoder`/`TextDecoder`, `Blob`, `URL.createObjectURL`
  - Next.js `NextResponse` モッキング
  - ダウンロードテスト用のカスタム `a` タグクリック動作

## Cryptocurrency-Specific Elements

- 主要取引所: Bybit (via CCXT 4.1.64)
- シンボル形式: "{BASE}/{QUOTE}:{QUOTE}" (例: "BTC/USDT:USDT")
- サポートされるタイムフレーム: ["15m", "30m", "1h", "4h", "1d"]
- データ収集制限: リクエストあたり最大 1000 レコード, ページ制限 200