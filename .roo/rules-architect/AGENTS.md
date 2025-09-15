# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Unified Configuration System

- Use `unified_config` singleton instance for all configuration access
- Environment variables support nested delimiter: `env_nested_delimiter = "__"`
- Configuration hierarchy: app, database, logging, security, market, data_collection, backtest, auto_strategy, ga, ml
- Default trading symbol: "BTC/USDT:USDT" (Bybit format)

## Error Handling Decorator

- `@safe_operation(context="...", is_api_call=True/False, default_return=...)` for robust error handling
- Use `context` parameter to specify operation context for better error logging
- Set `is_api_call=True` for API endpoints, `False` for internal operations
- Provide appropriate `default_return` values to prevent crashes

## Logging Infrastructure

- `DuplicateFilter(capacity=200, interval=1.0)` prevents log spam
- Log format: `"%(asctime)s - %(name)s - %(levelname)s - %(message)s"`
- Auto-strategy logger configured separately: `logging.getLogger("app.services.auto_strategy")`

## Import Order Standards

- Follow isort configuration: profile="black", multi_line_output=3
- Known first party: ["app", "backtest", "scripts"]
- Line length: 88 characters (Black convention)

## Type Checking Rigor

- **Backend**: MyPy strict mode enabled:
  - `disallow_untyped_defs = true`
  - `no_implicit_optional = true`
  - `warn_return_any = true`
  - `strict_equality = true`
- **Frontend**: TypeScript `strict: true` mode
- All third-party libraries (ccxt, pandas, numpy) ignore missing imports

## Test Setup Specialties

- **Backend conftest.py**: Automatically adds backend directory to sys.path for imports
- **Frontend Jest setup**: Extensive DOM mocking including:
  - `matchMedia`, `ResizeObserver`, `IntersectionObserver`
  - `TextEncoder`/`TextDecoder`, `Blob`, `URL.createObjectURL`
  - Next.js `NextResponse` mocking
  - Custom `a` tag click behavior for download tests

## Cryptocurrency-Specific Elements

- Primary exchange: Bybit (via CCXT 4.1.64)
- Symbol format: "{BASE}/{QUOTE}:{QUOTE}" (e.g., "BTC/USDT:USDT")
- Supported timeframes: ["15m", "30m", "1h", "4h", "1d"]
- Data collection limits: max 1000 records per request, page limit 200