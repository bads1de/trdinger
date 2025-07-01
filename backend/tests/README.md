# テストディレクトリ構成

このディレクトリには、バックエンドシステムの包括的なテストスイートが含まれています。

## ディレクトリ構成

```
tests/
├── README.md                           # このファイル
├── __init__.py
├── conftest.py                         # pytest設定
├── auto_strategy/                      # 自動戦略生成機能関連のテスト
│   ├── __init__.py
│   ├── test_auto_strategy_import.py
│   ├── test_corrected_strategy_generation.py
│   ├── test_daily_strategy_generation.py
│   ├── test_direct_ga.py
│   ├── test_extended_strategy_generation.py
│   ├── test_final_verification.py
│   ├── test_ga_api.py
│   ├── test_ga_config_comprehensive.py
│   ├── test_ga_engine_comprehensive.py
│   ├── test_ga_repositories.py
│   ├── test_ga_strategy_generation.py
│   ├── test_auto_strategy_fixes.py     # 自動戦略修正関連のテスト
│   └── test_parameter_generation_unification.py # パラメータ生成統合テスト
├── core/                               # コアロジック関連のテスト
│   ├── __init__.py
│   ├── indicators/                     # 指標計算関連のテスト
│   │   ├── __init__.py
│   │   └── test_indicator_calculation_refactoring.py # 指標計算リファクタリングテスト
│   └── ...
├── integration/                        # 統合テスト
│   ├── __init__.py
│   ├── test_api_endpoints.py           # APIエンドポイントの統合テスト
│   ├── test_preview_endpoint.py        # プレビューエンドポイントの統合テスト
│   ├── test_strategies_indicators_orchestrator_integration.py # 戦略と指標オーケストレーターの統合テスト
│   └── ...
├── performance/                        # パフォーマンステスト
│   ├── __init__.py
│   └── ...
├── unit/                               # 単体テスト
│   ├── __init__.py
│   ├── test_duplicate_filter.py        # 重複フィルターの単体テスト
│   ├── test_incremental_update.py      # 増分更新の単体テスト
│   ├── test_validate_direct.py         # 直接検証の単体テスト
│   ├── test_working_case.py            # 動作確認ケースの単体テスト
│   └── ...
└── ...
```

## テスト実行方法

### 1. 全体テスト実行 (pytest)
```bash
cd backend
pytest tests/
```

### 2. カテゴリ別テスト実行
```bash
# 自動戦略生成機能関連のテスト
pytest tests/auto_strategy/ -v

# コアロジック関連のテスト
pytest tests/core/ -v

# 統合テスト
pytest tests/integration/ -v

# パフォーマンステスト
pytest tests/performance/ -v

# 単体テスト
pytest tests/unit/ -v
```

### 3. 個別テスト実行
```bash
# 例: 特定の単体テストを実行
pytest tests/unit/test_duplicate_filter.py -v
```

## 注意事項

1.  **環境設定**: テスト実行前に必要な依存関係がインストールされていることを確認してください。`backend/requirements.txt` を参照してください。
2.  **データベース**: テストによってはデータベース接続が必要な場合があります。テスト環境のデータベース設定を確認してください。
3.  **依存関係**: `requirements.txt` の全パッケージがインストールされている必要があります。

## トラブルシューティング

### データベースエラー
データベース関連のエラーが発生する場合は、プロジェクトの初期化スクリプトを確認してください。

### API制限エラー
外部APIアクセスが制限されている場合、テストデータやモックの使用を検討してください。