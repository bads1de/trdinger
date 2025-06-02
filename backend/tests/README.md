# テストディレクトリ構成

このディレクトリには、バックテストシステムの包括的なテストスイートが含まれています。

## ディレクトリ構成

```
tests/
├── README.md                           # このファイル
├── __init__.py
├── conftest.py                         # pytest設定
├── run_comprehensive_backtest_tests.py # 包括的テスト実行スクリプト
│
├── unit/                               # 単体テスト
│   ├── test_backtest_service.py        # BacktestServiceのテスト
│   ├── test_enhanced_backtest_service.py # EnhancedBacktestServiceのテスト
│   ├── test_backtest_data_service.py   # データサービスのテスト
│   ├── test_ohlcv_repository.py        # OHLCVリポジトリのテスト
│   ├── test_market_data_service.py     # マーケットデータサービスのテスト
│   ├── test_indicators.py              # テクニカル指標のテスト
│   ├── test_sma_cross_strategy.py      # SMAクロス戦略のテスト
│   └── ...                             # その他の単体テスト
│
├── integration/                        # 統合テスト
│   ├── test_unified_backtest_system.py # 統一バックテストシステムのテスト
│   ├── test_backtesting_py_integration.py # backtesting.pyライブラリ統合テスト
│   ├── test_comprehensive_backtest_with_real_data.py # 実データでの包括的テスト
│   ├── test_backtest_error_handling_with_real_data.py # エラーハンドリングテスト
│   ├── test_backtest_api_live.py       # API統合テスト
│   ├── test_data_collection.py         # データ収集統合テスト
│   └── ...                             # その他の統合テスト
│
├── performance/                        # パフォーマンステスト
│   └── test_backtest_performance_with_real_data.py # 実データでのパフォーマンステスト
│
├── optimization/                       # 最適化テスト
│   ├── test_comprehensive_optimization.py # 包括的最適化テスト
│   ├── test_simple_backtest.py         # シンプルバックテスト（最適化なし）
│   ├── test_real_btc_optimization.py   # 実BTC データでの最適化テスト
│   ├── test_enhanced_optimization_demo.py # 拡張最適化デモ
│   ├── test_error_handling_optimization.py # 最適化エラーハンドリング
│   └── test_optimization_database_integration.py # 最適化DB統合テスト
│
├── accuracy/                           # 精度テスト
│   └── test_backtest_accuracy.py       # バックテスト精度テスト
│
├── scripts/                            # スクリプトテスト
│   └── test_updated_symbols.py         # シンボル更新テスト
│
└── demos/                              # デモ・比較テスト
    └── strategy_comparison_test.py     # 戦略比較テスト
```

## テスト実行方法

### 1. 全体テスト実行
```bash
cd backend
python tests/run_comprehensive_backtest_tests.py
```

### 2. カテゴリ別テスト実行
```bash
# 単体テスト
python -m pytest tests/unit/ -v

# 統合テスト
python -m pytest tests/integration/ -v

# パフォーマンステスト
python -m pytest tests/performance/ -v

# 最適化テスト
python -m pytest tests/optimization/ -v
```

### 3. 個別テスト実行
```bash
# シンプルバックテスト
PYTHONPATH=/mnt/persist/workspace/backend python tests/optimization/test_simple_backtest.py

# 包括的最適化テスト
PYTHONPATH=/mnt/persist/workspace/backend python tests/optimization/test_comprehensive_optimization.py
```

## テストの特徴

### 最適化テスト
- **test_simple_backtest.py**: 最適化機能のpickleエラーを回避したシンプルなバックテスト
- **test_comprehensive_optimization.py**: 包括的な最適化テスト（サンプルデータ使用）
- **test_real_btc_optimization.py**: 実際のBTCデータでの最適化テスト

### データ対応
- **サンプルデータ生成**: 実際のAPIアクセスが制限されている場合の代替
- **実データテスト**: データベース内の実際のOHLCVデータを使用
- **エラーハンドリング**: 様々なエラーケースに対する堅牢性テスト

### パフォーマンス
- **小規模データセット**: 高速テスト用
- **中規模データセット**: 実用的なテスト用
- **大規模データセット**: 本格的なパフォーマンステスト用

## 注意事項

1. **環境設定**: テスト実行前に`PYTHONPATH`を適切に設定してください
2. **データベース**: テストはSQLiteデータベースを使用します
3. **依存関係**: `requirements.txt`の全パッケージがインストールされている必要があります
4. **最適化**: 一部の最適化テストはマルチプロセシングの制限により制約があります

## トラブルシューティング

### pickleエラー
最適化テストでpickleエラーが発生する場合は、`test_simple_backtest.py`を使用してください。

### データベースエラー
データベース関連のエラーが発生する場合は、以下を実行してください：
```bash
PYTHONPATH=/mnt/persist/workspace/backend python scripts/init_database.py
```

### API制限エラー
外部APIアクセスが制限されている場合、サンプルデータ生成機能を使用します。
