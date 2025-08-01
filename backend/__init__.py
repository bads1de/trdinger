"""
Trdinger バックエンドパッケージ

このパッケージには、トレーディング戦略のバックテストを実行するための
Pythonモジュールが含まれています。

主要モジュール:
- app.services.backtest.backtest_service: backtesting.pyライブラリを使用したバックテストサービス
- backtest.runner: バックテスト実行スクリプト
- app.utils.data_standardization: データ標準化ユーティリティ

使用例:
    from app.services.backtest.backtest_service import BacktestService
    from app.services.utils.data_standardization import standardize_ohlcv_columns

作成者: Trdinger Development Team
バージョン: 2.0.0 (backtesting.py統一版)
"""
