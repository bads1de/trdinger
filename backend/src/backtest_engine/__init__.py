"""
Trdinger バックテストエンジンパッケージ

トレーディング戦略のバックテストを実行するためのコアエンジンです。
モジュール:
- indicators: テクニカル指標の計算ライブラリ
- strategy_executor: 戦略実行とパフォーマンス計算エンジン

機能:
- SMA, EMA, RSI, MACD等のテクニカル指標計算
- 戦略のエントリー・エグジットロジック実行
- パフォーマンス指標の計算（シャープレシオ、ドローダウン等）
- 取引履歴と損益曲線の記録

使用例:
    from backtest_engine.indicators import TechnicalIndicators
    from backtest_engine.strategy_executor import StrategyExecutor

    # 指標計算
    sma = TechnicalIndicators.sma(data['close'], 20)

    # バックテスト実行
    executor = StrategyExecutor(initial_capital=100000)
    result = executor.run_backtest(data, strategy_config)

作成者: Trdinger Development Team
バージョン: 1.0.0
"""
