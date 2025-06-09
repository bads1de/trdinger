"""
データベースモデル定義
"""

from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    Index,
    Boolean,
    Text,
    JSON,
    ForeignKey,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .connection import Base


class OHLCVData(Base):
    """
    OHLCV価格データテーブル

    TimescaleDBのハイパーテーブルとして最適化されています。
    """

    __tablename__ = "ohlcv_data"

    # 主キー（複合キー: symbol + timeframe + timestamp）
    id = Column(Integer, primary_key=True, autoincrement=True)

    # 取引ペア（例: BTC/USD:BTC）
    symbol = Column(String(50), nullable=False, index=True)

    # 時間軸（例: 1d, 1h, 1m）
    timeframe = Column(String(10), nullable=False, index=True)

    # タイムスタンプ（UTC）
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)

    # OHLCV データ
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)

    # メタデータ
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # インデックス定義
    __table_args__ = (
        # 複合インデックス（クエリ最適化）
        Index("idx_symbol_timeframe_timestamp", "symbol", "timeframe", "timestamp"),
        Index("idx_timestamp_symbol", "timestamp", "symbol"),
        # ユニーク制約（重複データ防止）
        Index(
            "uq_symbol_timeframe_timestamp",
            "symbol",
            "timeframe",
            "timestamp",
            unique=True,
        ),
    )

    def __repr__(self):
        return (
            f"<OHLCVData(symbol='{self.symbol}', timeframe='{self.timeframe}', "
            f"timestamp='{self.timestamp}', close={self.close})>"
        )


class FundingRateData(Base):
    """
    ファンディングレートデータテーブル

    無期限契約のファンディングレート履歴を保存します。
    TimescaleDBのハイパーテーブルとして最適化されています。
    """

    __tablename__ = "funding_rate_data"

    # 主キー
    id = Column(Integer, primary_key=True, autoincrement=True)

    # 取引ペア（例: BTC/USDT:USDT）
    symbol = Column(String(50), nullable=False, index=True)

    # ファンディングレート
    funding_rate = Column(Float, nullable=False)

    # ファンディング時刻（UTC）
    funding_timestamp = Column(DateTime(timezone=True), nullable=False, index=True)

    # データ取得時刻（UTC）
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)

    # 次回ファンディング時刻（UTC）
    next_funding_timestamp = Column(DateTime(timezone=True), nullable=True)

    # マーク価格（参考情報）
    mark_price = Column(Float, nullable=True)

    # インデックス価格（参考情報）
    index_price = Column(Float, nullable=True)

    # メタデータ
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # インデックス定義
    __table_args__ = (
        # 複合インデックス（クエリ最適化）
        Index("idx_funding_symbol_timestamp", "symbol", "funding_timestamp"),
        Index("idx_funding_timestamp_symbol", "funding_timestamp", "symbol"),
        Index("idx_funding_symbol_created", "symbol", "created_at"),
        # ユニーク制約（重複データ防止）
        Index(
            "uq_symbol_funding_timestamp", "symbol", "funding_timestamp", unique=True
        ),
    )

    def __repr__(self):
        return (
            f"<FundingRateData(symbol='{self.symbol}', "
            f"funding_timestamp='{self.funding_timestamp}', "
            f"funding_rate={self.funding_rate})>"
        )


class OpenInterestData(Base):
    """
    オープンインタレスト（建玉残高）データテーブル

    無期限契約のオープンインタレスト履歴を保存します。
    TimescaleDBのハイパーテーブルとして最適化されています。
    """

    __tablename__ = "open_interest_data"

    # 主キー
    id = Column(Integer, primary_key=True, autoincrement=True)

    # 取引ペア（例: BTC/USDT:USDT）
    symbol = Column(String(50), nullable=False, index=True)

    # オープンインタレスト値（USD建て）
    open_interest_value = Column(Float, nullable=False)

    # データ時刻（UTC）
    data_timestamp = Column(DateTime(timezone=True), nullable=False, index=True)

    # データ取得時刻（UTC）
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)

    # メタデータ
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # インデックス定義
    __table_args__ = (
        # 複合インデックス（クエリ最適化）
        Index("idx_oi_symbol_data_timestamp", "symbol", "data_timestamp"),
        Index("idx_oi_data_timestamp_symbol", "data_timestamp", "symbol"),
        Index("idx_oi_symbol_created", "symbol", "created_at"),
        # ユニーク制約（重複データ防止）
        Index("uq_symbol_data_timestamp", "symbol", "data_timestamp", unique=True),
    )

    def __repr__(self):
        return (
            f"<OpenInterestData(symbol='{self.symbol}', "
            f"data_timestamp='{self.data_timestamp}', "
            f"open_interest_value={self.open_interest_value})>"
        )

    def to_dict(self):
        """辞書形式に変換"""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "open_interest_value": self.open_interest_value,
            "data_timestamp": (
                self.data_timestamp.isoformat() if self.data_timestamp else None
            ),
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class DataCollectionLog(Base):
    """
    データ収集ログテーブル

    データ収集の履歴と状態を管理します。
    """

    __tablename__ = "data_collection_log"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # 収集対象
    symbol = Column(String(50), nullable=False)
    timeframe = Column(String(10), nullable=False)

    # 収集期間
    start_time = Column(DateTime(timezone=True), nullable=False)
    end_time = Column(DateTime(timezone=True), nullable=False)

    # 収集結果
    records_collected = Column(Integer, default=0)
    status = Column(String(20), nullable=False)  # 'success', 'error', 'partial'
    error_message = Column(String(500), nullable=True)

    # メタデータ
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return (
            f"<DataCollectionLog(symbol='{self.symbol}', timeframe='{self.timeframe}', "
            f"status='{self.status}', records={self.records_collected})>"
        )


class TechnicalIndicatorData(Base):
    """
    テクニカル指標データテーブル

    OHLCVデータから計算されたテクニカル指標の履歴を保存します。
    SQLiteデータベースに最適化されています。
    """

    __tablename__ = "technical_indicator_data"

    # 主キー
    id = Column(Integer, primary_key=True, autoincrement=True)

    # 取引ペア（例: BTC/USDT）
    symbol = Column(String(50), nullable=False, index=True)

    # 時間枠（例: 1h, 4h, 1d）
    timeframe = Column(String(10), nullable=False, index=True)

    # 指標タイプ（例: SMA, EMA, RSI, MACD）
    indicator_type = Column(String(20), nullable=False)

    # 期間（例: 14, 20, 50）
    period = Column(Integer, nullable=False)

    # メイン値（SMA/EMA/RSIの値、MACDのメインライン）
    value = Column(Float, nullable=False)

    # シグナル線（MACDのシグナルライン等）
    signal_value = Column(Float, nullable=True)

    # ヒストグラム（MACDのヒストグラム等）
    histogram_value = Column(Float, nullable=True)

    # ボリンジャーバンド上限（Bollinger Bands Upper Band）
    upper_band = Column(Float, nullable=True)

    # ボリンジャーバンド下限（Bollinger Bands Lower Band）
    lower_band = Column(Float, nullable=True)

    # データ時刻（UTC）
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)

    # メタデータ
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # SQLite対応インデックス定義
    __table_args__ = (
        # 複合インデックス（クエリ最適化）
        Index("idx_ti_symbol_type_timestamp", "symbol", "indicator_type", "timestamp"),
        Index("idx_ti_timestamp_symbol", "timestamp", "symbol"),
        Index("idx_ti_symbol_type_period", "symbol", "indicator_type", "period"),
        # ユニーク制約（重複データ防止）
        Index(
            "uq_symbol_type_period_timestamp",
            "symbol",
            "indicator_type",
            "period",
            "timestamp",
            unique=True,
        ),
    )

    def __repr__(self):
        return (
            f"<TechnicalIndicatorData(symbol='{self.symbol}', "
            f"indicator_type='{self.indicator_type}', "
            f"period={self.period}, "
            f"timestamp='{self.timestamp}', "
            f"value={self.value})>"
        )

    def to_dict(self):
        """辞書形式に変換"""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "indicator_type": self.indicator_type,
            "period": self.period,
            "value": self.value,
            "signal_value": self.signal_value,
            "histogram_value": self.histogram_value,
            "upper_band": self.upper_band,
            "lower_band": self.lower_band,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class BacktestResult(Base):
    """
    バックテスト結果テーブル

    backtesting.pyライブラリを使用したバックテスト結果を保存します。
    """

    __tablename__ = "backtest_results"

    # 主キー
    id = Column(Integer, primary_key=True, autoincrement=True)

    # 戦略名
    strategy_name = Column(String(100), nullable=False)

    # 取引ペア（例: BTC/USDT）
    symbol = Column(String(50), nullable=False, index=True)

    # 時間軸（例: 1h, 4h, 1d）
    timeframe = Column(String(10), nullable=False, index=True)

    # バックテスト期間
    start_date = Column(DateTime(timezone=True), nullable=False)
    end_date = Column(DateTime(timezone=True), nullable=False)

    # 初期資金
    initial_capital = Column(Float, nullable=False)

    # 手数料率
    commission_rate = Column(Float, nullable=True, default=0.001)

    # 戦略設定（JSON形式）
    config_json = Column(JSON, nullable=False)

    # パフォーマンス指標（JSON形式）
    performance_metrics = Column(JSON, nullable=False)

    # 資産曲線データ（JSON形式）
    equity_curve = Column(JSON, nullable=False)

    # 取引履歴（JSON形式）
    trade_history = Column(JSON, nullable=False)

    # 実行時間（秒）
    execution_time = Column(Float, nullable=True)

    # ステータス
    status = Column(String(20), nullable=False, default="completed")

    # エラーメッセージ
    error_message = Column(Text, nullable=True)

    # メタデータ
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # インデックス定義
    __table_args__ = (
        Index("idx_backtest_symbol_timeframe", "symbol", "timeframe"),
        Index("idx_backtest_strategy_created", "strategy_name", "created_at"),
        Index("idx_backtest_created_at", "created_at"),
    )

    def __repr__(self):
        return (
            f"<BacktestResult(strategy='{self.strategy_name}', "
            f"symbol='{self.symbol}', timeframe='{self.timeframe}', "
            f"created_at='{self.created_at}')>"
        )


class GAExperiment(Base):
    """
    GA実験テーブル

    遺伝的アルゴリズムによる戦略生成実験の情報を保存します。
    """

    __tablename__ = "ga_experiments"

    # 主キー
    id = Column(Integer, primary_key=True, autoincrement=True)

    # 実験名
    name = Column(String(255), nullable=False)

    # GA設定（JSON形式）
    config = Column(JSON, nullable=False)

    # 実行状態
    status = Column(String(20), nullable=False, default="running")

    # 進捗率（0.0-1.0）
    progress = Column(Float, nullable=False, default=0.0)

    # 最高フィットネス
    best_fitness = Column(Float, nullable=True)

    # 総世代数
    total_generations = Column(Integer, nullable=True)

    # 現在の世代数
    current_generation = Column(Integer, nullable=False, default=0)

    # メタデータ
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)

    # インデックス定義
    __table_args__ = (
        Index("idx_ga_experiments_status", "status"),
        Index("idx_ga_experiments_created", "created_at"),
    )

    def __repr__(self):
        return (
            f"<GAExperiment(name='{self.name}', "
            f"status='{self.status}', "
            f"progress={self.progress:.2f})>"
        )


class GeneratedStrategy(Base):
    """
    生成された戦略テーブル

    GAによって生成された戦略の遺伝子情報を保存します。
    """

    __tablename__ = "generated_strategies"

    # 主キー
    id = Column(Integer, primary_key=True, autoincrement=True)

    # 実験ID（外部キー）
    experiment_id = Column(Integer, ForeignKey("ga_experiments.id"), nullable=False)

    # 戦略遺伝子データ（JSON形式）
    gene_data = Column(JSON, nullable=False)

    # 世代数
    generation = Column(Integer, nullable=False)

    # フィットネススコア
    fitness_score = Column(Float, nullable=True)

    # 親戦略のID（JSON配列）
    parent_ids = Column(JSON, nullable=True)

    # バックテスト結果ID（外部キー）
    backtest_result_id = Column(
        Integer, ForeignKey("backtest_results.id"), nullable=True
    )

    # メタデータ
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # リレーション
    experiment = relationship("GAExperiment", backref="strategies")
    backtest_result = relationship("BacktestResult", backref="generated_strategy")

    # インデックス定義
    __table_args__ = (
        Index("idx_generated_strategies_experiment", "experiment_id"),
        Index("idx_generated_strategies_fitness", "fitness_score"),
        Index("idx_generated_strategies_generation", "generation"),
    )

    def __repr__(self):
        return (
            f"<GeneratedStrategy(experiment_id={self.experiment_id}, "
            f"generation={self.generation}, "
            f"fitness={self.fitness_score:.4f if self.fitness_score else 'None'})>"
        )

    def to_dict(self):
        """辞書形式に変換"""
        return {
            "id": self.id,
            "strategy_name": self.strategy_name,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "initial_capital": self.initial_capital,
            "config_json": self.config_json,
            "performance_metrics": self.performance_metrics,
            "equity_curve": self.equity_curve,
            "trade_history": self.trade_history,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class StrategyTemplate(Base):
    """
    戦略テンプレートテーブル

    再利用可能な戦略設定を保存します。
    """

    __tablename__ = "strategy_templates"

    # 主キー
    id = Column(Integer, primary_key=True, autoincrement=True)

    # テンプレート名（一意）
    name = Column(String(100), nullable=False, unique=True)

    # 説明
    description = Column(Text, nullable=True)

    # カテゴリ（例: trend_following, mean_reversion）
    category = Column(String(50), nullable=True)

    # 戦略設定（JSON形式）
    config_json = Column(JSON, nullable=False)

    # 公開フラグ
    is_public = Column(Boolean, default=False)

    # メタデータ
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # インデックス定義
    __table_args__ = (
        Index("idx_template_category", "category"),
        Index("idx_template_public", "is_public"),
        Index("idx_template_created", "created_at"),
    )

    def __repr__(self):
        return (
            f"<StrategyTemplate(name='{self.name}', "
            f"category='{self.category}', is_public={self.is_public})>"
        )

    def to_dict(self):
        """辞書形式に変換"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "config_json": self.config_json,
            "is_public": self.is_public,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
