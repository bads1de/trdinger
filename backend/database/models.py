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
    Text,
    JSON,
    ForeignKey,
)
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.sql import func
from .connection import Base


class OHLCVData(Base):
    """OHLCV価格データテーブル。

    取引所から取得したOHLCV（始値、高値、安値、終値、出来高）データを保存します。
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
    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[float] = mapped_column(Float, nullable=False)

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

    # uses ToDictMixin.to_dict


class BacktestResult(Base):
    """バックテスト結果テーブル。

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

    # to_dictメソッドを削除し、リポジトリ層でデータ変換を行うように変更


class GAExperiment(Base):
    """GA実験テーブル。

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
    """生成された戦略テーブル。

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

    # フィットネススコア（単一目的最適化用、後方互換性のため保持）
    fitness_score = Column(Float, nullable=True)

    # フィットネス値（多目的最適化用、JSON配列形式）
    fitness_values = Column(JSON, nullable=True)

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
        fitness_str = (
            f"{self.fitness_score:.4f}" if self.fitness_score is not None else "None"
        )
        return (
            f"<GeneratedStrategy(experiment_id={self.experiment_id}, "
            f"generation={self.generation}, "
            f"fitness={fitness_str})>"
        )
