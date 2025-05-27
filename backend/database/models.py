"""
データベースモデル定義
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, Index, BigInteger
from sqlalchemy.sql import func
from datetime import datetime
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

    # 時間軸（例: 1d, 4h, 1h, 30m, 15m）
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
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # インデックス定義
    __table_args__ = (
        # 複合インデックス（クエリ最適化）
        Index('idx_symbol_timeframe_timestamp', 'symbol', 'timeframe', 'timestamp'),
        Index('idx_timestamp_symbol', 'timestamp', 'symbol'),
        # ユニーク制約（重複データ防止）
        Index('uq_symbol_timeframe_timestamp', 'symbol', 'timeframe', 'timestamp', unique=True),
    )

    def __repr__(self):
        return (f"<OHLCVData(symbol='{self.symbol}', timeframe='{self.timeframe}', "
                f"timestamp='{self.timestamp}', close={self.close})>")

    def to_dict(self):
        """辞書形式に変換"""
        return {
            'id': self.id,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
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
        return (f"<DataCollectionLog(symbol='{self.symbol}', timeframe='{self.timeframe}', "
                f"status='{self.status}', records={self.records_collected})>")
