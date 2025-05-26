"""
データアクセス層（リポジトリパターン）
"""
from typing import List, Optional, Tuple
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc, asc, func
from sqlalchemy.dialects.postgresql import insert
import pandas as pd
import logging

from .models import OHLCVData, DataCollectionLog
from .connection import get_db

logger = logging.getLogger(__name__)


class OHLCVRepository:
    """OHLCV データのリポジトリクラス"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def insert_ohlcv_data(self, ohlcv_records: List[dict]) -> int:
        """
        OHLCV データを一括挿入（重複は無視）
        
        Args:
            ohlcv_records: OHLCV データのリスト
            
        Returns:
            挿入された件数
        """
        if not ohlcv_records:
            return 0
        
        try:
            # PostgreSQL の ON CONFLICT を使用して重複を無視
            stmt = insert(OHLCVData).values(ohlcv_records)
            stmt = stmt.on_conflict_do_nothing(
                index_elements=['symbol', 'timeframe', 'timestamp']
            )
            
            result = self.db.execute(stmt)
            self.db.commit()
            
            inserted_count = result.rowcount
            logger.info(f"OHLCV データを {inserted_count} 件挿入しました")
            return inserted_count
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"OHLCV データ挿入エラー: {e}")
            raise
    
    def get_ohlcv_data(
        self, 
        symbol: str, 
        timeframe: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[OHLCVData]:
        """
        OHLCV データを取得
        
        Args:
            symbol: 取引ペア
            timeframe: 時間軸
            start_time: 開始時刻
            end_time: 終了時刻
            limit: 取得件数制限
            
        Returns:
            OHLCV データのリスト
        """
        try:
            query = self.db.query(OHLCVData).filter(
                and_(
                    OHLCVData.symbol == symbol,
                    OHLCVData.timeframe == timeframe
                )
            )
            
            if start_time:
                query = query.filter(OHLCVData.timestamp >= start_time)
            
            if end_time:
                query = query.filter(OHLCVData.timestamp <= end_time)
            
            # 時系列順でソート
            query = query.order_by(asc(OHLCVData.timestamp))
            
            if limit:
                query = query.limit(limit)
            
            return query.all()
            
        except Exception as e:
            logger.error(f"OHLCV データ取得エラー: {e}")
            raise
    
    def get_ohlcv_dataframe(
        self, 
        symbol: str, 
        timeframe: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        OHLCV データをDataFrameとして取得
        
        Args:
            symbol: 取引ペア
            timeframe: 時間軸
            start_time: 開始時刻
            end_time: 終了時刻
            limit: 取得件数制限
            
        Returns:
            OHLCV データのDataFrame
        """
        records = self.get_ohlcv_data(symbol, timeframe, start_time, end_time, limit)
        
        if not records:
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        data = []
        for record in records:
            data.append({
                'timestamp': record.timestamp,
                'open': record.open,
                'high': record.high,
                'low': record.low,
                'close': record.close,
                'volume': record.volume
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df
    
    def get_latest_timestamp(self, symbol: str, timeframe: str) -> Optional[datetime]:
        """
        指定されたシンボル・時間軸の最新タイムスタンプを取得
        
        Args:
            symbol: 取引ペア
            timeframe: 時間軸
            
        Returns:
            最新のタイムスタンプ（データがない場合はNone）
        """
        try:
            result = self.db.query(func.max(OHLCVData.timestamp)).filter(
                and_(
                    OHLCVData.symbol == symbol,
                    OHLCVData.timeframe == timeframe
                )
            ).scalar()
            
            return result
            
        except Exception as e:
            logger.error(f"最新タイムスタンプ取得エラー: {e}")
            raise
    
    def get_data_count(self, symbol: str, timeframe: str) -> int:
        """
        指定されたシンボル・時間軸のデータ件数を取得
        
        Args:
            symbol: 取引ペア
            timeframe: 時間軸
            
        Returns:
            データ件数
        """
        try:
            count = self.db.query(OHLCVData).filter(
                and_(
                    OHLCVData.symbol == symbol,
                    OHLCVData.timeframe == timeframe
                )
            ).count()
            
            return count
            
        except Exception as e:
            logger.error(f"データ件数取得エラー: {e}")
            raise
    
    def get_available_symbols(self) -> List[str]:
        """
        利用可能なシンボルのリストを取得
        
        Returns:
            シンボルのリスト
        """
        try:
            symbols = self.db.query(OHLCVData.symbol).distinct().all()
            return [symbol[0] for symbol in symbols]
            
        except Exception as e:
            logger.error(f"利用可能シンボル取得エラー: {e}")
            raise
    
    def get_date_range(self, symbol: str, timeframe: str) -> Tuple[Optional[datetime], Optional[datetime]]:
        """
        指定されたシンボル・時間軸のデータ期間を取得
        
        Args:
            symbol: 取引ペア
            timeframe: 時間軸
            
        Returns:
            (最古のタイムスタンプ, 最新のタイムスタンプ)
        """
        try:
            result = self.db.query(
                func.min(OHLCVData.timestamp),
                func.max(OHLCVData.timestamp)
            ).filter(
                and_(
                    OHLCVData.symbol == symbol,
                    OHLCVData.timeframe == timeframe
                )
            ).first()
            
            return result if result else (None, None)
            
        except Exception as e:
            logger.error(f"データ期間取得エラー: {e}")
            raise


class DataCollectionLogRepository:
    """データ収集ログのリポジトリクラス"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def log_collection(
        self, 
        symbol: str, 
        timeframe: str,
        start_time: datetime,
        end_time: datetime,
        records_collected: int,
        status: str,
        error_message: Optional[str] = None
    ) -> DataCollectionLog:
        """
        データ収集ログを記録
        
        Args:
            symbol: 取引ペア
            timeframe: 時間軸
            start_time: 収集開始時刻
            end_time: 収集終了時刻
            records_collected: 収集件数
            status: ステータス
            error_message: エラーメッセージ
            
        Returns:
            作成されたログレコード
        """
        try:
            log_record = DataCollectionLog(
                symbol=symbol,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time,
                records_collected=records_collected,
                status=status,
                error_message=error_message
            )
            
            self.db.add(log_record)
            self.db.commit()
            self.db.refresh(log_record)
            
            logger.info(f"データ収集ログを記録しました: {symbol} {timeframe} {status}")
            return log_record
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"データ収集ログ記録エラー: {e}")
            raise
    
    def get_recent_logs(self, limit: int = 100) -> List[DataCollectionLog]:
        """
        最近のデータ収集ログを取得
        
        Args:
            limit: 取得件数制限
            
        Returns:
            データ収集ログのリスト
        """
        try:
            return self.db.query(DataCollectionLog)\
                .order_by(desc(DataCollectionLog.created_at))\
                .limit(limit)\
                .all()
                
        except Exception as e:
            logger.error(f"データ収集ログ取得エラー: {e}")
            raise
