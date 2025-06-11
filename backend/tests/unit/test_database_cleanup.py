"""
データベースクリア機能のテスト

TDD: まず失敗するテストを作成し、その後実装を行う
"""

import pytest
from datetime import datetime, timezone, timedelta
from sqlalchemy.orm import Session

from database.models import OHLCVData
from database.repositories.ohlcv_repository import OHLCVRepository
from database.connection import SessionLocal, Base, engine


class TestDatabaseCleanup:
    """データベースクリア機能のテストクラス"""

    @pytest.fixture(scope="function")
    def db_session(self):
        """テスト用データベースセッション"""
        # テスト用のテーブルを作成
        Base.metadata.create_all(bind=engine)
        
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()
            # テスト後にテーブルをクリア
            Base.metadata.drop_all(bind=engine)

    @pytest.fixture
    def ohlcv_repo(self, db_session):
        """OHLCVリポジトリのフィクスチャ"""
        return OHLCVRepository(db_session)

    @pytest.fixture
    def sample_ohlcv_data(self, db_session):
        """テスト用のサンプルOHLCVデータを作成"""
        sample_data = []
        
        # 複数のシンボルと時間足でサンプルデータを作成
        symbols = ["BTC/USDT:USDT", "BTC/USDT", "ETH/USDT"]
        timeframes = ["1d", "4h", "1h", "30m", "15m"]
        
        base_time = datetime.now(timezone.utc) - timedelta(days=10)
        
        for symbol in symbols:
            for timeframe in timeframes:
                for i in range(5):  # 各組み合わせで5件のデータ
                    timestamp = base_time + timedelta(hours=i)
                    
                    data = OHLCVData(
                        symbol=symbol,
                        timeframe=timeframe,
                        timestamp=timestamp,
                        open=50000.0 + i * 100,
                        high=51000.0 + i * 100,
                        low=49000.0 + i * 100,
                        close=50500.0 + i * 100,
                        volume=1000.0 + i * 10
                    )
                    sample_data.append(data)
                    db_session.add(data)
        
        db_session.commit()
        return sample_data

    def test_clear_all_ohlcv_data_method_exists(self, ohlcv_repo):
        """clear_all_ohlcv_data メソッドが存在することをテスト"""
        # このテストは最初は失敗する（メソッドが存在しないため）
        assert hasattr(ohlcv_repo, 'clear_all_ohlcv_data'), \
            "OHLCVRepository に clear_all_ohlcv_data メソッドが存在しません"

    def test_clear_all_ohlcv_data_removes_all_records(self, ohlcv_repo, sample_ohlcv_data, db_session):
        """全てのOHLCVデータが削除されることをテスト"""
        # 事前条件: データが存在することを確認
        initial_count = db_session.query(OHLCVData).count()
        assert initial_count > 0, "テストデータが正しく作成されていません"
        
        # clear_all_ohlcv_data メソッドを実行
        deleted_count = ohlcv_repo.clear_all_ohlcv_data()
        
        # 検証: 全てのデータが削除されていることを確認
        remaining_count = db_session.query(OHLCVData).count()
        assert remaining_count == 0, "データが完全に削除されていません"
        assert deleted_count == initial_count, "削除件数が正しく返されていません"

    def test_clear_ohlcv_data_by_symbol_method_exists(self, ohlcv_repo):
        """clear_ohlcv_data_by_symbol メソッドが存在することをテスト"""
        # このテストは最初は失敗する（メソッドが存在しないため）
        assert hasattr(ohlcv_repo, 'clear_ohlcv_data_by_symbol'), \
            "OHLCVRepository に clear_ohlcv_data_by_symbol メソッドが存在しません"

    def test_clear_ohlcv_data_by_symbol_removes_specific_symbol(self, ohlcv_repo, sample_ohlcv_data, db_session):
        """特定のシンボルのデータのみが削除されることをテスト"""
        # 事前条件: 複数シンボルのデータが存在することを確認
        btc_usdt_count = db_session.query(OHLCVData).filter(
            OHLCVData.symbol == "BTC/USDT:USDT"
        ).count()
        other_count = db_session.query(OHLCVData).filter(
            OHLCVData.symbol != "BTC/USDT:USDT"
        ).count()
        
        assert btc_usdt_count > 0, "BTC/USDT:USDTのテストデータが存在しません"
        assert other_count > 0, "他のシンボルのテストデータが存在しません"
        
        # 特定シンボルのデータを削除
        deleted_count = ohlcv_repo.clear_ohlcv_data_by_symbol("BTC/USDT:USDT")
        
        # 検証: BTC/USDT:USDTのデータのみが削除されていることを確認
        remaining_btc_count = db_session.query(OHLCVData).filter(
            OHLCVData.symbol == "BTC/USDT:USDT"
        ).count()
        remaining_other_count = db_session.query(OHLCVData).filter(
            OHLCVData.symbol != "BTC/USDT:USDT"
        ).count()
        
        assert remaining_btc_count == 0, "BTC/USDT:USDTのデータが削除されていません"
        assert remaining_other_count == other_count, "他のシンボルのデータが削除されています"
        assert deleted_count == btc_usdt_count, "削除件数が正しく返されていません"

    def test_clear_ohlcv_data_by_timeframe_method_exists(self, ohlcv_repo):
        """clear_ohlcv_data_by_timeframe メソッドが存在することをテスト"""
        # このテストは最初は失敗する（メソッドが存在しないため）
        assert hasattr(ohlcv_repo, 'clear_ohlcv_data_by_timeframe'), \
            "OHLCVRepository に clear_ohlcv_data_by_timeframe メソッドが存在しません"

    def test_clear_ohlcv_data_by_timeframe_removes_specific_timeframe(self, ohlcv_repo, sample_ohlcv_data, db_session):
        """特定の時間足のデータのみが削除されることをテスト"""
        # 事前条件: 複数時間足のデータが存在することを確認
        daily_count = db_session.query(OHLCVData).filter(
            OHLCVData.timeframe == "1d"
        ).count()
        other_count = db_session.query(OHLCVData).filter(
            OHLCVData.timeframe != "1d"
        ).count()
        
        assert daily_count > 0, "1dのテストデータが存在しません"
        assert other_count > 0, "他の時間足のテストデータが存在しません"
        
        # 特定時間足のデータを削除
        deleted_count = ohlcv_repo.clear_ohlcv_data_by_timeframe("1d")
        
        # 検証: 1dのデータのみが削除されていることを確認
        remaining_daily_count = db_session.query(OHLCVData).filter(
            OHLCVData.timeframe == "1d"
        ).count()
        remaining_other_count = db_session.query(OHLCVData).filter(
            OHLCVData.timeframe != "1d"
        ).count()
        
        assert remaining_daily_count == 0, "1dのデータが削除されていません"
        assert remaining_other_count == other_count, "他の時間足のデータが削除されています"
        assert deleted_count == daily_count, "削除件数が正しく返されていません"

    def test_clear_empty_database_returns_zero(self, ohlcv_repo, db_session):
        """空のデータベースに対してクリア操作を実行した場合、0が返されることをテスト"""
        # 事前条件: データベースが空であることを確認
        initial_count = db_session.query(OHLCVData).count()
        assert initial_count == 0, "データベースが空ではありません"
        
        # 空のデータベースに対してクリア操作を実行
        deleted_count = ohlcv_repo.clear_all_ohlcv_data()
        
        # 検証: 0が返されることを確認
        assert deleted_count == 0, "空のデータベースに対するクリア操作で0以外が返されました"
