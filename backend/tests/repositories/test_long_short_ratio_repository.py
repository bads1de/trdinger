"""
LongShortRatioRepositoryのテスト
"""

from datetime import datetime, timezone
from typing import List, Dict, Any
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest
from sqlalchemy.orm import Session

from database.models import LongShortRatioData
from database.repositories.long_short_ratio_repository import LongShortRatioRepository

# テスト用データ
SAMPLE_RECORDS = [
    {
        "symbol": "BTC/USDT:USDT",
        "period": "1h",
        "buyRatio": "0.6",
        "sellRatio": "0.4",
        "timestamp": "1609459200000",  # 2021-01-01 00:00:00 UTC
    },
    {
        "symbol": "BTC/USDT:USDT",
        "period": "1h",
        "buyRatio": "0.55",
        "sellRatio": "0.45",
        "timestamp": "1609462800000",  # 2021-01-01 01:00:00 UTC
    },
]

@pytest.fixture
def mock_session() -> MagicMock:
    """モックDBセッション"""
    session = MagicMock(spec=Session)
    session.execute = MagicMock()
    session.commit = MagicMock()
    session.rollback = MagicMock()
    session.scalar = MagicMock()
    session.scalars = MagicMock()

    # bind属性をモック化
    mock_bind = MagicMock()
    mock_engine = MagicMock()
    mock_dialect = MagicMock()
    mock_dialect.name = "sqlite"
    mock_engine.dialect = mock_dialect
    mock_bind.engine = mock_engine
    session.bind = mock_bind

    return session

@pytest.fixture
def repository(mock_session: MagicMock):
    return LongShortRatioRepository(mock_session)

def test_insert_long_short_ratio_data(repository, mock_session):
    """データの挿入テスト"""
    mock_result = MagicMock()
    mock_result.rowcount = 2
    mock_session.execute.return_value = mock_result
    
    count = repository.insert_long_short_ratio_data(SAMPLE_RECORDS)
    assert count == 2
    
    # executeが呼ばれたか確認
    mock_session.execute.assert_called()
    mock_session.commit.assert_called()

def test_get_long_short_ratio_data(repository, mock_session):
    """データの取得テスト"""
    # モックデータの準備
    mock_data = [
        LongShortRatioData(
            symbol="BTC/USDT:USDT",
            period="1h",
            buy_ratio=0.6,
            sell_ratio=0.4,
            timestamp=datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        ),
        LongShortRatioData(
            symbol="BTC/USDT:USDT",
            period="1h",
            buy_ratio=0.55,
            sell_ratio=0.45,
            timestamp=datetime(2021, 1, 1, 1, 0, 0, tzinfo=timezone.utc)
        )
    ]
    
    mock_scalars = MagicMock()
    mock_scalars.all.return_value = mock_data
    mock_session.scalars.return_value = mock_scalars

    # 全件取得
    results = repository.get_long_short_ratio_data(
        symbol="BTC/USDT:USDT",
        period="1h"
    )
    assert len(results) == 2
    assert results[0].symbol == "BTC/USDT:USDT"
    
    # クエリが発行されたか確認
    mock_session.scalars.assert_called()

def test_get_latest_ratio(repository, mock_session):
    """最新データの取得テスト"""
    mock_record = LongShortRatioData(
        symbol="BTC/USDT:USDT",
        period="1h",
        buy_ratio=0.55,
        sell_ratio=0.45,
        timestamp=datetime(2021, 1, 1, 1, 0, 0, tzinfo=timezone.utc)
    )
    
    mock_scalars = MagicMock()
    mock_scalars.all.return_value = [mock_record]
    mock_session.scalars.return_value = mock_scalars
    
    latest = repository.get_latest_ratio("BTC/USDT:USDT", "1h")
    assert latest is not None
    assert latest.timestamp == datetime(2021, 1, 1, 1, 0, 0, tzinfo=timezone.utc)

def test_get_ratio_dataframe(repository, mock_session):
    """DataFrame取得テスト"""
    mock_data = [
        LongShortRatioData(
            symbol="BTC/USDT:USDT",
            period="1h",
            buy_ratio=0.6,
            sell_ratio=0.4,
            timestamp=datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        )
    ]
    
    mock_scalars = MagicMock()
    mock_scalars.all.return_value = mock_data
    mock_session.scalars.return_value = mock_scalars
    
    df = repository.get_ratio_dataframe("BTC/USDT:USDT", "1h")
    assert not df.empty
    assert "ls_ratio" in df.columns
    assert df.iloc[0]["ls_ratio"] == pytest.approx(1.5)




