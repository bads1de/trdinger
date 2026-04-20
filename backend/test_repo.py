from database.connection import SessionLocal
from database.repositories.ohlcv_repository import OHLCVRepository
from datetime import datetime

db = SessionLocal()
repo = OHLCVRepository(db)
start = datetime(2024, 1, 1)
end = datetime(2024, 1, 31)
data = repo.get_ohlcv_data('BTC/USDT:USDT', '4h', start, end)
print(f'Data count: {len(data)}')
if data:
    print(f'First timestamp: {data[0].timestamp}')
    print(f'Last timestamp: {data[-1].timestamp}')
db.close()
