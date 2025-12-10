import ccxt
import time
from datetime import datetime, timezone

def check_history():
    bybit = ccxt.bybit()
    symbol = 'BTCUSDT'
    category = 'linear'
    period = '1d' # 1日足でチェック

    # 検証する日付リスト (UTC)
    dates_to_check = [
        datetime(2020, 1, 1, tzinfo=timezone.utc),
        datetime(2020, 6, 1, tzinfo=timezone.utc),
        datetime(2020, 7, 1, tzinfo=timezone.utc),
        datetime(2020, 8, 1, tzinfo=timezone.utc),
        datetime(2020, 10, 1, tzinfo=timezone.utc),
        datetime(2020, 10, 15, tzinfo=timezone.utc),
        datetime(2021, 1, 1, tzinfo=timezone.utc),
    ]

    print(f"Checking Long/Short Ratio history for {symbol} ({category})...\n")

    for date in dates_to_check:
        start_ts = int(date.timestamp() * 1000)
        end_ts = start_ts + (24 * 60 * 60 * 1000) - 1 # 1日の終わりまで
        
        try:
            # Bybit V5 API: GET /v5/market/account-ratio
            # startTimeとendTimeで期間を限定
            params = {
                'category': category,
                'symbol': symbol,
                'period': period,
                'startTime': start_ts,
                'endTime': end_ts,
                'limit': 1
            }
            
            response = bybit.publicGetV5MarketAccountRatio(params)
            
            result_list = response.get('result', {}).get('list', [])
            
            if result_list:
                data_entry = result_list[0]
                data_ts = int(data_entry['timestamp'])
                data_date = datetime.fromtimestamp(data_ts / 1000, tz=timezone.utc)
                print(f"[OK] {date.strftime('%Y-%m-%d')}: Data found! (Data Date: {data_date.strftime('%Y-%m-%d')})")
                print(f"  Sample Data Entry: {data_entry}") # データの中身を表示
            else:
                print(f"[NG] {date.strftime('%Y-%m-%d')}: No data.")
                
        except Exception as e:
            print(f"[ERR] {date.strftime('%Y-%m-%d')}: Error - {e}")
        
        time.sleep(0.2)

if __name__ == "__main__":
    check_history()
