import asyncio
from data_collector.external_market_collector import ExternalMarketDataCollector


async def test_status():
    async with ExternalMarketDataCollector() as collector:
        status = await collector.get_external_market_data_status()
        print("外部市場データ状態:")
        print(f"  成功: {status['success']}")
        if status["success"]:
            stats = status["statistics"]
            print(f"  データ件数: {stats['count']}")
            print(f"  シンボル: {stats['symbols']}")
            print(f"  最新タイムスタンプ: {status.get('latest_timestamp', 'なし')}")


print("=== 外部市場データテスト ===")
asyncio.run(test_status())
print("✅ テスト完了")
