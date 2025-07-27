"""
一括差分更新APIエンドポイントのテスト
"""

import asyncio
import aiohttp
import json


async def test_bulk_incremental_api():
    """一括差分更新APIのテスト"""

    print("=" * 60)
    print("一括差分更新API テスト開始")
    print("=" * 60)

    # APIエンドポイント
    url = "http://localhost:8000/api/data-collection/bulk-incremental-update"

    try:
        async with aiohttp.ClientSession() as session:
            print(f"\n📡 APIリクエスト送信: {url}")

            async with session.post(
                url, params={"symbol": "BTC/USDT:USDT"}
            ) as response:
                print(f"📊 レスポンスステータス: {response.status}")

                if response.status == 200:
                    data = await response.json()
                    print(f"✅ 成功: {data.get('success', False)}")
                    print(f"📝 メッセージ: {data.get('message', 'No message')}")

                    # 詳細結果を表示
                    if "data" in data:
                        result_data = data["data"]
                        print(f"\n📊 更新結果詳細:")

                        # Fear & Greed Index結果
                        if "fear_greed_index" in result_data:
                            fg_result = result_data["fear_greed_index"]
                            print(f"  😨 Fear & Greed Index:")
                            print(f"    成功: {fg_result.get('success', False)}")
                            print(f"    件数: {fg_result.get('inserted_count', 0)}件")
                            print(
                                f"    メッセージ: {fg_result.get('message', 'No message')}"
                            )

                            if not fg_result.get("success", False):
                                print(
                                    f"    ❌ エラー: {fg_result.get('error', 'Unknown error')}"
                                )

                        # その他の結果も表示
                        for key, value in result_data.items():
                            if key != "fear_greed_index":
                                print(f"  📈 {key}: {value}")

                else:
                    error_text = await response.text()
                    print(f"❌ エラー: {response.status}")
                    print(f"📝 エラー詳細: {error_text}")

    except aiohttp.ClientConnectorError:
        print(
            "❌ サーバーに接続できません。サーバーが起動していることを確認してください。"
        )
        print("💡 サーバー起動コマンド: python main.py")

    except Exception as e:
        print(f"❌ テスト中にエラーが発生しました: {e}")

    print("\n" + "=" * 60)
    print("一括差分更新API テスト完了")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_bulk_incremental_api())
