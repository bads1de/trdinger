"""
Fear & Greed Index 実データテスト
"""

import asyncio
import logging
from datetime import datetime, timezone
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.connection import Base
from database.repositories.fear_greed_repository import FearGreedIndexRepository
from app.core.services.data_collection.fear_greed.fear_greed_service import (
    FearGreedIndexService,
)
from app.core.services.data_collection.orchestration.fear_greed_orchestration_service import (
    FearGreedOrchestrationService,
)

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_real_fear_greed_data():
    """実際のFear & Greed Indexデータを取得してテスト"""

    # インメモリデータベースを作成
    engine = create_engine("sqlite:///:memory:", echo=True)
    Base.metadata.create_all(engine)

    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()

    try:
        print("=" * 60)
        print("Fear & Greed Index 実データテスト開始")
        print("=" * 60)

        # 1. サービスを使って実際のデータを取得
        print("\n1. 実際のAPIからデータを取得中...")
        async with FearGreedIndexService() as service:
            try:
                # APIからデータを取得
                api_data = await service.fetch_fear_greed_data(limit=10)
                print(f"✅ APIから {len(api_data)} 件のデータを取得しました")

                # 最初のデータを表示
                if api_data:
                    first_record = api_data[0]
                    print(
                        f"📊 最新データ: 値={first_record['value']}, 分類={first_record['value_classification']}"
                    )
                    print(f"📅 データ日時: {first_record['data_timestamp']}")
                    print(f"🕐 取得日時: {first_record['timestamp']}")

            except Exception as e:
                print(f"❌ APIデータ取得エラー: {e}")
                print("⚠️  モックデータでテストを続行します...")

                # モックデータを作成
                base_time = datetime.now(timezone.utc)
                api_data = [
                    {
                        "value": 25,
                        "value_classification": "Fear",
                        "data_timestamp": base_time.replace(
                            hour=0, minute=0, second=0, microsecond=0
                        ),
                        "timestamp": base_time,
                    },
                    {
                        "value": 75,
                        "value_classification": "Greed",
                        "data_timestamp": base_time.replace(
                            hour=0, minute=0, second=0, microsecond=0
                        )
                        - timedelta(days=1),
                        "timestamp": base_time,
                    },
                ]
                print(f"🔧 モックデータを {len(api_data)} 件作成しました")

        # 2. データベースに保存
        print("\n2. データベースに保存中...")
        repository = FearGreedIndexRepository(db)

        if api_data:
            inserted_count = repository.insert_fear_greed_data(api_data)
            print(f"✅ データベースに {inserted_count} 件保存しました")
        else:
            print("❌ 保存するデータがありません")
            return

        # 3. タイムスタンプ関連機能のテスト
        print("\n3. タイムスタンプ関連機能のテスト...")

        # 最新タイムスタンプを取得
        latest_timestamp = repository.get_latest_data_timestamp()
        print(f"📅 最新データタイムスタンプ: {latest_timestamp}")
        print(
            f"🌍 タイムゾーン情報: {latest_timestamp.tzinfo if latest_timestamp else 'None'}"
        )

        # データ範囲を取得
        data_range = repository.get_data_range()
        print(f"📊 データ範囲: {data_range}")

        # データ件数を取得
        data_count = repository.get_data_count()
        print(f"📈 総データ件数: {data_count}")

        # 4. データ取得テスト
        print("\n4. データ取得テスト...")

        # 全データを取得
        all_data = repository.get_fear_greed_data()
        print(f"📋 全データ件数: {len(all_data)}")

        for i, record in enumerate(all_data):
            print(f"  {i+1}. 値={record.value}, 分類={record.value_classification}")
            print(
                f"     データ日時={record.data_timestamp} (tzinfo: {record.data_timestamp.tzinfo})"
            )

        # 最新データを取得
        latest_data = repository.get_latest_fear_greed_data(limit=3)
        print(f"📊 最新データ {len(latest_data)} 件:")
        for i, record in enumerate(latest_data):
            print(f"  {i+1}. 値={record.value}, 日時={record.data_timestamp}")

        # 5. オーケストレーションサービスのテスト
        print("\n5. オーケストレーションサービスのテスト...")

        orchestration_service = FearGreedOrchestrationService()

        # ステータス取得
        status_result = await orchestration_service.get_fear_greed_data_status(db)
        print(f"📊 ステータス: {status_result['success']}")
        if status_result["success"]:
            status_data = status_result["data"]
            print(f"   総件数: {status_data['data_range']['total_count']}")
            print(f"   最新タイムスタンプ: {status_data['latest_timestamp']}")

        # データ取得
        data_result = await orchestration_service.get_fear_greed_data(db, limit=5)
        print(f"📋 データ取得: {data_result['success']}")
        if data_result["success"]:
            retrieved_data = data_result["data"]["data"]
            print(f"   取得件数: {len(retrieved_data)}")

        # 6. 差分更新のシミュレーション
        print("\n6. 差分更新のシミュレーション...")

        # 現在の最新タイムスタンプを記録
        before_latest = repository.get_latest_data_timestamp()
        print(f"📅 更新前の最新タイムスタンプ: {before_latest}")

        # 新しいデータを追加
        from datetime import timedelta

        new_data = [
            {
                "value": 50,
                "value_classification": "Neutral",
                "data_timestamp": datetime.now(timezone.utc).replace(microsecond=0),
                "timestamp": datetime.now(timezone.utc),
            }
        ]

        new_inserted = repository.insert_fear_greed_data(new_data)
        print(f"✅ 新しいデータを {new_inserted} 件追加しました")

        # 更新後の最新タイムスタンプを確認
        after_latest = repository.get_latest_data_timestamp()
        print(f"📅 更新後の最新タイムスタンプ: {after_latest}")

        # タイムスタンプ比較のテスト
        if before_latest and after_latest:
            print(f"🔄 タイムスタンプ比較: {after_latest > before_latest}")
            print(f"⏰ 時間差: {after_latest - before_latest}")

        print("\n" + "=" * 60)
        print("✅ Fear & Greed Index 実データテスト完了")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ テスト中にエラーが発生しました: {e}")
        import traceback

        traceback.print_exc()

    finally:
        db.close()


if __name__ == "__main__":
    # 必要なインポートを追加
    from datetime import timedelta

    # 非同期実行
    asyncio.run(test_real_fear_greed_data())
