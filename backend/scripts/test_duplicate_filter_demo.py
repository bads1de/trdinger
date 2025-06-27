"""
重複ログフィルターのデモスクリプト
実際のログ出力で重複フィルターの動作を確認します。
"""

import logging
import time
import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.utils.duplicate_filter_handler import DuplicateFilterHandler


def setup_demo_logging():
    """デモ用のログ設定"""
    # ルートロガーを取得
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # 既存のハンドラーをクリア
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # コンソールハンドラーを作成
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # フォーマッターを設定
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)

    # 重複フィルターを作成（1秒間隔で同じメッセージをフィルタリング）
    duplicate_filter = DuplicateFilterHandler(capacity=50, interval=1.0)

    # フィルターをコンソールハンドラーに追加
    console_handler.addFilter(duplicate_filter)

    # ハンドラーをルートロガーに追加
    root_logger.addHandler(console_handler)

    return duplicate_filter


def demo_duplicate_filtering():
    """重複フィルタリングのデモ"""
    print("=== 重複ログフィルターのデモ ===\n")

    # ログ設定
    duplicate_filter = setup_demo_logging()
    logger = logging.getLogger("demo")

    print("1. 通常のログメッセージ（重複なし）")
    logger.info("これは通常のログメッセージです")
    logger.info("これは別のログメッセージです")
    logger.warning("これは警告メッセージです")

    print("\n2. 重複ログメッセージのテスト")
    print("   同じメッセージを5回送信します（1回だけ表示されるはずです）")
    for i in range(5):
        logger.info("これは重複するメッセージです")
        time.sleep(0.1)  # 短い間隔で送信

    print("\n3. 一定時間経過後の同じメッセージ")
    print("   1.5秒待機後、同じメッセージを再送信します（再度表示されるはずです）")
    time.sleep(1.5)
    logger.info("これは重複するメッセージです")

    print("\n4. 異なるログレベルでの重複テスト")
    logger.error("エラーメッセージのテスト")
    logger.error("エラーメッセージのテスト")  # 重複でフィルタリング
    logger.warning("エラーメッセージのテスト")  # 異なるレベルなので表示される

    print("\n5. フィルター統計情報")
    stats = duplicate_filter.get_stats()
    print(f"   追跡中のメッセージ数: {stats['tracked_messages']}")
    print(f"   キャパシティ: {stats['capacity']}")

    print("\n6. キャッシュクリア後のテスト")
    duplicate_filter.clear_cache()
    logger.info("キャッシュクリア後のメッセージ")
    logger.info("キャッシュクリア後のメッセージ")  # 重複でフィルタリング

    print("\n7. 大量の異なるメッセージでキャパシティテスト")
    for i in range(10):
        logger.info(f"メッセージ番号 {i}")

    final_stats = duplicate_filter.get_stats()
    print(f"\n最終統計: 追跡中のメッセージ数: {final_stats['tracked_messages']}")

    print("\n=== デモ完了 ===")


def demo_real_world_scenario():
    """実際のシナリオでのデモ"""
    print("\n=== 実際のシナリオでのデモ ===")

    duplicate_filter = setup_demo_logging()
    logger = logging.getLogger("trading_system")

    print("\n仮想通貨取引システムのログシミュレーション:")

    # データ収集の重複ログをシミュレート
    print("\n1. データ収集の重複ログ")
    for i in range(3):
        logger.info("BTC/USDT の OHLCV データを収集中...")
        time.sleep(0.2)

    # 接続エラーの重複ログをシミュレート
    print("\n2. 接続エラーの重複ログ")
    for i in range(4):
        logger.error("取引所への接続に失敗しました")
        time.sleep(0.1)

    # 正常なログと重複ログの混在
    print("\n3. 正常なログと重複ログの混在")
    logger.info("取引戦略を開始します")
    logger.info("市場データを分析中...")
    logger.info("市場データを分析中...")  # 重複
    logger.info("取引シグナルを検出しました")
    logger.info("市場データを分析中...")  # 重複

    # 時間経過後の同じメッセージ
    print("\n4. 時間経過後の同じメッセージ")
    time.sleep(1.2)
    logger.info("市場データを分析中...")  # 時間経過後なので表示される

    stats = duplicate_filter.get_stats()
    print(f"\n最終統計: 追跡中のメッセージ数: {stats['tracked_messages']}")

    print("\n=== 実際のシナリオデモ完了 ===")


if __name__ == "__main__":
    try:
        demo_duplicate_filtering()
        demo_real_world_scenario()
    except KeyboardInterrupt:
        print("\n\nデモが中断されました。")
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        import traceback

        traceback.print_exc()
