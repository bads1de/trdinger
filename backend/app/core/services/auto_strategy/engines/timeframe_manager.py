"""
時間軸管理器

データベースから利用可能な時間軸を取得し、バックテスト設定を管理するモジュール。
"""

import random
import logging
from typing import List, Dict, Any
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)


class TimeframeManager:
    """
    時間軸管理器
    
    データベースから利用可能な時間軸を取得し、バックテスト設定を管理します。
    """

    def __init__(self):
        """初期化"""
        self.available_timeframes = self._get_available_timeframes()

    def _get_available_timeframes(self) -> List[str]:
        """
        データベースから利用可能な時間軸を取得

        Returns:
            利用可能な時間軸のリスト
        """
        try:
            from database.connection import SessionLocal
            from database.repositories.ohlcv_repository import OHLCVRepository

            db = SessionLocal()
            try:
                repo = OHLCVRepository(db)
                symbols = repo.get_available_symbols()

                # BTC/USDT系のシンボルを優先的に使用
                target_symbols = ["BTC/USDT:USDT", "BTC/USDT", "BTCUSDT"]
                available_timeframes = []

                for symbol in target_symbols:
                    if symbol in symbols:
                        timeframes = repo.get_available_timeframes(symbol)
                        if timeframes:
                            available_timeframes = timeframes
                            logger.info(
                                f"利用可能な時間軸を取得: {symbol} -> {timeframes}"
                            )
                            break

                if not available_timeframes:
                    # フォールバック: 最初に見つかったシンボルの時間軸を使用
                    if symbols:
                        first_symbol = symbols[0]
                        available_timeframes = repo.get_available_timeframes(
                            first_symbol
                        )
                        logger.info(
                            f"フォールバック時間軸を取得: {first_symbol} -> {available_timeframes}"
                        )

                if not available_timeframes:
                    # 最終フォールバック
                    available_timeframes = ["1d"]
                    logger.warning(
                        "データベースに時間軸データが見つからないため、デフォルト値を使用"
                    )

                return available_timeframes

            finally:
                db.close()

        except Exception as e:
            logger.error(f"利用可能な時間軸の取得エラー: {e}")
            # エラー時のフォールバック
            return ["1d"]

    def select_random_timeframe_config(
        self, base_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        利用可能なデータに基づいてランダムな時間足を選択し、適切な期間を設定

        Args:
            base_config: ベースとなるバックテスト設定

        Returns:
            時間足と期間が調整された設定
        """
        try:
            from database.connection import SessionLocal
            from database.repositories.ohlcv_repository import OHLCVRepository
            from app.config.market_config import MarketDataConfig

            db = SessionLocal()
            try:
                repo = OHLCVRepository(db)
                symbols = repo.get_available_symbols()

                # シンボルの正規化
                input_symbol = base_config.get("symbol", "BTC/USDT")
                try:
                    # MarketDataConfigを使用してシンボルを正規化
                    if input_symbol == "BTC/USDT":
                        normalized_symbol = "BTC/USDT:USDT"  # データベース形式に変換
                    else:
                        normalized_symbol = MarketDataConfig.normalize_symbol(
                            input_symbol
                        )
                except ValueError:
                    # 正規化に失敗した場合のフォールバック
                    normalized_symbol = "BTC/USDT:USDT"
                    logger.warning(
                        f"シンボル正規化失敗、フォールバック使用: {input_symbol} -> {normalized_symbol}"
                    )

                # BTC/USDT系のシンボルを優先的に使用
                target_symbols = ["BTC/USDT:USDT", "BTC/USDT", "BTCUSDT"]
                selected_symbol = normalized_symbol
                available_timeframes = []

                # 正規化されたシンボルが利用可能かチェック
                if selected_symbol in symbols:
                    available_timeframes = repo.get_available_timeframes(
                        selected_symbol
                    )
                    logger.info(f"正規化シンボル使用: {selected_symbol}")

                # 正規化シンボルが利用できない場合、優先シンボルを使用
                if not available_timeframes:
                    for symbol in target_symbols:
                        if symbol in symbols:
                            timeframes = repo.get_available_timeframes(symbol)
                            if timeframes:
                                selected_symbol = symbol
                                available_timeframes = timeframes
                                logger.info(f"シンボル変更: {input_symbol} -> {symbol}")
                                break

                # それでも見つからない場合、最初のシンボルを使用
                if not available_timeframes and symbols:
                    selected_symbol = symbols[0]
                    available_timeframes = repo.get_available_timeframes(
                        selected_symbol
                    )
                    logger.info(f"フォールバックシンボル使用: {selected_symbol}")

                # 時間軸をランダム選択
                if available_timeframes:
                    selected_timeframe = random.choice(available_timeframes)
                    logger.info(f"ランダム時間足選択: {selected_timeframe}")
                else:
                    selected_timeframe = "1d"  # 最終フォールバック
                    logger.warning("利用可能な時間足が見つからず、デフォルト使用: 1d")

                logger.info(
                    f"選択されたシンボル・時間軸: {selected_symbol} {selected_timeframe}"
                )

            finally:
                db.close()

        except Exception as e:
            logger.error(f"データベース確認エラー: {e}")
            # エラー時のフォールバック
            selected_symbol = "BTC/USDT:USDT"
            selected_timeframe = "1d"

        # 元の設定の日付範囲を使用（現在時刻ベースではなく）
        original_start = base_config.get("start_date")
        original_end = base_config.get("end_date")

        if original_start and original_end:
            # 元の設定に日付がある場合はそれを使用
            start_date = original_start
            end_date = original_end
            logger.info(f"元の日付範囲を使用: {start_date} ～ {end_date}")
        else:
            # 日付が指定されていない場合のみ、現在時刻を基準に設定
            end_date_dt = datetime.now(timezone.utc)

            # 時間足に応じて適切な期間を設定
            if selected_timeframe == "15m":
                start_date_dt = end_date_dt - timedelta(days=7)  # 15分足: 1週間
            elif selected_timeframe == "30m":
                start_date_dt = end_date_dt - timedelta(days=14)  # 30分足: 2週間
            elif selected_timeframe == "1h":
                start_date_dt = end_date_dt - timedelta(days=30)  # 1時間足: 1ヶ月
            elif selected_timeframe == "4h":
                start_date_dt = end_date_dt - timedelta(days=60)  # 4時間足: 2ヶ月
            else:  # 1d
                start_date_dt = end_date_dt - timedelta(days=90)  # 日足: 3ヶ月

            start_date = start_date_dt.isoformat()
            end_date = end_date_dt.isoformat()
            logger.info(f"自動生成日付範囲: {start_date} ～ {end_date}")

        # 設定をコピーして更新
        config = base_config.copy()
        config["symbol"] = selected_symbol
        config["timeframe"] = selected_timeframe
        config["start_date"] = start_date
        config["end_date"] = end_date

        return config

    def get_available_timeframes(self) -> List[str]:
        """利用可能な時間軸を取得"""
        return self.available_timeframes
