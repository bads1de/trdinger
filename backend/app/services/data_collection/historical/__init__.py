"""
Historical Data Collection パッケージ

バックテスト用の一括市場データ収集サービスを提供します。
過去のOHLCVデータを指定された期間で一括取得し、データベースに保存します。

主なコンポーネント:
- historical_data_service.py: 履歴データ収集サービス（期間指定、複数時間軸対応）
"""

from .historical_data_service import HistoricalDataService

__all__ = [
    "HistoricalDataService",
]
