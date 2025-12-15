"""
Bybitデータ収集サービスの設定クラス

各データタイプ（ファンディングレート、オープンインタレストなど）の
固有設定を管理し、基底クラスでの共通処理を可能にします。
"""

from dataclasses import dataclass
from typing import Any, Type


@dataclass
class DataServiceConfig:
    """データサービスの設定"""

    # リポジトリ関連
    repository_class: Type[Any]
    get_timestamp_method_name: str

    # データ変換関連
    data_converter_class: Type[Any]
    converter_method_name: str

    # API関連
    fetch_history_method_name: str
    fetch_current_method_name: str

    # ページネーション設定
    pagination_strategy: str = "until"  # "until" or "time_range"
    default_limit: int = 100
    page_limit: int = 200
    max_pages: int = 50

    # データベース保存関連
    insert_method_name: str = "insert_data"

    # ログ用プレフィックス
    log_prefix: str = "DATA"


# 遅延インポートを使用してクラス参照を設定
def get_funding_rate_config():
    """ファンディングレート設定を取得"""
    from app.utils.data_conversion import FundingRateDataConverter
    from database.repositories.funding_rate_repository import FundingRateRepository

    return DataServiceConfig(
        repository_class=FundingRateRepository,
        get_timestamp_method_name="get_latest_funding_timestamp",
        data_converter_class=FundingRateDataConverter,
        converter_method_name="ccxt_to_db_format",
        fetch_history_method_name="fetch_funding_rate_history",
        fetch_current_method_name="fetch_funding_rate",
        pagination_strategy="until",
        default_limit=100,
        page_limit=200,
        max_pages=50,
        insert_method_name="insert_funding_rate_data",
        log_prefix="FR",
    )


def get_open_interest_config():
    """オープンインタレスト設定を取得"""
    from app.utils.data_conversion import OpenInterestDataConverter
    from database.repositories.open_interest_repository import OpenInterestRepository

    return DataServiceConfig(
        repository_class=OpenInterestRepository,
        get_timestamp_method_name="get_latest_open_interest_timestamp",
        data_converter_class=OpenInterestDataConverter,
        converter_method_name="ccxt_to_db_format",
        fetch_history_method_name="fetch_open_interest_history",
        fetch_current_method_name="fetch_open_interest",
        pagination_strategy="time_range",
        default_limit=100,
        page_limit=200,
        max_pages=500,
        insert_method_name="insert_open_interest_data",
        log_prefix="OI",
    )



