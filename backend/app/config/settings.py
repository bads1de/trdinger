"""
アプリケーション設定管理

環境変数とアプリケーション設定を管理します。
"""

from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """アプリケーション設定クラス"""

    # アプリケーション設定
    app_name: str = Field(default="Trdinger Trading API", alias="APP_NAME")
    app_version: str = Field(default="1.0.0", alias="APP_VERSION")
    debug: bool = Field(default=False, alias="DEBUG")

    # サーバー設定
    host: str = Field(default="127.0.0.1", alias="HOST")
    port: int = Field(default=8000, alias="PORT")

    # CORS設定
    cors_origins: list[str] = Field(
        default=["http://localhost:3000"], alias="CORS_ORIGINS"
    )

    # データベース設定
    database_url: Optional[str] = Field(default=None, alias="DATABASE_URL")
    db_host: str = Field(default="localhost", alias="DB_HOST")
    db_port: int = Field(default=5432, alias="DB_PORT")
    db_name: str = Field(default="trdinger", alias="DB_NAME")
    db_user: str = Field(default="postgres", alias="DB_USER")
    db_password: str = Field(default="", alias="DB_PASSWORD")

    # ログ設定
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        alias="LOG_FORMAT",
    )

    # セキュリティ設定
    secret_key: str = Field(default="your-secret-key-here", alias="SECRET_KEY")

    # GA フォールバック設定
    ga_fallback_symbol: str = Field(default="BTC/USDT", alias="GA_FALLBACK_SYMBOL")
    ga_fallback_timeframe: str = Field(default="1d", alias="GA_FALLBACK_TIMEFRAME")
    ga_fallback_start_date: str = Field(
        default="2024-01-01", alias="GA_FALLBACK_START_DATE"
    )
    ga_fallback_end_date: str = Field(
        default="2024-04-09", alias="GA_FALLBACK_END_DATE"
    )
    ga_fallback_initial_capital: float = Field(
        default=100000.0, alias="GA_FALLBACK_INITIAL_CAPITAL"
    )
    ga_fallback_commission_rate: float = Field(
        default=0.001, alias="GA_FALLBACK_COMMISSION_RATE"
    )

    @property
    def database_url_complete(self) -> str:
        """完全なデータベースURLを生成"""
        if self.database_url:
            return self.database_url
        return (
            f"postgresql://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # 追加の環境変数を無視


# 設定のシングルトンインスタンス
settings = Settings()
