"""
バックテスト環境変数設定

バックテストのデフォルトパラメータを環境変数から読み込みます。
unified_config.py 経由でアプリケーション全体に提供されます。
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BacktestConfig(BaseSettings):
    """バックテスト設定。

    バックテストの初期資金、手数料、結果取得制限などの設定を管理します。
    """

    # デフォルトパラメータ
    default_initial_capital: float = Field(default=10000.0, description="初期資金")
    default_commission_rate: float = Field(default=0.001, description="手数料率")

    # バックテスト実行設定
    max_results_limit: int = Field(default=50, description="結果取得最大件数")
    default_results_limit: int = Field(default=20, description="デフォルト結果件数")

    model_config = SettingsConfigDict(env_prefix="BACKTEST_", extra="ignore")
