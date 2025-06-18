"""
基底戦略クラス

backtesting.pyライブラリを使用した戦略の基底クラスを定義します。
"""

from backtesting import Strategy
from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd


class BaseStrategy(Strategy, ABC):
    """
    基底戦略クラス

    すべてのバックテスト戦略はこのクラスを継承します。
    backtesting.pyのStrategyクラスを拡張し、共通機能を提供します。
    """

    def __init__(self, broker, data, params):
        """初期化"""
        super().__init__(broker, data, params)
        self._strategy_name = self.__class__.__name__
        self._parameters = {}

    @abstractmethod
    def init(self):
        """
        指標の初期化

        戦略で使用するテクニカル指標を初期化します。
        各戦略で実装が必要です。
        """

    @abstractmethod
    def next(self):
        """
        売買ロジック

        各バーで実行される売買判定ロジックです。
        各戦略で実装が必要です。
        """

    def get_strategy_info(self) -> Dict[str, Any]:
        """
        戦略情報を取得

        Returns:
            戦略の基本情報を含む辞書
        """
        return {
            "name": self._strategy_name,
            "parameters": self._get_parameters(),
            "description": self.__doc__ or "No description available",
        }

    def _get_parameters(self) -> Dict[str, Any]:
        """
        戦略パラメータを取得

        Returns:
            クラス変数として定義されたパラメータの辞書
        """
        parameters = {}
        for attr_name in dir(self):
            if not attr_name.startswith("_") and not callable(getattr(self, attr_name)):
                attr_value = getattr(self, attr_name)
                if isinstance(attr_value, (int, float, str, bool)):
                    parameters[attr_name] = attr_value
        return parameters

    def validate_parameters(self) -> bool:
        """
        パラメータの妥当性を検証

        Returns:
            パラメータが有効な場合True

        Raises:
            ValueError: パラメータが無効な場合
        """
        # 基本的な検証（各戦略でオーバーライド可能）
        return True

    def log_trade_signal(self, signal_type: str, price: float, reason: str = ""):
        """
        取引シグナルをログ出力

        Args:
            signal_type: シグナルタイプ（'BUY', 'SELL', 'CLOSE'）
            price: 価格
            reason: シグナルの理由
        """
        timestamp = self.data.index[-1] if hasattr(self.data, "index") else "Unknown"
        print(f"[{timestamp}] {signal_type} signal at {price:.2f} - {reason}")

    def get_current_indicators(self) -> Dict[str, float]:
        """
        現在の指標値を取得

        Returns:
            現在の指標値を含む辞書
        """
        indicators = {}

        # 戦略で定義された指標を取得
        for attr_name in dir(self):
            if not attr_name.startswith("_"):
                attr = getattr(self, attr_name)
                if hasattr(attr, "__len__") and hasattr(attr, "__getitem__"):
                    try:
                        # 最新の値を取得
                        current_value = attr[-1]
                        if isinstance(current_value, (int, float)) and not pd.isna(
                            current_value
                        ):
                            indicators[attr_name] = float(current_value)
                    except (IndexError, TypeError):
                        continue

        return indicators

    def calculate_position_size(self, price: float, risk_percent: float = 1.0) -> float:
        """
        ポジションサイズを計算

        Args:
            price: エントリー価格
            risk_percent: リスク許容度（%）

        Returns:
            ポジションサイズ
        """
        # 利用可能な資金
        available_cash = self.equity

        # リスク金額を計算
        risk_amount = available_cash * (risk_percent / 100)

        # ポジションサイズを計算（簡単な例）
        position_size = risk_amount / price

        return position_size

    def is_market_condition_favorable(self) -> bool:
        """
        市場状況が戦略に適しているかを判定

        Returns:
            市場状況が良好な場合True
        """
        # 基本実装（各戦略でオーバーライド可能）
        return True

    def should_exit_position(self) -> bool:
        """
        ポジション決済の判定

        Returns:
            決済すべき場合True
        """
        # 基本実装（各戦略でオーバーライド可能）
        return False
