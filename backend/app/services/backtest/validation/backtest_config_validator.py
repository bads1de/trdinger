"""
バックテスト設定検証サービス

"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class BacktestConfigValidationError(Exception):
    """バックテスト設定検証エラー"""

    def __init__(self, message: str, errors: Optional[List[str]] = None):
        super().__init__(message)
        self.errors = errors or []


class BacktestConfigValidator:
    """
    バックテスト設定検証サービス

    設定の妥当性検証を専門に担当します。
    """

    REQUIRED_FIELDS = [
        "strategy_name",
        "symbol",
        "timeframe",
        "start_date",
        "end_date",
        "initial_capital",
        "commission_rate",
    ]

    VALID_TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]

    def validate_config(self, config: Dict[str, Any]) -> None:
        """
        バックテスト設定を検証

        Args:
            config: バックテスト設定

        Raises:
            BacktestConfigValidationError: 設定が無効な場合
        """
        errors = []

        # 必須フィールドの検証
        errors.extend(self._validate_required_fields(config))

        # 各フィールドの詳細検証
        if not errors:  # 必須フィールドがある場合のみ詳細検証
            errors.extend(self._validate_field_values(config))

        if errors:
            raise BacktestConfigValidationError(
                f"バックテスト設定が無効です: {', '.join(errors)}", errors
            )

    def _validate_required_fields(self, config: Dict[str, Any]) -> List[str]:
        """必須フィールドの検証"""
        errors = []

        for field in self.REQUIRED_FIELDS:
            if field not in config:
                errors.append(f"必須フィールド '{field}' が見つかりません")
            elif config[field] is None:
                errors.append(f"フィールド '{field}' がNullです")

        return errors

    def _validate_field_values(self, config: Dict[str, Any]) -> List[str]:
        """各フィールド値の詳細検証"""
        errors = []

        # 戦略名の検証
        if (
            not isinstance(config["strategy_name"], str)
            or not config["strategy_name"].strip()
        ):
            errors.append("strategy_nameは空でない文字列である必要があります")

        # シンボルの検証
        if not isinstance(config["symbol"], str) or not config["symbol"].strip():
            errors.append("symbolは空でない文字列である必要があります")

        # 時間軸の検証
        if config["timeframe"] not in self.VALID_TIMEFRAMES:
            errors.append(
                f"timeframeは {self.VALID_TIMEFRAMES} のいずれかである必要があります"
            )

        # 日付の検証
        errors.extend(self._validate_dates(config))

        # 数値フィールドの検証
        errors.extend(self._validate_numeric_fields(config))

        # 戦略設定の検証
        errors.extend(self._validate_strategy_config(config))

        return errors

    def _validate_dates(self, config: Dict[str, Any]) -> List[str]:
        """日付フィールドの検証"""
        errors = []

        try:
            start_date = self._parse_date(config["start_date"])
            end_date = self._parse_date(config["end_date"])

            if start_date >= end_date:
                errors.append("start_dateはend_dateより前である必要があります")

            # 未来の日付チェック
            now = datetime.now()
            if end_date > now:
                errors.append("end_dateは現在時刻より前である必要があります")

        except ValueError as e:
            errors.append(f"日付形式が無効です: {e}")

        return errors

    def _validate_numeric_fields(self, config: Dict[str, Any]) -> List[str]:
        """数値フィールドの検証"""
        errors = []

        # 初期資金の検証
        try:
            initial_capital = float(config["initial_capital"])
            if initial_capital <= 0:
                errors.append("initial_capitalは正の数値である必要があります")
        except (ValueError, TypeError):
            errors.append("initial_capitalは数値である必要があります")

        # 手数料率の検証
        try:
            commission_rate = float(config["commission_rate"])
            if not 0 <= commission_rate <= 1:
                errors.append("commission_rateは0から1の間である必要があります")
        except (ValueError, TypeError):
            errors.append("commission_rateは数値である必要があります")

        return errors

    def _validate_strategy_config(self, config: Dict[str, Any]) -> List[str]:
        """戦略設定の検証"""
        errors = []

        strategy_config = config.get("strategy_config")
        if strategy_config is not None:
            if not isinstance(strategy_config, dict):
                errors.append("strategy_configは辞書である必要があります")
            else:
                # 戦略タイプの検証
                strategy_type = strategy_config.get("strategy_type")
                if strategy_type and not isinstance(strategy_type, str):
                    errors.append("strategy_typeは文字列である必要があります")

        return errors

    def _parse_date(self, date_value: Any) -> datetime:
        """日付値をdatetimeオブジェクトに変換"""
        if isinstance(date_value, datetime):
            return date_value
        elif isinstance(date_value, str):
            try:
                return datetime.fromisoformat(date_value.replace("Z", "+00:00"))
            except ValueError:
                # ISO形式でない場合の代替パース
                return datetime.strptime(date_value, "%Y-%m-%d")
        else:
            raise ValueError(f"サポートされていない日付形式: {type(date_value)}")
