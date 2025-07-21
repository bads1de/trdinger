"""
設定バリデーター

設定値の妥当性を検証するためのバリデーションロジックを提供します。
単一責任原則に従い、バリデーション機能のみを担当します。
"""

from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class MarketDataValidator:
    """市場データ設定のバリデーター"""

    @staticmethod
    def validate_symbol(symbol: str, supported_symbols: List[str]) -> bool:
        """
        シンボルが有効かどうかを検証

        Args:
            symbol: 検証するシンボル
            supported_symbols: サポートされているシンボルのリスト

        Returns:
            有効な場合True、無効な場合False
        """
        return symbol in supported_symbols

    @staticmethod
    def validate_timeframe(timeframe: str, supported_timeframes: List[str]) -> bool:
        """
        時間軸が有効かどうかを検証

        Args:
            timeframe: 検証する時間軸
            supported_timeframes: サポートされている時間軸のリスト

        Returns:
            有効な場合True、無効な場合False
        """
        return timeframe in supported_timeframes

    @staticmethod
    def validate_limit(limit: int, min_limit: int, max_limit: int) -> bool:
        """
        制限値が有効かどうかを検証

        Args:
            limit: 検証する制限値
            min_limit: 最小制限値
            max_limit: 最大制限値

        Returns:
            有効な場合True、無効な場合False
        """
        return min_limit <= limit <= max_limit

    @staticmethod
    def normalize_symbol(
        symbol: str, symbol_mapping: Dict[str, str], supported_symbols: List[str]
    ) -> str:
        """
        シンボルを正規化

        Args:
            symbol: 正規化するシンボル
            symbol_mapping: シンボルマッピング辞書
            supported_symbols: サポートされているシンボルのリスト

        Returns:
            正規化されたシンボル

        Raises:
            ValueError: サポートされていないシンボルの場合
        """
        # 大文字に変換し、空白を除去
        symbol = symbol.strip().upper()

        # マッピングテーブルから検索
        if symbol in symbol_mapping:
            normalized = symbol_mapping[symbol]
        else:
            normalized = symbol

        # サポートされているシンボルかチェック
        if normalized not in supported_symbols:
            raise ValueError(
                f"サポートされていないシンボルです: '{symbol}'。"
                f"サポートされているシンボルは {', '.join(supported_symbols)} です。"
            )

        return normalized


class MLConfigValidator:
    """ML設定のバリデーター"""

    @staticmethod
    def validate_predictions(predictions: Dict[str, float]) -> bool:
        """
        予測値の妥当性を検証

        Args:
            predictions: 予測値の辞書

        Returns:
            有効な場合True、無効な場合False
        """
        try:
            # 必要なキーの存在確認
            required_keys = ["up", "down", "range"]
            if not all(key in predictions for key in required_keys):
                return False

            # 値の範囲確認（0-1の範囲）
            for key, value in predictions.items():
                if not isinstance(value, (int, float)):
                    return False
                if not (0.0 <= value <= 1.0):
                    return False

            # 合計値の確認（0.8-1.2の範囲）
            total = sum(predictions.values())
            if not (0.8 <= total <= 1.2):
                return False

            return True

        except Exception:
            return False

    @staticmethod
    def validate_probability_range(
        min_prob: float, max_prob: float, sum_min: float, sum_max: float
    ) -> bool:
        """
        確率範囲設定の妥当性を検証

        Args:
            min_prob: 最小確率
            max_prob: 最大確率
            sum_min: 合計最小値
            sum_max: 合計最大値

        Returns:
            有効な場合True、無効な場合False
        """
        try:
            # 基本範囲の確認
            if not (0.0 <= min_prob <= max_prob <= 1.0):
                return False

            # 合計範囲の確認
            if not (0.0 < sum_min <= sum_max <= 2.0):
                return False

            return True

        except Exception:
            return False

    @staticmethod
    def validate_data_processing_config(
        max_ohlcv_rows: int,
        max_feature_rows: int,
        feature_timeout: int,
        training_timeout: int,
        prediction_timeout: int,
    ) -> List[str]:
        """
        データ処理設定の妥当性を検証

        Args:
            max_ohlcv_rows: 最大OHLCV行数
            max_feature_rows: 最大特徴量行数
            feature_timeout: 特徴量計算タイムアウト
            training_timeout: 学習タイムアウト
            prediction_timeout: 予測タイムアウト

        Returns:
            エラーメッセージのリスト（空の場合は有効）
        """
        errors = []

        if max_ohlcv_rows <= 0:
            errors.append("最大OHLCV行数は正の値である必要があります")

        if max_feature_rows <= 0:
            errors.append("最大特徴量行数は正の値である必要があります")

        if feature_timeout <= 0:
            errors.append("特徴量計算タイムアウトは正の値である必要があります")

        if training_timeout <= 0:
            errors.append("学習タイムアウトは正の値である必要があります")

        if prediction_timeout <= 0:
            errors.append("予測タイムアウトは正の値である必要があります")

        # 論理的な関係の確認（制限を緩和）
        # 制限を外したため、厳密な関係チェックは不要
        # if max_feature_rows < max_ohlcv_rows:
        #     errors.append("最大特徴量行数は最大OHLCV行数以上である必要があります")

        # if training_timeout < feature_timeout:
        #     errors.append(
        #         "学習タイムアウトは特徴量計算タイムアウト以上である必要があります"
        #     )

        return errors

    @staticmethod
    def validate_model_config(
        model_save_path: str,
        max_versions: int,
        retention_days: int,
    ) -> List[str]:
        """
        モデル設定の妥当性を検証

        Args:
            model_save_path: モデル保存パス
            max_versions: 最大バージョン数
            retention_days: 保持日数

        Returns:
            エラーメッセージのリスト（空の場合は有効）
        """
        errors = []

        if not model_save_path or not model_save_path.strip():
            errors.append("モデル保存パスが指定されていません")

        if max_versions <= 0:
            errors.append("最大バージョン数は正の値である必要があります")

        if retention_days <= 0:
            errors.append("保持日数は正の値である必要があります")

        # パスの重複確認

        return errors


class DatabaseValidator:
    """データベース設定のバリデーター"""

    @staticmethod
    def validate_connection_params(
        host: str, port: int, name: str, user: str
    ) -> List[str]:
        """
        データベース接続パラメータの妥当性を検証

        Args:
            host: ホスト名
            port: ポート番号
            name: データベース名
            user: ユーザー名

        Returns:
            エラーメッセージのリスト（空の場合は有効）
        """
        errors = []

        if not host or not host.strip():
            errors.append("ホスト名が指定されていません")

        if not (1 <= port <= 65535):
            errors.append("ポート番号は1-65535の範囲である必要があります")

        if not name or not name.strip():
            errors.append("データベース名が指定されていません")

        if not user or not user.strip():
            errors.append("ユーザー名が指定されていません")

        return errors


class AppValidator:
    """アプリケーション設定のバリデーター"""

    @staticmethod
    def validate_server_config(host: str, port: int) -> List[str]:
        """
        サーバー設定の妥当性を検証

        Args:
            host: ホスト名
            port: ポート番号

        Returns:
            エラーメッセージのリスト（空の場合は有効）
        """
        errors = []

        if not host or not host.strip():
            errors.append("ホスト名が指定されていません")

        if not (1 <= port <= 65535):
            errors.append("ポート番号は1-65535の範囲である必要があります")

        # 予約ポートの確認
        if port < 1024:
            logger.warning(
                f"ポート{port}は予約ポートです。管理者権限が必要な場合があります。"
            )

        return errors

    @staticmethod
    def validate_cors_origins(origins: List[str]) -> List[str]:
        """
        CORS設定の妥当性を検証

        Args:
            origins: CORS許可オリジンのリスト

        Returns:
            エラーメッセージのリスト（空の場合は有効）
        """
        errors = []

        if not origins:
            errors.append("CORS許可オリジンが指定されていません")

        for origin in origins:
            if not origin or not origin.strip():
                errors.append("空のCORSオリジンが含まれています")
            elif not (
                origin.startswith("http://")
                or origin.startswith("https://")
                or origin == "*"
            ):
                errors.append(f"無効なCORSオリジン形式: {origin}")

        return errors
