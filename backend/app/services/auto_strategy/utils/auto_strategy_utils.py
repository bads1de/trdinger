"""
Auto Strategy 統合ユーティリティ

分散していたユーティリティ関数を統合し、Auto Strategy専用の便利機能を提供します。
common_utils.py、strategy_gene_utils.py、gene_utils.pyの機能を統合しています。
"""

import logging
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)


class AutoStrategyUtils:
    """Auto Strategy専用ユーティリティクラス"""

    # === データ変換ユーティリティ ===

    @staticmethod
    def safe_convert_to_float(value: Any, default: float = 0.0) -> float:
        """安全なfloat変換（Auto Strategy用）"""
        try:
            if isinstance(value, str) and value.strip() == "":
                return default
            return float(value)
        except (ValueError, TypeError, AttributeError):
            logger.warning(f"float変換失敗: {value} -> {default}")
            return default

    @staticmethod
    def safe_convert_to_int(value: Any, default: int = 0) -> int:
        """安全なint変換（Auto Strategy用）"""
        try:
            if isinstance(value, str) and value.strip() == "":
                return default
            return int(value)
        except (ValueError, TypeError, AttributeError):
            logger.warning(f"int変換失敗: {value} -> {default}")
            return default

    @staticmethod
    def ensure_list(value: Any, default: Optional[List] = None) -> List:
        """値をリストに安全に変換"""
        if default is None:
            default = []

        if isinstance(value, list):
            return value
        elif value is None:
            return default
        else:
            return [value]

    @staticmethod
    def ensure_dict(value: Any, default: Optional[Dict] = None) -> Dict:
        """値を辞書に安全に変換"""
        if default is None:
            default = {}

        if isinstance(value, dict):
            return value
        else:
            return default

    @staticmethod
    def normalize_symbol(symbol: str) -> str:
        """シンボルを正規化"""
        if not symbol:
            return "BTC:USDT"

        if ":" not in symbol:
            return f"{symbol}:USDT"

        return symbol

    # === 戦略遺伝子関連ユーティリティ ===

    @staticmethod
    def create_default_strategy_gene(
        include_exit_conditions: bool = False, strategy_gene_class=None
    ):
        """
        デフォルト戦略遺伝子を作成（統合版）

        Args:
            include_exit_conditions: 出口条件を含めるか（TP/SL使用時はFalse）
            strategy_gene_class: StrategyGeneクラス（後方互換性のため）

        Returns:
            デフォルトの戦略遺伝子オブジェクト
        """
        try:
            from ..models.gene_strategy import StrategyGene, IndicatorGene, Condition

            # デフォルト指標
            indicators = [
                IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True),
                IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True),
            ]

            # エントリー条件
            entry_conditions = [
                Condition(left_operand="RSI", operator="<", right_operand=30)
            ]

            # 出口条件（オプション）
            exit_conditions = []
            if include_exit_conditions:
                exit_conditions = [
                    Condition(left_operand="RSI", operator=">", right_operand=70)
                ]

            # リスク管理設定（strategy_gene_utils.py互換）
            risk_management = {
                "stop_loss": 0.03,
                "take_profit": 0.15,
                "position_size": 0.1,
            }

            # クラスが指定された場合は使用（後方互換性）
            gene_class = strategy_gene_class or StrategyGene

            return gene_class(
                indicators=indicators,
                entry_conditions=entry_conditions,
                exit_conditions=exit_conditions,
                risk_management=risk_management,
                metadata={"generated_by": "AutoStrategyUtils", "version": "2.0"},
            )
        except Exception as e:
            logger.error(f"デフォルト戦略遺伝子作成エラー: {e}")
            if strategy_gene_class:
                return strategy_gene_class()
            return None

    @staticmethod
    def get_all_indicator_ids() -> Dict[str, int]:
        """全指標のIDマッピングを取得（config.constants に統合）"""
        from ..config.constants import get_all_indicator_ids

        return get_all_indicator_ids()

    @staticmethod
    def get_encoding_info(indicator_ids: Optional[Dict[str, int]] = None) -> Dict:
        """エンコーディング情報を取得"""
        if indicator_ids is None:
            indicator_ids = AutoStrategyUtils.get_all_indicator_ids()

        return {
            "indicator_count": len(indicator_ids) - 1,
            "max_indicators": 5,
            "encoding_length": 32,  # 5指標×2 + 条件×6 + TP/SL×8 + ポジションサイジング×8
            "tpsl_encoding_length": 8,
            "position_sizing_encoding_length": 8,
            "supported_indicators": list(indicator_ids.keys())[1:],
        }

    # === パラメータ正規化ユーティリティ（gene_utils.pyから統合） ===

    @staticmethod
    def normalize_parameter(
        value: float, min_val: float = 1, max_val: float = 200
    ) -> float:
        """
        パラメータを0-1の範囲に正規化（scikit-learn MinMaxScaler互換）

        Args:
            value: 正規化する値
            min_val: 最小値
            max_val: 最大値

        Returns:
            正規化された値（0-1の範囲）
        """
        try:
            from sklearn.preprocessing import MinMaxScaler
            import numpy as np

            # scikit-learnのMinMaxScalerを使用
            scaler = MinMaxScaler(feature_range=(0, 1))

            # 単一値の場合の処理
            if min_val == max_val:
                return 0.0

            # 値を配列に変換してスケーリング
            values = np.array([[min_val], [max_val], [value]])
            scaled_values = scaler.fit_transform(values)

            return float(scaled_values[2, 0])

        except Exception as e:
            logger.error(f"正規化エラー: {e}")
            # フォールバック: 手動実装
            try:
                if max_val == min_val:
                    return 0.0
                return (value - min_val) / (max_val - min_val)
            except ZeroDivisionError:
                return 0.0

    @staticmethod
    def denormalize_parameter(
        normalized_val: float, min_val: float = 1, max_val: float = 200
    ) -> int:
        """
        正規化されたパラメータを元の範囲に戻す（scikit-learn MinMaxScaler互換）

        Args:
            normalized_val: 正規化された値（0-1の範囲）
            min_val: 元の最小値
            max_val: 元の最大値

        Returns:
            元の範囲に戻された整数値
        """
        try:
            from sklearn.preprocessing import MinMaxScaler
            import numpy as np

            # scikit-learnのMinMaxScalerを使用
            scaler = MinMaxScaler(feature_range=(0, 1))

            # 単一値の場合の処理
            if min_val == max_val:
                return int(min_val)

            # 範囲を学習
            range_values = np.array([[min_val], [max_val]])
            scaler.fit(range_values)

            # 逆変換
            denormalized = scaler.inverse_transform([[normalized_val]])
            value = denormalized[0, 0]

            return int(max(min_val, min(max_val, int(value))))

        except Exception as e:
            logger.error(f"非正規化エラー: {e}")
            # フォールバック: 手動実装
            try:
                if max_val == min_val:
                    return int(min_val)
                value = min_val + normalized_val * (max_val - min_val)
                return int(max(min_val, min(max_val, int(value))))
            except Exception:
                return int(min_val)

    # === 検証ユーティリティ ===

    @staticmethod
    def validate_range(value: Any, min_val: float, max_val: float) -> bool:
        """値が指定範囲内かチェック"""
        try:
            num_value = float(value)
            return min_val <= num_value <= max_val
        except (ValueError, TypeError):
            return False

    @staticmethod
    def validate_required_fields(
        data: Dict, required_fields: List[str]
    ) -> tuple[bool, List[str]]:
        """必須フィールドの存在チェック"""
        errors = []
        for field in required_fields:
            if field not in data or data[field] is None:
                errors.append(f"必須フィールド '{field}' が不足しています")

        return len(errors) == 0, errors

    @staticmethod
    def validate_strategy_gene(gene) -> bool:
        """戦略遺伝子の妥当性を検証"""
        try:
            if not gene:
                return False

            # 基本的な構造チェック
            if not hasattr(gene, "indicators") or not gene.indicators:
                return False

            if not hasattr(gene, "entry_conditions") or not gene.entry_conditions:
                return False

            return True
        except Exception as e:
            logger.error(f"戦略遺伝子バリデーションエラー: {e}")
            return False

    # === パフォーマンス測定ユーティリティ ===

    @staticmethod
    def time_function(func, *args, **kwargs):
        """関数の実行時間を測定"""
        import time

        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"{func.__name__} 実行時間: {execution_time:.4f}秒")
            return result, execution_time
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} 実行エラー ({execution_time:.4f}秒): {e}")
            raise

    # === 設定管理ユーティリティ ===

    @staticmethod
    def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
        """設定辞書をマージ"""
        try:
            merged = base_config.copy()
            for key, value in override_config.items():
                if (
                    isinstance(value, dict)
                    and key in merged
                    and isinstance(merged[key], dict)
                ):
                    merged[key] = AutoStrategyUtils.merge_configs(merged[key], value)
                else:
                    merged[key] = value
            return merged
        except Exception as e:
            logger.error(f"設定マージエラー: {e}")
            return base_config

    @staticmethod
    def extract_config_subset(config: Dict, keys: List[str]) -> Dict:
        """設定から指定キーのサブセットを抽出"""
        return {key: config.get(key) for key in keys if key in config}

    # === ログ管理ユーティリティ ===

    @staticmethod
    def setup_auto_strategy_logger(name: str, level: str = "INFO") -> logging.Logger:
        """Auto Strategy専用ロガーを設定"""
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))

        # ハンドラーが既に設定されている場合はスキップ
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger


# 後方互換性のためのエイリアス
ensure_float = AutoStrategyUtils.safe_convert_to_float
ensure_int = AutoStrategyUtils.safe_convert_to_int
ensure_list = AutoStrategyUtils.ensure_list
ensure_dict = AutoStrategyUtils.ensure_dict
normalize_symbol = AutoStrategyUtils.normalize_symbol
validate_range = AutoStrategyUtils.validate_range
validate_required_fields = AutoStrategyUtils.validate_required_fields
time_function = AutoStrategyUtils.time_function
