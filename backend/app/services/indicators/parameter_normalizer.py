"""
パラメータ正規化モジュール

インジケーターパラメータの正規化を担当します。
"""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class ParameterNormalizer:
    """
    パラメータ正規化クラス

    入力パラメータをpandas-taが期待する形式に正規化します。
    エイリアスの解決、デフォルト値の補完、最小値ガードを適用します。
    ユーザーが指定したパラメータ名（例: 'length', 'period'）を
    pandas-taが期待する標準パラメータ名に変換します。
    """

    def normalize_params(
        self, params: Dict[str, Any], config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        入力パラメータをpandas-taが期待する形式に正規化

        ユーザー指定のパラメータをエイリアス解決し、デフォルト値を補完し、
        最小値ガードを適用して正規化します。

        Args:
            params: ユーザー指定のパラメータ辞書（例: {'length': 20}）
            config: pandas-ta設定（params、default_values、min_length等を含む）

        Returns:
            Dict[str, Any]: 正規化されたパラメータ辞書

        処理手順:
            1. config['params']に定義された各パラメータについて:
               - エイリアスリストからユーザー指定の値を検索
               - 値が見つからない場合はデフォルト値を使用
               - min_lengthガードを適用（length/periodパラメータ）
        """
        normalized = {}
        for param_name, aliases in config["params"].items():
            value = None
            for alias in aliases:
                if alias in params:
                    value = params[alias]
                    break

            if value is None:
                value = config["default_values"].get(param_name)

            if value is not None:
                # min_lengthガードの適用
                value = self._apply_min_length_guard(
                    param_name, value, config
                )

                normalized[param_name] = value

        return normalized

    def _apply_min_length_guard(
        self, param_name: str, value: Any, config: Dict[str, Any]
    ) -> Any:
        """
        min_lengthガードを適用

        lengthやperiodなどのパラメータに対して、
        設定された最小値未満の場合は最小値に調整します。

        Args:
            param_name: パラメータ名（例: 'length', 'period'）
            value: パラメータ値
            config: pandas-ta設定（min_lengthを含む）

        Returns:
            Any: ガード適用後のパラメータ値

        ガード条件:
            - param_nameが'length'または'period'
            - configに'min_length'が定義されている
            - valueがmin_length未満

        Note:
            min_lengthは関数または固定値として定義できます。
        """
        if param_name in ["length", "period"] and "min_length" in config:
            min_length_func = config["min_length"]
            if callable(min_length_func):
                min_length = min_length_func({param_name: value})
                if (
                    isinstance(value, (int, float))
                    and isinstance(min_length, (int, float))
                    and value < min_length
                ):
                    logger.debug(
                        f"パラメータ {param_name}={value} が最小値 {min_length} 未満のため調整"
                    )
                    value = min_length
            elif (
                isinstance(min_length_func, (int, float))
                and isinstance(value, (int, float))
                and value < min_length_func
            ):
                value = min_length_func

        return value
