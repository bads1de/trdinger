"""
テクニカル指標統合サービス

pandas-taと独自実装のテクニカル指標を統一的に管理し、
効率的な計算とキャッシュを提供します。

主な特徴:
- 動的指標検出による自動設定
- pandas-ta直接呼び出しによる高効率
- 独自実装へのフォールバック（アダプター方式）
- 計算結果のLRUキャッシュ
"""

import logging
from typing import Any, Dict, Union

import numpy as np
import pandas as pd

from .adapter_handler import AdapterHandler
from .cache_manager import IndicatorCacheManager
from .config import IndicatorConfig, indicator_registry
from .indicator_validator import IndicatorValidator
from .pandas_ta_caller import PandasTaCaller
from .parameter_normalizer import ParameterNormalizer
from .post_processor import PostProcessor

logger = logging.getLogger(__name__)


class TechnicalIndicatorService:
    """テクニカル指標統合サービス

    pandas-ta と独自実装のテクニカル指標を統一的なインターフェースで提供し、
    オートストラテジー（GA）からの利用を最適化します。
    """

    _STANDARD_SUPPORT_DATA = {
        "open",
        "open_",
        "high",
        "low",
        "close",
        "volume",
    }
    _EXTENDED_MARKET_DATA = {
        "open_interest",
        "funding_rate",
        "openinterest",
        "fundingrate",
    }

    def __init__(self):
        """
        TechnicalIndicatorServiceを初期化

        テクニカル指標計算に必要なコンポーネントを初期化します。
        指標レジストリ、キャッシュマネージャー、パラメータ正規化器、
        バリデーター、pandas-ta呼び出し器、後処理器、アダプターハンドラー
        を設定します。

        Note:
            初期化時に指標レジストリが自動的に初期化され、
            利用可能な全指標の設定が読み込まれます。
        """
        self.registry = indicator_registry
        self.registry.ensure_initialized()

        # 分離されたコンポーネント
        self.cache_manager = IndicatorCacheManager()
        self.parameter_normalizer = ParameterNormalizer()
        self.validator = IndicatorValidator()
        self.pandas_ta_caller = PandasTaCaller(self.validator)
        self.post_processor = PostProcessor()
        self.adapter_handler = AdapterHandler(self.validator)

    def _get_indicator_config(self, indicator_type: str) -> IndicatorConfig:
        """
        指標設定を取得

        指標レジストリから指定された指標タイプの設定を取得します。

        Args:
            indicator_type: 指標タイプ（例: "RSI", "MACD"）

        Returns:
            IndicatorConfig: 指標設定オブジェクト

        Raises:
            ValueError: サポートされていない指標タイプが指定された場合
        """
        config = self.registry.get_indicator_config(indicator_type)
        if not config:
            raise ValueError(
                f"サポートされていない指標タイプ: {indicator_type}"
            )
        return config

    def _resolve_indicator_name(self, indicator_type: str) -> str:
        """
        指標名を正規化（大文字変換）

        指標タイプ名を大文字に変換して正規化します。
        これにより、大文字小文字を区別せずに指標を指定できます。

        Args:
            indicator_type: 正規化前の指標タイプ名

        Returns:
            str: 大文字に変換された指標タイプ名
        """
        return indicator_type.upper()

    def _resolve_column_name(
        self, df: pd.DataFrame, data_key: str
    ) -> str | None:
        """
        カラム名解決をバリデータに委譲する

        DataFrame内のカラム名を解決します。
        実際の解決ロジックはIndicatorValidatorに委譲されます。

        Args:
            df: 対象のDataFrame
            data_key: 解決するデータキー（カラム名またはエイリアス）

        Returns:
            str | None: 解決されたカラム名、見つからない場合はNone
        """
        return self.validator.resolve_column_name(df, data_key)

    def clear_cache(self) -> None:
        """
        計算キャッシュをクリアする

        指標計算結果のLRUキャッシュをクリアします。
        メモリ節約や再計算の強制が必要な場合に使用します。

        Note:
            キャッシュをクリアすると、次回の指標計算時には
            再度計算が実行されます。
        """
        self.cache_manager.clear_cache()

    def calculate_indicator(
        self, df: pd.DataFrame, indicator_type: str, params: Dict[str, Any]
    ) -> Union[
        np.ndarray, pd.Series, pd.DataFrame, tuple, tuple[pd.Series, ...], None
    ]:
        """
        OHLCVデータから指定されたテクニカル指標を計算します。

        このメソッドは以下の手順で指標計算をオーケストレーションします：
        1. 指標名の正規化とキャッシュキーの生成（`IndicatorCacheManager`）。
        2. キャッシュヒット時は即座に結果を返却。
        3. `pandas-ta` での実装が存在するか確認し、あれば以下の手順を実行：
           - パラメータの型変換・正規化（`ParameterNormalizer`）。
           - データのバリデーション（最小期間、欠損値チェック等）。
           - `pandas-ta` 関数の呼び出し（`PandasTaCaller`）。
           - 出力の後処理（単一シリーズか複数シリーズかの調整）。
        4. `pandas-ta` に存在しない、または失敗した場合、アダプター方式の独自実装へフォールバック。
        5. 計算結果をキャッシュに保存して返却。

        Args:
            df (pd.DataFrame): 計算の基準となるOHLCVデータを含むDataFrame。
            indicator_type (str): 指標の種類（例: "RSI", "MACD", "SMA"）。大文字小文字は区別されません。
            params (Dict[str, Any]): 指標に渡すパラメータ（期間、ソース、シグナル期間等）。

        Returns:
            Union[np.ndarray, pd.Series, tuple]: 指標の計算結果。
                単一指標（RSI等）の場合は1次元配列、複数指標（MACD等）の場合はタプルを返します。
                計算不可の場合は入力と同じ長さのNaNを含む配列が返されます。

        Raises:
            ValueError: 指定された指標タイプがレジストリに存在せず、アダプターも見つからない場合。
            Exception: 指標計算エンジン内で予期しない致命的なエラーが発生した場合。

        Note:
            - 高速化: 同一データ・同一パラメータでの再計算はLRUキャッシュにより回避されます。
            - 堅牢性: データの不足や異常値に対してはNaNを返却し、戦略の実行を継続させます。
        """
        indicator_type = self._resolve_indicator_name(indicator_type)

        # キャッシュチェック
        cache_key = self.cache_manager.make_cache_key(
            indicator_type, params, df
        )
        cached_result = self.cache_manager.get_cached_result(cache_key)
        if cached_result is not None:
            return cached_result

        try:
            # adapter_functionがある場合はアダプターパスを優先
            config_obj = self._get_indicator_config(indicator_type)
            pandas_config = None  # 初期化
            if config_obj and config_obj.adapter_function:
                # アダプター方式で処理
                result = self.adapter_handler.calculate_with_adapter(
                    df, indicator_type, params, config_obj
                )
            else:
                # pandas-ta設定を取得
                pandas_config = self.pandas_ta_caller.get_pandas_ta_config(
                    indicator_type, self.registry
                )
                result = None

                if pandas_config:
                    # pandas-ta方式で処理
                    normalized_params = (
                        self.parameter_normalizer.normalize_params(
                            params, pandas_config
                        )
                    )

                    if not self.validator.basic_validation(
                        df, pandas_config, normalized_params
                    ):
                        result = self.validator.create_nan_result(
                            df, pandas_config
                        )
                    else:
                        raw_result = self.pandas_ta_caller.call_pandas_ta(
                            df, pandas_config, normalized_params
                        )
                        if raw_result is not None:
                            result = self.post_processor.post_process(
                                raw_result, pandas_config, df
                            )

            # アダプター方式にフォールバック
            if result is None:
                try:
                    config_obj = self._get_indicator_config(indicator_type)
                    if config_obj and config_obj.adapter_function:
                        result = self.adapter_handler.calculate_with_adapter(
                            df, indicator_type, params, config_obj
                        )
                    else:
                        # アダプター関数がない場合はNaN結果を返す
                        if pandas_config:
                            result = self.validator.create_nan_result(
                                df, pandas_config
                            )
                            logger.warning(
                                f"指標 {indicator_type} のアダプター関数がありません。NaN結果を返します"
                            )
                        else:
                            raise ValueError(
                                f"指標 {indicator_type} の実装が見つかりません"
                            )
                except ValueError:
                    # pandas_configがない場合は、デフォルトのNaN結果を作成
                    if pandas_config:
                        result = self.validator.create_nan_result(
                            df, pandas_config
                        )
                    else:
                        # pandas_configがない場合は、デフォルト設定でNaN結果を作成
                        default_config = {
                            "function": indicator_type,
                            "returns": "single",
                            "return_cols": [indicator_type],
                        }
                        result = self.validator.create_nan_result(
                            df, default_config
                        )
                        logger.warning(
                            f"指標 {indicator_type} のpandas_configがありません。デフォルトNaN結果を返します"
                        )

            # キャッシュに保存
            if result is not None and cache_key:
                self.cache_manager.cache_result(cache_key, result)

            return result

        except Exception as e:
            logger.error(f"指標計算エラー {indicator_type}: {e}")
            # 例外が発生した場合はNaN結果を返す
            default_config = {
                "function": indicator_type,
                "returns": "single",
                "return_cols": [indicator_type],
            }
            result = self.validator.create_nan_result(df, default_config)
            if result is not None and cache_key:
                self.cache_manager.cache_result(cache_key, result)
            return result

    def _get_support_tier(self, config: IndicatorConfig) -> str:
        """
        required_dataからサポート水準を分類する

        指標が必要とするデータ種別に基づいて、サポート水準を分類します。
        - standard: OHLCVデータのみで計算可能
        - extended_market: OHLCV + オープンインタレスト/ファンディングレートで計算可能
        - experimental: その他のデータが必要（実験的）

        Args:
            config: 指標設定オブジェクト

        Returns:
            str: サポート水準（"standard", "extended_market", "experimental"のいずれか）
        """
        normalized_required_data = {
            data_key.lower()
            for data_key in config.required_data
            if isinstance(data_key, str) and data_key
        }

        if not normalized_required_data:
            return "standard"

        if normalized_required_data.issubset(self._STANDARD_SUPPORT_DATA):
            return "standard"

        if normalized_required_data.issubset(
            self._STANDARD_SUPPORT_DATA | self._EXTENDED_MARKET_DATA
        ):
            return "extended_market"

        return "experimental"

    def get_supported_indicators(self) -> Dict[str, Any]:
        """
        サポートされている指標の情報を取得

        利用可能な全テクニカル指標の情報を取得します。
        パラメータ範囲、結果タイプ、必要データ、スケールタイプ、
        サポート水準などの情報を含みます。

        Returns:
            Dict[str, Any]: サポート指標の情報辞書。
                キーは指標名、値は以下の情報を含む辞書：
                - parameters: パラメータ範囲
                - result_type: 結果タイプ
                - required_data: 必要データ
                - scale_type: スケールタイプ
                - support_tier: サポート水準

        Note:
            実装がない指標（adapter_functionもpandas_functionもない）は
            含まれません。
        """
        infos = {}
        for name, config in self.registry.get_all_indicators().items():
            if not config.adapter_function and not config.pandas_function:
                continue
            infos[name] = {
                "parameters": config.get_parameter_ranges(),
                "result_type": config.result_type.value,
                "required_data": config.required_data,
                "scale_type": (
                    config.scale_type.value if config.scale_type else None
                ),
                "support_tier": self._get_support_tier(config),
            }
        return infos
