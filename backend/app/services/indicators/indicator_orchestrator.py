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
from typing import Any, Dict, List, Union

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

    _STANDARD_SUPPORT_DATA = {"open", "open_", "high", "low", "close", "volume"}
    _EXTENDED_MARKET_DATA = {
        "open_interest",
        "funding_rate",
        "openinterest",
        "fundingrate",
    }

    def __init__(self):
        """サービスを初期化"""
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
        """指標設定を取得"""
        config = self.registry.get_indicator_config(indicator_type)
        if not config:
            raise ValueError(f"サポートされていない指標タイプ: {indicator_type}")
        return config

    def _resolve_indicator_name(self, indicator_type: str) -> str:
        """指標名を正規化（大文字変換）"""
        return indicator_type.upper()

    def _resolve_column_name(self, df: pd.DataFrame, data_key: Any) -> Any:
        """後方互換のためにバリデータへ委譲する。"""
        return self.validator.resolve_column_name(df, data_key)

    def clear_cache(self) -> None:
        """計算キャッシュをクリアする"""
        self.cache_manager.clear_cache()

    def calculate_indicator(
        self, df: pd.DataFrame, indicator_type: str, params: Dict[str, Any]
    ) -> Union[np.ndarray, pd.Series, tuple, tuple[pd.Series, ...]]:
        """
        OHLCVデータから指定されたテクニカル指標を計算

        Args:
            df: OHLCVデータを含むDataFrame
            indicator_type: 指標タイプ名（例: RSI, MACD, SMA）
            params: 指標のパラメータ

        Returns:
            計算結果（numpy配列またはタプル）
        """
        indicator_type = self._resolve_indicator_name(indicator_type)

        # キャッシュチェック
        cache_key = self.cache_manager.make_cache_key(indicator_type, params, df)
        cached_result = self.cache_manager.get_cached_result(cache_key)
        if cached_result is not None:
            return cached_result

        try:
            # pandas-ta設定を取得
            pandas_config = self.pandas_ta_caller.get_pandas_ta_config(indicator_type, self.registry)
            result = None

            if pandas_config:
                # pandas-ta方式で処理
                normalized_params = self.parameter_normalizer.normalize_params(params, pandas_config)

                if not self.validator.basic_validation(df, pandas_config, normalized_params):
                    result = self.validator.create_nan_result(df, pandas_config)
                else:
                    raw_result = self.pandas_ta_caller.call_pandas_ta(
                        df, pandas_config, normalized_params
                    )
                    if raw_result is not None:
                        result = self.post_processor.post_process(raw_result, pandas_config, df)

            # アダプター方式にフォールバック
            if result is None:
                try:
                    config_obj = self._get_indicator_config(indicator_type)
                    if config_obj.adapter_function:
                        result = self.adapter_handler.calculate_with_adapter(
                            df, indicator_type, params, config_obj
                        )
                    else:
                        raise ValueError(
                            f"指標 {indicator_type} の実装が見つかりません"
                        )
                except ValueError:
                    if pandas_config:
                        result = self.validator.create_nan_result(df, pandas_config)
                    else:
                        raise ValueError(
                            f"指標 {indicator_type} の実装が見つかりません"
                        )

            # キャッシュに保存
            if result is not None and cache_key:
                self.cache_manager.cache_result(cache_key, result)

            return result

        except Exception as e:
            logger.error(f"指標計算エラー {indicator_type}: {e}")
            raise



    def _get_support_tier(self, config: IndicatorConfig) -> str:
        """required_data からサポート水準を分類する。"""
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

        Returns:
            サポート指標の情報辞書
        """
        infos = {}
        for name, config in self.registry.get_all_indicators().items():
            if not config.adapter_function and not config.pandas_function:
                continue
            infos[name] = {
                "parameters": config.get_parameter_ranges(),
                "result_type": config.result_type.value,
                "required_data": config.required_data,
                "scale_type": config.scale_type.value if config.scale_type else None,
                "support_tier": self._get_support_tier(config),
            }
        return infos
