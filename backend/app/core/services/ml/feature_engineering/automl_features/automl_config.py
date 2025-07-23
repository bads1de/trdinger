"""
AutoML設定管理クラス

AutoML特徴量エンジニアリングの設定を管理します。
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class TSFreshConfig:
    """TSFresh設定クラス"""

    enabled: bool = True
    feature_selection: bool = True
    fdr_level: float = 0.05
    feature_count_limit: int = 500  # 生成する特徴量の最大数（大幅増加）
    parallel_jobs: int = 4
    custom_settings: Optional[Dict[str, Any]] = None


@dataclass
class FeaturetoolsConfig:
    """Featuretools設定クラス"""

    enabled: bool = True
    max_depth: int = 2
    max_features: int = 200  # 特徴量数を大幅増加
    agg_primitives: Optional[list] = None
    trans_primitives: Optional[list] = None


@dataclass
class AutoFeatConfig:
    """AutoFeat設定クラス"""

    enabled: bool = True
    max_features: int = 100
    feateng_steps: int = 2  # 特徴量エンジニアリングのステップ数
    max_gb: float = 1.0  # 最大メモリ使用量（GB）


@dataclass
class AutoMLConfig:
    """AutoML全体設定クラス"""

    tsfresh: TSFreshConfig
    featuretools: FeaturetoolsConfig
    autofeat: AutoFeatConfig

    def __init__(
        self,
        tsfresh_config: Optional[TSFreshConfig] = None,
        featuretools_config: Optional[FeaturetoolsConfig] = None,
        autofeat_config: Optional[AutoFeatConfig] = None,
    ):
        self.tsfresh = tsfresh_config or TSFreshConfig()
        self.featuretools = featuretools_config or FeaturetoolsConfig()
        self.autofeat = autofeat_config or AutoFeatConfig()

    @classmethod
    def get_default_config(cls) -> "AutoMLConfig":
        """デフォルト設定を取得"""
        return cls()

    @classmethod
    def get_financial_optimized_config(cls) -> "AutoMLConfig":
        """金融データ最適化設定を取得"""
        tsfresh_config = TSFreshConfig(
            enabled=True,
            feature_selection=True,
            fdr_level=0.01,  # より厳しい選択
            feature_count_limit=500,  # 金融データ用に大幅増加
            parallel_jobs=4,
        )

        featuretools_config = FeaturetoolsConfig(
            enabled=True, max_depth=3, max_features=200  # 金融データ用に大幅増加
        )

        autofeat_config = AutoFeatConfig(
            enabled=True,
            max_features=100,
            feateng_steps=3,  # より多くの特徴量エンジニアリングステップ
            max_gb=2.0,  # より多くのメモリ使用を許可
        )

        return cls(tsfresh_config, featuretools_config, autofeat_config)

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "tsfresh": {
                "enabled": self.tsfresh.enabled,
                "feature_selection": self.tsfresh.feature_selection,
                "fdr_level": self.tsfresh.fdr_level,
                "feature_count_limit": self.tsfresh.feature_count_limit,
                "parallel_jobs": self.tsfresh.parallel_jobs,
                "custom_settings": self.tsfresh.custom_settings,
            },
            "featuretools": {
                "enabled": self.featuretools.enabled,
                "max_depth": self.featuretools.max_depth,
                "max_features": self.featuretools.max_features,
                "agg_primitives": self.featuretools.agg_primitives,
                "trans_primitives": self.featuretools.trans_primitives,
            },
            "autofeat": {
                "enabled": self.autofeat.enabled,
                "max_features": self.autofeat.max_features,
                "feateng_steps": self.autofeat.feateng_steps,
                "max_gb": self.autofeat.max_gb,
            },
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "AutoMLConfig":
        """辞書から設定を作成"""
        tsfresh_config = TSFreshConfig(**config_dict.get("tsfresh", {}))
        featuretools_config = FeaturetoolsConfig(**config_dict.get("featuretools", {}))
        autofeat_config = AutoFeatConfig(**config_dict.get("autofeat", {}))

        return cls(tsfresh_config, featuretools_config, autofeat_config)
