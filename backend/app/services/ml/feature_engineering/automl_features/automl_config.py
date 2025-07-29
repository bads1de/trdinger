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
    feature_count_limit: int = 100
    parallel_jobs: int = 2
    custom_settings: Optional[Dict[str, Any]] = None


# FeaturetoolsConfigは削除されました（メモリ最適化のため）


@dataclass
class AutoFeatConfig:
    """AutoFeat設定クラス"""

    enabled: bool = True
    max_features: int = 100
    feateng_steps: int = 2
    max_gb: float = 1.0
    featsel_runs: int = 1  # 特徴量選択の実行回数（メモリ節約のため1に設定）
    verbose: int = 0  # ログレベル（0=最小限、1=詳細）
    n_jobs: int = 1  # 並列処理数（メモリ使用量制御のため1に設定）

    def get_memory_optimized_config(self, data_size_mb: float) -> "AutoFeatConfig":
        """
        データサイズに基づいてメモリ最適化された設定を取得

        Args:
            data_size_mb: データサイズ（MB）

        Returns:
            最適化されたAutoFeatConfig
        """
        # データサイズに基づく動的設定（メモリ使用量を大幅に制限）
        if data_size_mb > 500:  # 500MB以上の大量データ
            return AutoFeatConfig(
                enabled=self.enabled,
                max_features=10,  # 特徴量数を大幅制限
                feateng_steps=1,  # ステップ数を最小限に
                max_gb=0.2,  # メモリ使用量を厳しく制限
                featsel_runs=1,
                verbose=0,
                n_jobs=1,
            )
        elif data_size_mb > 100:  # 100MB以上の中量データ
            return AutoFeatConfig(
                enabled=self.enabled,
                max_features=15,  # 特徴量数を制限
                feateng_steps=1,  # ステップ数を最小限に
                max_gb=0.3,  # メモリ使用量を制限
                featsel_runs=1,
                verbose=0,
                n_jobs=1,
            )
        elif data_size_mb > 10:  # 10MB以上の中小量データ
            return AutoFeatConfig(
                enabled=self.enabled,
                max_features=20,
                feateng_steps=1,
                max_gb=0.4,
                featsel_runs=1,
                verbose=0,
                n_jobs=1,
            )
        elif data_size_mb > 1:  # 1MB以上の小量データ
            return AutoFeatConfig(
                enabled=self.enabled,
                max_features=10,
                feateng_steps=1,
                max_gb=0.2,
                featsel_runs=1,
                verbose=0,
                n_jobs=1,
            )
        else:  # 極小量データ（1MB未満）
            return AutoFeatConfig(
                enabled=self.enabled,
                max_features=3,  # 極小量データでは最小限の特徴量
                feateng_steps=1,  # ステップ数を最小限に
                max_gb=0.05,  # 最小限のメモリ使用量（50MB）
                featsel_runs=1,
                verbose=0,
                n_jobs=1,
            )


@dataclass
class AutoMLConfig:
    """AutoML全体設定クラス（Featuretools削除版）"""

    tsfresh: TSFreshConfig
    autofeat: AutoFeatConfig

    def __init__(
        self,
        tsfresh_config: Optional[TSFreshConfig] = None,
        autofeat_config: Optional[AutoFeatConfig] = None,
    ):
        self.tsfresh = tsfresh_config or TSFreshConfig()
        self.autofeat = autofeat_config or AutoFeatConfig()

        # Featuretoolsは削除されました（メモリ最適化のため）
        # 後方互換性のためのダミー属性
        self.featuretools = type('FeaturetoolsConfig', (), {'enabled': False})()

    @classmethod
    def get_default_config(cls) -> "AutoMLConfig":
        """デフォルト設定を取得"""
        return cls()

    @classmethod
    def get_financial_optimized_config(cls) -> "AutoMLConfig":
        """金融データ最適化設定を取得（Featuretools削除版）"""
        tsfresh_config = TSFreshConfig(
            enabled=True,
            feature_selection=True,
            fdr_level=0.01,  # より厳しい選択
            feature_count_limit=500,  # 金融データ用に大幅増加
            parallel_jobs=4,
        )

        autofeat_config = AutoFeatConfig(
            enabled=True,
            max_features=100,
            feateng_steps=3,  # より多くの特徴量エンジニアリングステップ
            max_gb=2.0,  # より多くのメモリ使用を許可
        )

        return cls(tsfresh_config, autofeat_config)

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
                "enabled": False,  # Featuretoolsは削除されました
                "max_depth": 0,
                "max_features": 0,
                "agg_primitives": None,
                "trans_primitives": None,
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
        autofeat_config = AutoFeatConfig(**config_dict.get("autofeat", {}))

        return cls(tsfresh_config, autofeat_config)
