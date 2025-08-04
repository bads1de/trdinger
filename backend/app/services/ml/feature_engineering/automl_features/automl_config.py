"""
AutoML設定管理クラス

AutoML特徴量エンジニアリングの設定を管理します。
このモジュールでは、TSFreshとAutoFeatの2つの特徴量エンジニアリングライブラリの
設定を管理し、データサイズに応じた動的な最適化機能を提供します。
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class TSFreshConfig:
    """
    TSFreshライブラリの設定を管理するデータクラス

    TSFreshは時系列データから自動的に特徴量を抽出するライブラリです。
    このクラスでは、特徴量選択、パフォーマンスモード、並列処理などの
    設定を管理します。

    Attributes:
        enabled (bool): TSFreshを有効にするかどうか
        feature_selection (bool): 特徴量選択を実行するかどうか
        fdr_level (float): False Discovery Rateの閾値（0.0-1.0）
        feature_count_limit (int): 生成する特徴量の最大数
        parallel_jobs (int): 並列処理ジョブ数
        performance_mode (str): パフォーマンスモード（"balanced", "fast", "financial_optimized"）
        custom_settings (Optional[Dict[str, Any]]): カスタム設定辞書
    """

    enabled: bool = True
    feature_selection: bool = True
    fdr_level: float = 0.05
    feature_count_limit: int = 100
    parallel_jobs: int = 2
    performance_mode: str = "balanced"
    custom_settings: Optional[Dict[str, Any]] = None


@dataclass
class AutoFeatConfig:
    """
    AutoFeatライブラリの設定を管理するデータクラス

    AutoFeatは遺伝的アルゴリズムを使用して自動特徴量エンジニアリングを行うライブラリです。
    このクラスでは、特徴量生成のパラメータ、メモリ使用量制御、
    遺伝的アルゴリズムの設定などを管理します。

    Attributes:
        enabled (bool): AutoFeatを有効にするかどうか
        max_features (int): 生成する特徴量の最大数
        feateng_steps (int): 特徴量エンジニアリングのステップ数
        max_gb (float): 使用メモリの最大量（GB）
        featsel_runs (int): 特徴量選択の実行回数（メモリ節約のため1に設定）
        verbose (int): ログレベル（0=最小限、1=詳細）
        n_jobs (int): 並列処理数（メモリ使用量制御のため1に設定）
        generations (int): 遺伝的アルゴリズムの世代数
        population_size (int): 遺伝的アルゴリズムの集団サイズ
        tournament_size (int): 遺伝的アルゴリズムのトーナメントサイズ
    """

    enabled: bool = True
    max_features: int = 100
    feateng_steps: int = 2
    max_gb: float = 1.0
    featsel_runs: int = 1  # 特徴量選択の実行回数（メモリ節約のため1に設定）
    verbose: int = 0  # ログレベル（0=最小限、1=詳細）
    n_jobs: int = 1  # 並列処理数（メモリ使用量制御のため1に設定）
    generations: int = 20  # 世代数
    population_size: int = 50  # 集団サイズ
    tournament_size: int = 3  # トーナメントサイズ

    def get_memory_optimized_config(self, data_size_mb: float) -> "AutoFeatConfig":
        """
        データサイズに基づいてメモリ最適化された設定を取得

        データサイズに応じて動的にパラメータを調整し、メモリ使用量を最適化します。
        大量データの場合は特徴量数や計算リソースを制限し、小量データの場合は
        より多くの特徴量生成を許可します。

        Args:
            data_size_mb (float): データサイズ（MB）

        Returns:
            AutoFeatConfig: メモリ使用量が最適化された設定

        Note:
            データサイズに基づく動的設定（メモリ使用量を大幅に制限）:
            - 500MB以上: 最小限の特徴量と計算リソース
            - 100MB以上: 中程度の制限
            - 10MB以上: 緩やかな制限
            - 1MB以上: 基本的な制限
            - 1MB未満: 最小限の設定
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
                generations=5,  # 世代数を制限
                population_size=20,  # 集団サイズを制限
                tournament_size=2,  # トーナメントサイズを最小に
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
                generations=10,
                population_size=30,
                tournament_size=3,
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
                generations=15,
                population_size=40,
                tournament_size=3,
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
                generations=10,
                population_size=30,
                tournament_size=3,
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
                generations=5,
                population_size=20,
                tournament_size=2,
            )


@dataclass
class AutoMLConfig:
    """
    AutoML全体の設定を管理するメインクラス

    TSFreshとAutoFeatの設定を統合管理し、プリセット設定の提供や
    設定のシリアライズ/デシリアライズ機能を提供します。
    金融データ向けの最適化設定など、用途に応じた設定も提供します。

    Attributes:
        tsfresh (TSFreshConfig): TSFreshライブラリの設定
        autofeat (AutoFeatConfig): AutoFeatライブラリの設定
    """

    tsfresh: TSFreshConfig
    autofeat: AutoFeatConfig

    def __init__(
        self,
        tsfresh_config: Optional[TSFreshConfig] = None,
        autofeat_config: Optional[AutoFeatConfig] = None,
    ):
        """
        AutoMLConfigを初期化します

        Args:
            tsfresh_config (Optional[TSFreshConfig]): TSFresh設定、Noneの場合はデフォルト値を使用
            autofeat_config (Optional[AutoFeatConfig]): AutoFeat設定、Noneの場合はデフォルト値を使用
        """
        self.tsfresh = tsfresh_config or TSFreshConfig()
        self.autofeat = autofeat_config or AutoFeatConfig()

    @classmethod
    def get_default_config(cls) -> "AutoMLConfig":
        """
        デフォルト設定を取得

        Returns:
            AutoMLConfig: デフォルト値で初期化された設定
        """
        return cls()

    @classmethod
    def get_financial_optimized_config(cls) -> "AutoMLConfig":
        """
        金融データ最適化設定を取得

        金融時系列データに特化した最適化設定を提供します。
        より厳しい特徴量選択と、金融データ向けの特徴量生成パラメータを設定します。

        Returns:
            AutoMLConfig: 金融データ向けに最適化された設定

        Note:
            Featuretoolsは削除済み（後方互換プロパティは保持しない）
        """
        tsfresh_config = TSFreshConfig(
            enabled=True,
            feature_selection=True,
            fdr_level=0.01,  # より厳しい選択
            feature_count_limit=500,  # 金融データ用に大幅増加
            parallel_jobs=4,
            performance_mode="financial_optimized",
        )

        autofeat_config = AutoFeatConfig(
            enabled=True,
            max_features=100,
            feateng_steps=3,  # より多くの特徴量エンジニアリングステップ
            max_gb=2.0,  # より多くのメモリ使用を許可
            generations=20,
            population_size=50,
            tournament_size=3,
        )

        return cls(tsfresh_config, autofeat_config)

    def to_dict(self) -> Dict[str, Any]:
        """
        設定を辞書形式に変換（dataclass自動シリアライゼーション使用）

        手動マッピングを削除し、dataclassの asdict() を活用して
        保守性を向上させました。

        Returns:
            Dict[str, Any]: シリアライズされた設定辞書
        """
        from dataclasses import asdict

        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "AutoMLConfig":
        """
        辞書から設定を作成

        シリアライズされた設定辞書からAutoMLConfigオブジェクトを復元します。
        旧キー名の後方互換（例: 'generations' -> 'feateng_steps'）にも対応します。

        Args:
            config_dict (Dict[str, Any]): 設定辞書

        Returns:
            AutoMLConfig: 復元された設定オブジェクト

        Note:
            - 不明なキーは無視され、デフォルト値が使用されます
            - 'autofeat.generations' が存在し 'feateng_steps' が無い場合は、feateng_steps に転記します
        """
        tsfresh_raw = dict(config_dict.get("tsfresh", {}))
        autofeat_raw = dict(config_dict.get("autofeat", {}))

        # 後方互換: generations -> feateng_steps
        if "feateng_steps" not in autofeat_raw and "generations" in autofeat_raw:
            autofeat_raw["feateng_steps"] = autofeat_raw.get("generations")

        tsfresh_config = TSFreshConfig(**tsfresh_raw)
        autofeat_config = AutoFeatConfig(**autofeat_raw)

        return cls(tsfresh_config, autofeat_config)
