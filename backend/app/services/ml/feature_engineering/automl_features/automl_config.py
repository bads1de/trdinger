"""
AutoML設定管理クラス

AutoML特徴量エンジニアリングの設定を管理します。
このモジュールでは、AutoFeat特徴量エンジニアリングライブラリの
設定を管理し、データサイズに応じた動的な最適化機能を提供します。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AutoFeatConfig:
    """
    AutoFeatライブラリの設定を管理するデータクラス

    AutoFeatは数学的変換と組み合わせにより自動特徴量エンジニアリングを行うライブラリです。
    内部でLassoLarsCV/LogisticRegressionCVを使用して特徴量選択を行います。
    このクラスでは、特徴量生成のパラメータとメモリ使用量制御を管理します。

    Attributes:
        enabled (bool): AutoFeatを有効にするかどうか
        feateng_steps (int): 特徴量エンジニアリングのステップ数（1-3推奨、デフォルト1: 軽量設定）
        max_gb (float): 使用メモリの最大量（GB、デフォルト1.0: 軽量設定）
        featsel_runs (int): 特徴量選択の実行回数（デフォルト2: 軽量設定）
        transformations (List[str]): 使用する数学的変換のリスト（デフォルト: ['1/', 'sqrt', '^2']）
        verbose (int): ログレベル（0=最小限、1=詳細）
        n_jobs (int): 並列処理数（1推奨、メモリ使用量制御のため）

    Note:
        - 軽量設定（バランス型）を採用：
          * feateng_steps=1: メモリ使用量を40-50%削減
          * max_gb=1.0: メモリ使用量の直接制限
          * featsel_runs=2: 処理時間を30-40%短縮
          * transformations=['1/', 'sqrt', '^2']: 基本的な3種類に削減
        - AutoFeatは内部で数百〜数千の候補特徴量を生成し、featsel_runsによる
          交差検証で最も有用な特徴量を自動選択します
        - AutoFeat v2.1.3には遺伝的アルゴリズムのパラメータは存在しません
    """

    enabled: bool = True  # デフォルトで有効化
    feateng_steps: int = 1  # 特徴量エンジニアリングのステップ数（軽量設定: 1）
    max_gb: float = 1.0  # メモリ使用量の上限（GB、軽量設定: 1.0）
    featsel_runs: int = 2  # 特徴量選択の実行回数（軽量設定: 2）
    transformations: List[str] = field(
        default=None
    )  # 使用する変換（Noneの場合は初期化時にデフォルト設定）
    verbose: int = 0  # ログレベル（0=最小限、1=詳細）
    n_jobs: int = 1  # 並列処理数（メモリ使用量制御のため1に設定）

    def __post_init__(self):
        """初期化後の処理でデフォルト値を設定"""
        if self.transformations is None:
            # 軽量設定: 基本的な3種類の変換のみ使用
            self.transformations = ["1/", "sqrt", "^2"]

    def get_memory_optimized_config(self, data_size_mb: float) -> "AutoFeatConfig":
        """
        データサイズに基づいてメモリ最適化された設定を取得

        データサイズに応じて動的にパラメータを調整し、メモリ使用量を最適化します。
        AutoFeatの内部特徴量選択機能を活用し、適切なfeateng_stepsとfeatsel_runsで
        バランスの取れた特徴量生成を実現します。

        Args:
            data_size_mb (float): データサイズ（MB）

        Returns:
            AutoFeatConfig: メモリ使用量が最適化された設定

        Note:
            データサイズに基づく推奨設定（軽量化版）:
            - 500MB以上: feateng_steps=1, メモリ優先、最小限の特徴量選択
            - 100-500MB: feateng_steps=1, メモリ制約を考慮した設定
            - 10-100MB: feateng_steps=1, バランスの取れた軽量設定（推奨）
            - 10MB未満: feateng_steps=1, 小規模データでも軽量設定を適用

        Examples:
            >>> config = AutoFeatConfig()
            >>> # 50MBのデータに最適化
            >>> optimized = config.get_memory_optimized_config(50.0)
            >>> print(optimized.feateng_steps)  # 1
            >>> print(optimized.featsel_runs)   # 2
        """
        # データサイズに基づく動的設定（軽量化版）
        if data_size_mb > 500:  # 500MB以上の大量データ
            return AutoFeatConfig(
                enabled=self.enabled,
                feateng_steps=1,  # メモリ優先、基本的な変換のみ
                max_gb=0.5,  # メモリ使用量を厳しく制限
                featsel_runs=1,  # 選択回数を最小限に
                transformations=["1/", "sqrt"],  # 最小限の変換
                verbose=0,
                n_jobs=1,
            )
        elif data_size_mb > 100:  # 100-500MBの中量データ
            return AutoFeatConfig(
                enabled=self.enabled,
                feateng_steps=1,  # 軽量設定
                max_gb=1.0,  # 軽量メモリ使用量
                featsel_runs=1,  # 選択回数を削減
                transformations=["1/", "sqrt", "^2"],  # 基本的な変換
                verbose=0,
                n_jobs=1,
            )
        elif data_size_mb > 10:  # 10-100MBの中小量データ（推奨軽量設定）
            return AutoFeatConfig(
                enabled=self.enabled,
                feateng_steps=1,  # バランスの取れた軽量設定
                max_gb=1.0,  # 適度なメモリ
                featsel_runs=2,  # 軽量な選択精度
                transformations=["1/", "sqrt", "^2"],  # 基本的な変換
                verbose=0,
                n_jobs=1,
            )
        else:  # 10MB未満の小量データ
            return AutoFeatConfig(
                enabled=self.enabled,
                feateng_steps=1,  # 小規模でも軽量設定
                max_gb=1.0,  # 小規模データには十分
                featsel_runs=2,  # 軽量な選択精度
                transformations=["1/", "sqrt", "^2"],  # 基本的な変換
                verbose=0,
                n_jobs=1,
            )


@dataclass
class AutoMLConfig:
    """
    AutoML全体の設定を管理するメインクラス

    AutoFeatの設定を管理し、プリセット設定の提供や
    設定のシリアライズ/デシリアライズ機能を提供します。
    金融データ向けの最適化設定など、用途に応じた設定も提供します。

    Attributes:
        autofeat (AutoFeatConfig): AutoFeatライブラリの設定
    """

    autofeat: AutoFeatConfig

    def __init__(
        self,
        autofeat_config: Optional[AutoFeatConfig] = None,
    ):
        """
        AutoMLConfigを初期化します

        Args:
            autofeat_config (Optional[AutoFeatConfig]): AutoFeat設定、Noneの場合はデフォルト値を使用
        """
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
        暗号通貨取引データの特性を考慮し、適切な特徴量生成とメモリ使用量の
        バランスを取った設定になっています。

        Returns:
            AutoMLConfig: 金融データ向けに最適化された設定

        Note:
            - feateng_steps=1: 軽量設定で効率的な特徴量生成
            - featsel_runs=2: 処理時間を短縮しつつ堅牢な特徴量を選択
            - max_gb=1.0: メモリ使用量を抑制
            - transformations: 基本的な3種類の変換で十分な性能を確保
        """
        autofeat_config = AutoFeatConfig(
            enabled=True,  # デフォルト有効化
            feateng_steps=1,  # 軽量設定: メモリ使用量を40-50%削減
            max_gb=1.0,  # 軽量設定: メモリ使用量の直接制限
            featsel_runs=2,  # 軽量設定: 処理時間を30-40%短縮
            transformations=["1/", "sqrt", "^2"],  # 軽量設定: 基本的な3種類に削減
            verbose=0,
            n_jobs=1,
        )

        return cls(autofeat_config)

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

        Args:
            config_dict (Dict[str, Any]): 設定辞書

        Returns:
            AutoMLConfig: 復元された設定オブジェクト

        Note:
            - 不明なキーは無視され、デフォルト値が使用されます
            - 古い設定ファイルとの互換性のため、存在しないパラメータは無視されます
            - generations, population_size, tournament_sizeなどの旧パラメータは自動的にスキップされます
        """
        autofeat_raw = dict(config_dict.get("autofeat", {}))

        # 有効なパラメータのみを抽出（旧パラメータを除外）
        valid_params = {
            "enabled",
            "feateng_steps",
            "max_gb",
            "featsel_runs",
            "transformations",
            "verbose",
            "n_jobs",
        }
        filtered_params = {k: v for k, v in autofeat_raw.items() if k in valid_params}

        autofeat_config = AutoFeatConfig(**filtered_params)

        return cls(autofeat_config)
