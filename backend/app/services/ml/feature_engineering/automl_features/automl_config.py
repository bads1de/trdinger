"""
AutoML設定管理クラス

AutoML特徴量エンジニアリングの設定を管理します。
このモジュールでは、AutoFeat特徴量エンジニアリングライブラリの
設定を管理し、データサイズに応じた動的な最適化機能を提供します。
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class AutoFeatConfig:
    """
    AutoFeatライブラリの設定を管理するデータクラス

    AutoFeatは数学的変換と組み合わせにより自動特徴量エンジニアリングを行うライブラリです。
    内部でLassoLarsCV/LogisticRegressionCVを使用して特徴量選択を行います。
    このクラスでは、特徴量生成のパラメータとメモリ使用量制御を管理します。

    Attributes:
        enabled (bool): AutoFeatを有効にするかどうか
        feateng_steps (int): 特徴量エンジニアリングのステップ数（1-3推奨、デフォルト2）
        max_gb (float): 使用メモリの最大量（GB）
        featsel_runs (int): 特徴量選択の実行回数（デフォルト5、精度と速度のバランス）
        verbose (int): ログレベル（0=最小限、1=詳細）
        n_jobs (int): 並列処理数（1推奨、メモリ使用量制御のため）

    Note:
        - AutoFeatは内部で数百〜数千の候補特徴量を生成し、featsel_runsによる
          交差検証で最も有用な特徴量を自動選択します
        - feateng_steps=2が推奨（バランスが良い）
        - feateng_steps=3以上は指数関数的に特徴量が増加するため注意
        - AutoFeat v2.1.3には遺伝的アルゴリズムのパラメータは存在しません
    """

    enabled: bool = True  # デフォルトで有効化
    feateng_steps: int = 2  # 特徴量エンジニアリングのステップ数（推奨: 2）
    max_gb: float = 2.0  # メモリ使用量の上限（GB）
    featsel_runs: int = 5  # 特徴量選択の実行回数（推奨: 5、精度を確保）
    verbose: int = 0  # ログレベル（0=最小限、1=詳細）
    n_jobs: int = 1  # 並列処理数（メモリ使用量制御のため1に設定）

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
            データサイズに基づく推奨設定:
            - 500MB以上: feateng_steps=1, メモリ優先、最小限の特徴量選択
            - 100-500MB: feateng_steps=1-2, メモリ制約を考慮しつつ特徴量生成
            - 10-100MB: feateng_steps=2, バランスの取れた設定（推奨）
            - 10MB未満: feateng_steps=2, 十分なメモリでフル機能を活用

        Examples:
            >>> config = AutoFeatConfig()
            >>> # 50MBのデータに最適化
            >>> optimized = config.get_memory_optimized_config(50.0)
            >>> print(optimized.feateng_steps)  # 2
            >>> print(optimized.featsel_runs)   # 5
        """
        # データサイズに基づく動的設定
        if data_size_mb > 500:  # 500MB以上の大量データ
            return AutoFeatConfig(
                enabled=self.enabled,
                feateng_steps=1,  # メモリ優先、基本的な変換のみ
                max_gb=1.0,  # メモリ使用量を制限
                featsel_runs=1,  # 選択回数を最小限に
                verbose=0,
                n_jobs=1,
            )
        elif data_size_mb > 100:  # 100-500MBの中量データ
            return AutoFeatConfig(
                enabled=self.enabled,
                feateng_steps=2,  # 適度な特徴量生成
                max_gb=2.0,  # 適度なメモリ使用量
                featsel_runs=3,  # 選択精度を確保
                verbose=0,
                n_jobs=1,
            )
        elif data_size_mb > 10:  # 10-100MBの中小量データ（推奨設定）
            return AutoFeatConfig(
                enabled=self.enabled,
                feateng_steps=2,  # バランスの取れた特徴量生成
                max_gb=2.0,  # 十分なメモリ
                featsel_runs=5,  # デフォルトの選択精度
                verbose=0,
                n_jobs=1,
            )
        else:  # 10MB未満の小量データ
            return AutoFeatConfig(
                enabled=self.enabled,
                feateng_steps=2,  # フル機能を活用
                max_gb=1.0,  # 小規模データには十分
                featsel_runs=5,  # デフォルトの選択精度
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
            - feateng_steps=2: 価格、出来高、テクニカル指標の組み合わせに最適
            - featsel_runs=5: 十分な交差検証で堅牢な特徴量を選択
            - max_gb=2.0: 一般的な取引データサイズに対応
        """
        autofeat_config = AutoFeatConfig(
            enabled=True,  # デフォルト有効化
            feateng_steps=2,  # 金融データに最適なステップ数
            max_gb=2.0,  # 適度なメモリ使用量
            featsel_runs=5,  # 堅牢な特徴量選択
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
            "verbose",
            "n_jobs",
        }
        filtered_params = {
            k: v for k, v in autofeat_raw.items() if k in valid_params
        }

        autofeat_config = AutoFeatConfig(**filtered_params)

        return cls(autofeat_config)
