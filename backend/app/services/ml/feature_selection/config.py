"""
特徴量選択の設定モジュール

SelectionMethod列挙型とFeatureSelectionConfigデータクラスを定義します。
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class SelectionMethod(Enum):
    """特徴量選択手法"""

    # Filter手法（高速・モデル非依存）
    VARIANCE = "variance"
    UNIVARIATE_F = "univariate_f"
    UNIVARIATE_CHI2 = "univariate_chi2"
    MUTUAL_INFO = "mutual_info"

    # Wrapper手法（計算コスト高・精度重視）
    RFE = "rfe"
    RFECV = "rfecv"
    PERMUTATION = "permutation"

    # Embedded手法（モデル学習と同時）
    LASSO = "lasso"
    RANDOM_FOREST = "random_forest"

    # 組み合わせ手法（推奨）
    SHADOW = "shadow"  # Boruta風シャドウ特徴量ベース
    STAGED = "staged"  # 段階的フィルタリング

    @classmethod
    def default_staged_methods(cls) -> List["SelectionMethod"]:
        """Staged 選択のデフォルト段階を返す。"""
        return [
            cls.VARIANCE,
            cls.MUTUAL_INFO,
            cls.RFECV,
        ]


@dataclass
class FeatureSelectionConfig:
    """
    特徴量選択設定

    ベストプラクティスに基づくデフォルト値を提供。
    """

    method: SelectionMethod = SelectionMethod.STAGED

    # --- Filter設定 ---
    variance_threshold: float = 0.0  # 定数・準定数の削除
    correlation_threshold: float = 0.90  # 高相関ペアの削除

    # --- Wrapper/Embedded設定 ---
    target_k: Optional[int] = None  # None = 自動決定 (RFECV)
    min_features: int = 5  # 最小特徴量数
    max_features: Optional[int] = None  # 最大特徴量数 (追加)

    # --- 質による選別 ---
    cumulative_importance: float = 0.95  # 累積重要度閾値
    min_relative_importance: float = 0.01  # トップ比での足切り
    importance_threshold: float = 0.001  # 絶対閾値

    # --- クロスバリデーション ---
    cv_folds: int = 5
    cv_strategy: str = "stratified"  # "stratified" or "timeseries"

    # --- 並列処理 ---
    random_state: int = 42
    n_jobs: int = 1  # デフォルトは安全のため1

    # --- シャドウ特徴量設定 (Boruta風) ---
    shadow_iterations: int = 20
    shadow_percentile: float = 100.0  # シャドウ最大値のパーセンタイル

    # --- Staged選択の段階 ---
    staged_methods: List[SelectionMethod] = field(
        default_factory=SelectionMethod.default_staged_methods
    )
