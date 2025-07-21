"""
ML最適化プリセット定義

ユーザーフレンドリーなUXのために、ハイ・ミドル・ローの3段階の
最適化プリセットを提供します。
"""

from typing import Dict, Any
from dataclasses import dataclass
from enum import Enum


class OptimizationLevel(str, Enum):
    """最適化レベル"""

    HIGH = "high"  # 高精度（時間がかかるが精度重視）
    MEDIUM = "medium"  # 標準（バランス重視）
    LOW = "low"  # 高速（速度重視）


class OptimizationMethod(str, Enum):
    """最適化手法"""

    BAYESIAN = "bayesian"  # ベイジアン最適化
    GRID = "grid"  # グリッドサーチ
    RANDOM = "random"  # ランダムサーチ


@dataclass
class OptimizationPreset:
    """最適化プリセット設定"""

    name: str
    description: str
    method: str
    n_calls: int
    parameter_space: Dict[str, Any]
    estimated_time_minutes: int


class OptimizationPresets:
    """最適化プリセット管理クラス"""

    @staticmethod
    def get_preset(
        method: OptimizationMethod, level: OptimizationLevel
    ) -> OptimizationPreset:
        """指定された手法とレベルのプリセットを取得"""
        preset_key = f"{method.value}_{level.value}"
        presets = OptimizationPresets._get_all_preset_combinations()
        return presets[preset_key]

    @staticmethod
    def get_all_presets() -> Dict[str, OptimizationPreset]:
        """全てのプリセットを取得（手法×レベルの組み合わせ）"""
        return OptimizationPresets._get_all_preset_combinations()

    @staticmethod
    def get_presets_by_method(
        method: OptimizationMethod,
    ) -> Dict[str, OptimizationPreset]:
        """指定された手法のプリセット一覧を取得"""
        all_presets = OptimizationPresets._get_all_preset_combinations()
        return {
            key: preset
            for key, preset in all_presets.items()
            if key.startswith(method.value)
        }

    @staticmethod
    def _get_all_preset_combinations() -> Dict[str, OptimizationPreset]:
        """全ての手法×レベル組み合わせのプリセットを生成"""
        presets = {}

        for method in OptimizationMethod:
            for level in OptimizationLevel:
                preset_key = f"{method.value}_{level.value}"
                presets[preset_key] = OptimizationPresets._create_preset(method, level)

        return presets

    @staticmethod
    def _create_preset(
        method: OptimizationMethod, level: OptimizationLevel
    ) -> OptimizationPreset:
        """指定された手法とレベルのプリセットを作成"""
        # 基本パラメータ空間
        base_params = OptimizationPresets._get_base_parameter_space(level)

        # 手法別の設定
        method_config = OptimizationPresets._get_method_config(method, level)

        # 名前と説明を生成
        method_names = {
            OptimizationMethod.BAYESIAN: "ベイジアン",
            OptimizationMethod.GRID: "グリッド",
            OptimizationMethod.RANDOM: "ランダム",
        }

        level_names = {
            OptimizationLevel.HIGH: "ハイ（高精度）",
            OptimizationLevel.MEDIUM: "ミドル（標準）",
            OptimizationLevel.LOW: "ロー（高速）",
        }

        name = f"{method_names[method]} × {level_names[level]}"
        description = f"{method_names[method]}最適化で{level_names[level]}設定"

        return OptimizationPreset(
            name=name,
            description=description,
            method=method.value,
            n_calls=method_config["n_calls"],
            parameter_space=base_params,
            estimated_time_minutes=method_config["estimated_time"],
        )

    @staticmethod
    def _get_base_parameter_space(level: OptimizationLevel) -> Dict[str, Any]:
        """レベルに応じた基本パラメータ空間を取得"""
        if level == OptimizationLevel.HIGH:
            return OptimizationPresets._get_high_parameter_space()
        elif level == OptimizationLevel.MEDIUM:
            return OptimizationPresets._get_medium_parameter_space()
        else:  # LOW
            return OptimizationPresets._get_low_parameter_space()

    @staticmethod
    def _get_method_config(
        method: OptimizationMethod, level: OptimizationLevel
    ) -> Dict[str, Any]:
        """手法とレベルに応じた設定を取得"""
        # 基本試行回数（レベル別）
        base_calls = {
            OptimizationLevel.HIGH: 100,
            OptimizationLevel.MEDIUM: 50,
            OptimizationLevel.LOW: 20,
        }

        # 手法別の調整係数
        method_multipliers = {
            OptimizationMethod.BAYESIAN: 1.0,  # 効率的なので基本値
            OptimizationMethod.GRID: 0.5,  # 組み合わせ爆発するので少なめ
            OptimizationMethod.RANDOM: 1.5,  # ランダムなので多めに試行
        }

        # 時間の調整係数
        time_multipliers = {
            OptimizationMethod.BAYESIAN: 1.0,
            OptimizationMethod.GRID: 1.2,  # 計算が重い
            OptimizationMethod.RANDOM: 0.8,  # 計算が軽い
        }

        base_time = {
            OptimizationLevel.HIGH: 60,
            OptimizationLevel.MEDIUM: 30,
            OptimizationLevel.LOW: 15,
        }

        n_calls = int(base_calls[level] * method_multipliers[method])
        estimated_time = int(base_time[level] * time_multipliers[method])

        return {"n_calls": n_calls, "estimated_time": estimated_time}

    @staticmethod
    def _get_high_parameter_space() -> Dict[str, Any]:
        """ハイ（高精度）レベルのパラメータ空間"""
        return {
            "learning_rate": {"type": "real", "low": 0.01, "high": 0.2},
            "num_leaves": {"type": "integer", "low": 20, "high": 100},
            "feature_fraction": {"type": "real", "low": 0.7, "high": 1.0},
            "bagging_fraction": {"type": "real", "low": 0.7, "high": 1.0},
            "max_depth": {"type": "integer", "low": 5, "high": 15},
        }

    @staticmethod
    def _get_medium_parameter_space() -> Dict[str, Any]:
        """ミドル（標準）レベルのパラメータ空間"""
        return {
            "learning_rate": {"type": "real", "low": 0.05, "high": 0.15},
            "num_leaves": {"type": "integer", "low": 30, "high": 80},
            "feature_fraction": {"type": "real", "low": 0.8, "high": 0.95},
        }

    @staticmethod
    def _get_low_parameter_space() -> Dict[str, Any]:
        """ロー（高速）レベルのパラメータ空間"""
        return {
            "learning_rate": {"type": "real", "low": 0.05, "high": 0.2},
            "num_leaves": {"type": "integer", "low": 20, "high": 60},
        }

    @staticmethod
    def preset_to_dict(preset: OptimizationPreset) -> Dict[str, Any]:
        """プリセットを辞書形式に変換"""
        return {
            "name": preset.name,
            "description": preset.description,
            "method": preset.method,
            "n_calls": preset.n_calls,
            "parameter_space": preset.parameter_space,
            "estimated_time_minutes": preset.estimated_time_minutes,
        }
