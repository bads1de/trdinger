"""
カスタム特徴量設定管理

金融データに特化した高度な特徴量設定を管理します。
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import json


class FeatureCategory(Enum):
    """特徴量カテゴリ"""
    STATISTICAL = "statistical"
    TEMPORAL = "temporal"
    FREQUENCY = "frequency"
    ENTROPY = "entropy"
    TREND = "trend"
    EXTREMES = "extremes"
    CHANGE_POINTS = "change_points"
    COMPLEXITY = "complexity"


class MarketRegime(Enum):
    """市場レジーム"""
    TRENDING = "trending"
    RANGING = "ranging"
    VOLATILE = "volatile"
    CALM = "calm"


@dataclass
class FeatureProfile:
    """特徴量プロファイル"""
    name: str
    description: str
    categories: List[FeatureCategory]
    market_regimes: List[MarketRegime]
    computational_cost: int  # 1-10のスケール
    feature_count: int
    settings: Dict[str, Any]


class FinancialFeatureSettings:
    """金融データ専用特徴量設定管理クラス"""

    def __init__(self):
        """初期化"""
        self.profiles = self._initialize_profiles()

    def _initialize_profiles(self) -> Dict[str, FeatureProfile]:
        """特徴量プロファイルを初期化"""
        profiles = {}

        # 基本統計プロファイル
        profiles["basic_stats"] = FeatureProfile(
            name="基本統計",
            description="平均、標準偏差、歪度、尖度などの基本統計量",
            categories=[FeatureCategory.STATISTICAL],
            market_regimes=[MarketRegime.TRENDING, MarketRegime.RANGING, MarketRegime.VOLATILE, MarketRegime.CALM],
            computational_cost=2,
            feature_count=8,
            settings={
                'mean': None,
                'median': None,
                'std': None,
                'var': None,
                'skewness': None,
                'kurtosis': None,
                'minimum': None,
                'maximum': None,
            }
        )

        # 高度統計プロファイル
        profiles["advanced_stats"] = FeatureProfile(
            name="高度統計",
            description="分位点、範囲統計、モーメント統計",
            categories=[FeatureCategory.STATISTICAL],
            market_regimes=[MarketRegime.VOLATILE, MarketRegime.TRENDING],
            computational_cost=3,
            feature_count=12,
            settings={
                'quantile': [
                    {'q': 0.05}, {'q': 0.1}, {'q': 0.25}, 
                    {'q': 0.75}, {'q': 0.9}, {'q': 0.95}
                ],
                'range_count': [
                    {'min': -2, 'max': 2},
                    {'min': -1, 'max': 1},
                    {'min': -0.5, 'max': 0.5}
                ],
                'count_above_mean': None,
                'count_below_mean': None,
                'ratio_beyond_r_sigma': [{'r': 1}, {'r': 2}, {'r': 3}],
            }
        )

        # 時系列プロファイル
        profiles["temporal"] = FeatureProfile(
            name="時系列",
            description="自己相関、部分自己相関、時系列パターン",
            categories=[FeatureCategory.TEMPORAL],
            market_regimes=[MarketRegime.TRENDING, MarketRegime.RANGING],
            computational_cost=4,
            feature_count=15,
            settings={
                'autocorrelation': [{'lag': i} for i in [1, 2, 3, 5, 10, 20, 50]],
                'partial_autocorrelation': [{'lag': i} for i in [1, 2, 3, 5, 10]],
                'c3': [{'lag': i} for i in [1, 2, 3]],
                'cid_ce': [{'normalize': True}, {'normalize': False}],
                'symmetry_looking': [{'r': 0.1}, {'r': 0.2}],
            }
        )

        # 周波数領域プロファイル
        profiles["frequency"] = FeatureProfile(
            name="周波数領域",
            description="FFT係数、スペクトル密度、周波数統計",
            categories=[FeatureCategory.FREQUENCY],
            market_regimes=[MarketRegime.TRENDING, MarketRegime.VOLATILE],
            computational_cost=6,
            feature_count=20,
            settings={
                'fft_coefficient': [
                    {'coeff': i, 'attr': 'real'} for i in range(10)
                ] + [
                    {'coeff': i, 'attr': 'imag'} for i in range(10)
                ],
                'fft_aggregated': [
                    {'aggtype': 'centroid'},
                    {'aggtype': 'variance'},
                    {'aggtype': 'skew'},
                    {'aggtype': 'kurtosis'}
                ],
                'spkt_welch_density': [{'coeff': i} for i in range(5)],
            }
        )

        # エントロピープロファイル
        profiles["entropy"] = FeatureProfile(
            name="エントロピー",
            description="サンプルエントロピー、近似エントロピー、複雑性指標",
            categories=[FeatureCategory.ENTROPY, FeatureCategory.COMPLEXITY],
            market_regimes=[MarketRegime.VOLATILE, MarketRegime.CALM],
            computational_cost=8,
            feature_count=10,
            settings={
                'sample_entropy': None,
                'approximate_entropy': [{'m': 2, 'r': 0.1}, {'m': 2, 'r': 0.2}],
                'permutation_entropy': [{'order': 3}, {'order': 4}],
                'svd_entropy': [{'order': 3}, {'order': 4}],
                'lempel_ziv_complexity': [{'bins': 10}, {'bins': 20}],
                'fourier_entropy': [{'bins': 10}, {'bins': 20}],
            }
        )

        # トレンドプロファイル
        profiles["trend"] = FeatureProfile(
            name="トレンド",
            description="線形トレンド、集約トレンド、トレンド変化",
            categories=[FeatureCategory.TREND],
            market_regimes=[MarketRegime.TRENDING, MarketRegime.RANGING],
            computational_cost=4,
            feature_count=12,
            settings={
                'linear_trend': [
                    {'attr': 'slope'}, 
                    {'attr': 'intercept'},
                    {'attr': 'rvalue'},
                    {'attr': 'pvalue'},
                    {'attr': 'stderr'}
                ],
                'agg_linear_trend': [
                    {'attr': 'slope', 'chunk_len': 5, 'f_agg': 'mean'},
                    {'attr': 'slope', 'chunk_len': 10, 'f_agg': 'mean'},
                    {'attr': 'slope', 'chunk_len': 5, 'f_agg': 'std'},
                    {'attr': 'slope', 'chunk_len': 10, 'f_agg': 'std'},
                    {'attr': 'intercept', 'chunk_len': 5, 'f_agg': 'mean'},
                    {'attr': 'intercept', 'chunk_len': 10, 'f_agg': 'mean'},
                ],
                'augmented_dickey_fuller': [
                    {'attr': 'teststat'},
                    {'attr': 'pvalue'}
                ],
            }
        )

        # 極値プロファイル
        profiles["extremes"] = FeatureProfile(
            name="極値",
            description="ピーク検出、極値統計、異常値検出",
            categories=[FeatureCategory.EXTREMES],
            market_regimes=[MarketRegime.VOLATILE, MarketRegime.TRENDING],
            computational_cost=5,
            feature_count=15,
            settings={
                'number_peaks': [{'n': 1}, {'n': 3}, {'n': 5}, {'n': 10}],
                'number_cwt_peaks': [{'n': 1}, {'n': 3}, {'n': 5}],
                'peak_to_peak_distance': None,
                'abs_energy': None,
                'mean_abs_change': None,
                'mean_change': None,
                'mean_second_derivative_central': None,
                'median_change': None,
                'variation_coefficient': None,
            }
        )

        # 変化点プロファイル
        profiles["change_points"] = FeatureProfile(
            name="変化点",
            description="分位点変化、レベル変化、ボラティリティ変化",
            categories=[FeatureCategory.CHANGE_POINTS],
            market_regimes=[MarketRegime.VOLATILE, MarketRegime.TRENDING],
            computational_cost=6,
            feature_count=10,
            settings={
                'change_quantiles': [
                    {'ql': 0.0, 'qh': 0.2, 'isabs': False},
                    {'ql': 0.0, 'qh': 0.4, 'isabs': False},
                    {'ql': 0.6, 'qh': 1.0, 'isabs': False},
                    {'ql': 0.8, 'qh': 1.0, 'isabs': False},
                    {'ql': 0.0, 'qh': 0.2, 'isabs': True},
                    {'ql': 0.8, 'qh': 1.0, 'isabs': True},
                ],
                'ratio_value_number_to_time_series_length': None,
                'first_location_of_maximum': None,
                'first_location_of_minimum': None,
                'last_location_of_maximum': None,
                'last_location_of_minimum': None,
            }
        )

        return profiles

    def get_profile(self, profile_name: str) -> Optional[FeatureProfile]:
        """特徴量プロファイルを取得"""
        return self.profiles.get(profile_name)

    def get_profiles_by_category(self, category: FeatureCategory) -> List[FeatureProfile]:
        """カテゴリ別の特徴量プロファイルを取得"""
        return [
            profile for profile in self.profiles.values()
            if category in profile.categories
        ]

    def get_profiles_by_market_regime(self, regime: MarketRegime) -> List[FeatureProfile]:
        """市場レジーム別の特徴量プロファイルを取得"""
        return [
            profile for profile in self.profiles.values()
            if regime in profile.market_regimes
        ]

    def create_custom_settings(
        self,
        profile_names: List[str],
        max_computational_cost: int = 10,
        max_features: int = 100
    ) -> Dict[str, Any]:
        """カスタム特徴量設定を作成"""
        combined_settings = {}
        total_cost = 0
        total_features = 0

        # 計算コストと特徴量数でソート
        sorted_profiles = sorted(
            [self.profiles[name] for name in profile_names if name in self.profiles],
            key=lambda p: (p.computational_cost, p.feature_count)
        )

        for profile in sorted_profiles:
            if (total_cost + profile.computational_cost <= max_computational_cost and
                total_features + profile.feature_count <= max_features):
                
                combined_settings.update(profile.settings)
                total_cost += profile.computational_cost
                total_features += profile.feature_count

        return combined_settings

    def get_optimized_settings_for_regime(
        self,
        regime: MarketRegime,
        max_computational_cost: int = 8,
        max_features: int = 80
    ) -> Dict[str, Any]:
        """市場レジーム最適化設定を取得"""
        suitable_profiles = self.get_profiles_by_market_regime(regime)
        profile_names = [profile.name for profile in suitable_profiles]
        
        return self.create_custom_settings(
            profile_names,
            max_computational_cost,
            max_features
        )

    def save_settings_to_file(self, settings: Dict[str, Any], filepath: str):
        """設定をファイルに保存"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=2, ensure_ascii=False)

    def load_settings_from_file(self, filepath: str) -> Dict[str, Any]:
        """ファイルから設定を読み込み"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_all_profile_names(self) -> List[str]:
        """全プロファイル名を取得"""
        return list(self.profiles.keys())

    def get_profile_summary(self) -> Dict[str, Dict[str, Any]]:
        """プロファイル概要を取得"""
        summary = {}
        for name, profile in self.profiles.items():
            summary[name] = {
                "description": profile.description,
                "categories": [cat.value for cat in profile.categories],
                "market_regimes": [regime.value for regime in profile.market_regimes],
                "computational_cost": profile.computational_cost,
                "feature_count": profile.feature_count
            }
        return summary
