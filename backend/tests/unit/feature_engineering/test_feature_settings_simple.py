"""
FinancialFeatureSettingsの簡単なテスト
"""

import pytest
from app.services.ml.feature_engineering.automl_features.feature_settings import (
    FinancialFeatureSettings,
    FeatureCategory,
    MarketRegime,
    FeatureProfile
)


class TestFinancialFeatureSettings:
    """FinancialFeatureSettingsのテストクラス"""

    def setup_method(self):
        """各テストメソッドの前に実行される初期化"""
        self.settings = FinancialFeatureSettings()

    def test_initialization(self):
        """初期化テスト"""
        assert self.settings.profiles is not None
        assert len(self.settings.profiles) > 0

    def test_get_profile(self):
        """プロファイル取得テスト"""
        # 利用可能なプロファイル名を取得
        profile_names = self.settings.get_all_profile_names()
        assert len(profile_names) > 0
        
        # 最初のプロファイルを取得してテスト
        first_profile_name = profile_names[0]
        profile = self.settings.get_profile(first_profile_name)
        
        assert profile is not None
        assert isinstance(profile.settings, dict)

    def test_get_profile_invalid(self):
        """無効なプロファイル名でのテスト"""
        invalid_profile = self.settings.get_profile("invalid_profile")
        assert invalid_profile is None

    def test_create_custom_settings(self):
        """カスタム設定作成テスト"""
        profile_names = self.settings.get_all_profile_names()
        if len(profile_names) > 0:
            custom_settings = self.settings.create_custom_settings([profile_names[0]])
            assert isinstance(custom_settings, dict)

    def test_get_profiles_by_market_regime(self):
        """市場レジーム別プロファイル取得テスト"""
        trending_profiles = self.settings.get_profiles_by_market_regime(MarketRegime.TRENDING)
        assert isinstance(trending_profiles, list)

    def test_get_all_profile_names(self):
        """全プロファイル名取得テスト"""
        names = self.settings.get_all_profile_names()
        assert isinstance(names, list)
        assert len(names) > 0

    def test_get_profile_summary(self):
        """プロファイル概要取得テスト"""
        summary = self.settings.get_profile_summary()
        assert isinstance(summary, dict)
        assert len(summary) > 0


class TestEnums:
    """Enumクラスのテストクラス"""

    def test_feature_category_enum(self):
        """FeatureCategoryのテスト"""
        categories = list(FeatureCategory)
        assert len(categories) > 0

    def test_market_regime_enum(self):
        """MarketRegimeのテスト"""
        regimes = list(MarketRegime)
        assert len(regimes) > 0
