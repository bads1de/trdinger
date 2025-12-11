"""
MarketDataHandlerのテスト

市場データの準備、キャッシュ管理のテスト
"""

import pytest
from datetime import datetime, timedelta

# unittest.mockは現在未使用だが、将来のテスト拡張用に残す

import pandas as pd


class TestMarketDataCache:
    """MarketDataCacheのテスト"""

    def test_cache_initialization(self):
        """キャッシュの初期化"""
        from app.services.auto_strategy.positions.market_data_handler import (
            MarketDataCache,
        )

        now = datetime.now()
        cache = MarketDataCache(
            atr_values={"atr": 100.0},
            volatility_metrics={"volatility": 0.02},
            price_data=None,
            last_updated=now,
        )

        assert cache.atr_values == {"atr": 100.0}
        assert cache.volatility_metrics == {"volatility": 0.02}
        assert cache.price_data is None
        assert cache.last_updated == now

    def test_is_expired_returns_false_for_fresh_cache(self):
        """新鮮なキャッシュは期限切れではない"""
        from app.services.auto_strategy.positions.market_data_handler import (
            MarketDataCache,
        )

        cache = MarketDataCache(
            atr_values={},
            volatility_metrics={},
            price_data=None,
            last_updated=datetime.now(),
        )

        assert cache.is_expired(max_age_minutes=5) is False

    def test_is_expired_returns_true_for_old_cache(self):
        """古いキャッシュは期限切れ"""
        from app.services.auto_strategy.positions.market_data_handler import (
            MarketDataCache,
        )

        old_time = datetime.now() - timedelta(minutes=10)
        cache = MarketDataCache(
            atr_values={},
            volatility_metrics={},
            price_data=None,
            last_updated=old_time,
        )

        assert cache.is_expired(max_age_minutes=5) is True

    def test_is_expired_with_custom_max_age(self):
        """カスタム最大経過時間で期限切れ判定"""
        from app.services.auto_strategy.positions.market_data_handler import (
            MarketDataCache,
        )

        mid_time = datetime.now() - timedelta(minutes=8)
        cache = MarketDataCache(
            atr_values={},
            volatility_metrics={},
            price_data=None,
            last_updated=mid_time,
        )

        # 5分のmax_ageでは期限切れ
        assert cache.is_expired(max_age_minutes=5) is True
        # 10分のmax_ageでは期限切れではない
        assert cache.is_expired(max_age_minutes=10) is False


class TestMarketDataHandlerInit:
    """MarketDataHandler初期化のテスト"""

    def test_init_creates_empty_cache(self):
        """初期状態でキャッシュはNone"""
        from app.services.auto_strategy.positions.market_data_handler import (
            MarketDataHandler,
        )

        handler = MarketDataHandler()
        assert handler._cache is None


class TestPrepareMarketData:
    """prepare_market_dataのテスト"""

    @pytest.fixture
    def handler(self):
        """テスト用ハンドラ"""
        from app.services.auto_strategy.positions.market_data_handler import (
            MarketDataHandler,
        )

        return MarketDataHandler()

    def test_returns_enhanced_data_with_defaults(self, handler):
        """デフォルト値を含む拡張データを返す"""
        result = handler.prepare_market_data(
            symbol="BTCUSDT",
            current_price=50000.0,
            market_data=None,
            use_cache=False,
        )

        assert "atr" in result
        assert "atr_pct" in result
        assert "atr_source" in result
        assert result["atr_source"] == "default"

    def test_preserves_existing_market_data(self, handler):
        """既存の市場データを保持"""
        market_data = {"custom_field": "custom_value", "atr": 500.0}
        result = handler.prepare_market_data(
            symbol="BTCUSDT",
            current_price=50000.0,
            market_data=market_data,
            use_cache=False,
        )

        assert result["custom_field"] == "custom_value"
        assert result["atr"] == 500.0

    def test_uses_cache_when_available_and_valid(self, handler):
        """有効なキャッシュがある場合はキャッシュを使用"""
        from app.services.auto_strategy.positions.market_data_handler import (
            MarketDataCache,
        )

        handler._cache = MarketDataCache(
            atr_values={"atr": 1000.0, "atr_pct": 0.02},
            volatility_metrics={"volatility": 0.03},
            price_data=None,
            last_updated=datetime.now(),
        )

        result = handler.prepare_market_data(
            symbol="BTCUSDT",
            current_price=50000.0,
            market_data={},
            use_cache=True,
        )

        assert result["atr"] == 1000.0
        assert result["atr_pct"] == 0.02
        assert result["volatility"] == 0.03

    def test_skips_cache_when_expired(self, handler):
        """期限切れのキャッシュはスキップ"""
        from app.services.auto_strategy.positions.market_data_handler import (
            MarketDataCache,
        )

        old_time = datetime.now() - timedelta(minutes=10)
        handler._cache = MarketDataCache(
            atr_values={"atr": 1000.0},
            volatility_metrics={},
            price_data=None,
            last_updated=old_time,
        )

        result = handler.prepare_market_data(
            symbol="BTCUSDT",
            current_price=50000.0,
            market_data={},
            use_cache=True,
        )

        # 期限切れのためデフォルト値が使用される
        assert result.get("atr_source") == "default"

    def test_skips_cache_when_use_cache_false(self, handler):
        """use_cache=Falseの場合はキャッシュをスキップ"""
        from app.services.auto_strategy.positions.market_data_handler import (
            MarketDataCache,
        )

        handler._cache = MarketDataCache(
            atr_values={"atr": 1000.0},
            volatility_metrics={},
            price_data=None,
            last_updated=datetime.now(),
        )

        result = handler.prepare_market_data(
            symbol="BTCUSDT",
            current_price=50000.0,
            market_data={},
            use_cache=False,  # キャッシュを使用しない
        )

        # キャッシュ値が適用されていない（データにATRがない場合デフォルトが使用される）
        assert result.get("atr_source") == "default"

    def test_adds_volatility_from_atr_pct(self, handler):
        """atr_pctからボラティリティを追加"""
        market_data = {"atr_pct": 0.025}
        result = handler.prepare_market_data(
            symbol="BTCUSDT",
            current_price=50000.0,
            market_data=market_data,
            use_cache=False,
        )

        assert result["volatility"] == 0.025


class TestUpdateCache:
    """update_cacheのテスト"""

    def test_updates_cache_with_new_values(self):
        """新しい値でキャッシュを更新"""
        from app.services.auto_strategy.positions.market_data_handler import (
            MarketDataHandler,
            MarketDataCache,
        )

        handler = MarketDataHandler()
        df = pd.DataFrame({"close": [100, 101, 102]})

        handler.update_cache(
            atr_values={"atr": 500.0},
            volatility_metrics={"volatility": 0.02},
            price_data=df,
        )

        assert handler._cache is not None
        assert handler._cache.atr_values == {"atr": 500.0}
        assert handler._cache.volatility_metrics == {"volatility": 0.02}
        pd.testing.assert_frame_equal(handler._cache.price_data, df)

    def test_updates_last_updated_timestamp(self):
        """last_updatedタイムスタンプを更新"""
        from app.services.auto_strategy.positions.market_data_handler import (
            MarketDataHandler,
        )

        handler = MarketDataHandler()
        before_update = datetime.now()

        handler.update_cache(
            atr_values={},
            volatility_metrics={},
        )

        after_update = datetime.now()
        assert before_update <= handler._cache.last_updated <= after_update


class TestGetCache:
    """get_cacheのテスト"""

    def test_returns_none_when_no_cache(self):
        """キャッシュがない場合はNoneを返す"""
        from app.services.auto_strategy.positions.market_data_handler import (
            MarketDataHandler,
        )

        handler = MarketDataHandler()
        assert handler.get_cache() is None

    def test_returns_cache_when_exists(self):
        """キャッシュが存在する場合はキャッシュを返す"""
        from app.services.auto_strategy.positions.market_data_handler import (
            MarketDataHandler,
            MarketDataCache,
        )

        handler = MarketDataHandler()
        handler._cache = MarketDataCache(
            atr_values={"atr": 100.0},
            volatility_metrics={},
            price_data=None,
            last_updated=datetime.now(),
        )

        cache = handler.get_cache()
        assert cache is not None
        assert cache.atr_values == {"atr": 100.0}


class TestClearCache:
    """clear_cacheのテスト"""

    def test_clears_existing_cache(self):
        """既存のキャッシュをクリア"""
        from app.services.auto_strategy.positions.market_data_handler import (
            MarketDataHandler,
            MarketDataCache,
        )

        handler = MarketDataHandler()
        handler._cache = MarketDataCache(
            atr_values={},
            volatility_metrics={},
            price_data=None,
            last_updated=datetime.now(),
        )

        handler.clear_cache()
        assert handler._cache is None

    def test_is_idempotent(self):
        """クリアは冪等"""
        from app.services.auto_strategy.positions.market_data_handler import (
            MarketDataHandler,
        )

        handler = MarketDataHandler()
        handler.clear_cache()
        handler.clear_cache()  # 2回呼び出してもエラーにならない
        assert handler._cache is None


class TestIsCacheValid:
    """is_cache_validのテスト"""

    def test_returns_false_when_no_cache(self):
        """キャッシュがない場合はFalse"""
        from app.services.auto_strategy.positions.market_data_handler import (
            MarketDataHandler,
        )

        handler = MarketDataHandler()
        assert handler.is_cache_valid() is False

    def test_returns_true_for_fresh_cache(self):
        """新鮮なキャッシュの場合はTrue"""
        from app.services.auto_strategy.positions.market_data_handler import (
            MarketDataHandler,
            MarketDataCache,
        )

        handler = MarketDataHandler()
        handler._cache = MarketDataCache(
            atr_values={},
            volatility_metrics={},
            price_data=None,
            last_updated=datetime.now(),
        )

        assert handler.is_cache_valid() is True

    def test_returns_false_for_expired_cache(self):
        """期限切れのキャッシュの場合はFalse"""
        from app.services.auto_strategy.positions.market_data_handler import (
            MarketDataHandler,
            MarketDataCache,
        )

        handler = MarketDataHandler()
        old_time = datetime.now() - timedelta(minutes=10)
        handler._cache = MarketDataCache(
            atr_values={},
            volatility_metrics={},
            price_data=None,
            last_updated=old_time,
        )

        assert handler.is_cache_valid() is False
