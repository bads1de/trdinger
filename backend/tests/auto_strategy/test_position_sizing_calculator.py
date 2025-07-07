"""
PositionSizingCalculatorServiceクラスのテスト
"""

import pytest
import sys
import os
from datetime import datetime, timedelta

# テスト対象のモジュールをインポートするためのパス設定
sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "app",
        "core",
        "services",
        "auto_strategy",
        "models",
    )
)

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "app",
        "core",
        "services",
        "auto_strategy",
        "calculators",
    )
)

from position_sizing_gene import PositionSizingGene, PositionSizingMethod
from position_sizing_calculator import (
    PositionSizingCalculatorService,
    PositionSizingResult,
    MarketDataCache,
)


class TestPositionSizingCalculatorService:
    """PositionSizingCalculatorServiceクラスのテスト"""

    def setup_method(self):
        """各テストメソッドの前に実行される初期化"""
        self.calculator = PositionSizingCalculatorService()
        self.account_balance = 10000.0
        self.current_price = 50000.0
        self.symbol = "BTCUSDT"

    def test_initialization(self):
        """初期化のテスト"""
        calculator = PositionSizingCalculatorService()
        
        assert calculator._cache is None
        assert len(calculator._calculation_history) == 0

    def test_calculate_position_size_fixed_ratio(self):
        """固定比率方式の計算テスト"""
        gene = PositionSizingGene(
            method=PositionSizingMethod.FIXED_RATIO,
            fixed_ratio=0.2,
            min_position_size=0.01,
            max_position_size=5.0,
        )
        
        result = self.calculator.calculate_position_size(
            gene=gene,
            account_balance=self.account_balance,
            current_price=self.current_price,
            symbol=self.symbol,
        )
        
        assert isinstance(result, PositionSizingResult)
        assert result.position_size == self.account_balance * 0.2  # 2000.0
        assert result.method_used == "fixed_ratio"
        assert result.confidence_score > 0
        assert len(result.warnings) == 0
        assert "fixed_ratio" in result.calculation_details
        assert result.calculation_details["account_balance"] == self.account_balance

    def test_calculate_position_size_fixed_quantity(self):
        """固定枚数方式の計算テスト"""
        gene = PositionSizingGene(
            method=PositionSizingMethod.FIXED_QUANTITY,
            fixed_quantity=3.0,
            min_position_size=0.01,
            max_position_size=5.0,
        )
        
        result = self.calculator.calculate_position_size(
            gene=gene,
            account_balance=self.account_balance,
            current_price=self.current_price,
            symbol=self.symbol,
        )
        
        assert result.position_size == 3.0
        assert result.method_used == "fixed_quantity"
        assert "fixed_quantity" in result.calculation_details

    def test_calculate_position_size_volatility_based(self):
        """ボラティリティベース方式の計算テスト"""
        gene = PositionSizingGene(
            method=PositionSizingMethod.VOLATILITY_BASED,
            atr_multiplier=2.0,
            risk_per_trade=0.02,
            min_position_size=0.01,
            max_position_size=5.0,
        )
        
        market_data = {"atr": 1000.0, "atr_source": "real"}
        
        result = self.calculator.calculate_position_size(
            gene=gene,
            account_balance=self.account_balance,
            current_price=self.current_price,
            symbol=self.symbol,
            market_data=market_data,
        )
        
        # risk_amount = 10000 * 0.02 = 200
        # atr_pct = 1000 / 50000 = 0.02
        # volatility_factor = 0.02 * 2.0 = 0.04
        # position_size = 200 / (50000 * 0.04) = 0.1
        expected = 200.0 / (self.current_price * 0.02 * 2.0)
        assert result.position_size == expected
        assert result.method_used == "volatility_based"
        assert result.calculation_details["atr_value"] == 1000.0
        assert result.calculation_details["atr_source"] == "real"

    def test_calculate_position_size_volatility_based_default_atr(self):
        """ボラティリティベース方式（デフォルトATR）の計算テスト"""
        gene = PositionSizingGene(
            method=PositionSizingMethod.VOLATILITY_BASED,
            atr_multiplier=2.0,
            risk_per_trade=0.02,
        )
        
        result = self.calculator.calculate_position_size(
            gene=gene,
            account_balance=self.account_balance,
            current_price=self.current_price,
            symbol=self.symbol,
        )
        
        # デフォルトATR = 50000 * 0.02 = 1000
        assert result.calculation_details["atr_source"] == "default"
        assert result.calculation_details["atr_value"] == self.current_price * 0.02

    def test_calculate_position_size_half_optimal_f_insufficient_data(self):
        """ハーフオプティマルF方式（データ不足）の計算テスト"""
        gene = PositionSizingGene(
            method=PositionSizingMethod.HALF_OPTIMAL_F,
            fixed_ratio=0.1,  # フォールバック用
        )
        
        # データ不足の場合
        trade_history = []
        
        result = self.calculator.calculate_position_size(
            gene=gene,
            account_balance=self.account_balance,
            current_price=self.current_price,
            symbol=self.symbol,
            trade_history=trade_history,
        )
        
        # データ不足時は固定比率にフォールバック
        assert result.position_size == self.account_balance * 0.1
        assert result.method_used == "half_optimal_f"
        assert any("フォールバック" in warning for warning in result.warnings)
        assert result.calculation_details["fallback_reason"] == "insufficient_trade_history"

    def test_calculate_position_size_half_optimal_f_with_data(self):
        """ハーフオプティマルF方式（データあり）の計算テスト"""
        gene = PositionSizingGene(
            method=PositionSizingMethod.HALF_OPTIMAL_F,
            optimal_f_multiplier=0.5,
            lookback_period=10,
        )
        
        # 勝率60%、平均利益100、平均損失50のサンプルデータ
        trade_history = [
            {"pnl": 100}, {"pnl": -50}, {"pnl": 100}, {"pnl": 100}, {"pnl": -50},
            {"pnl": 100}, {"pnl": -50}, {"pnl": 100}, {"pnl": -50}, {"pnl": 100},
        ]
        
        result = self.calculator.calculate_position_size(
            gene=gene,
            account_balance=self.account_balance,
            current_price=self.current_price,
            symbol=self.symbol,
            trade_history=trade_history,
        )
        
        assert result.method_used == "half_optimal_f"
        assert result.position_size > 0
        assert "win_rate" in result.calculation_details
        assert "avg_win" in result.calculation_details
        assert "avg_loss" in result.calculation_details
        assert "optimal_f" in result.calculation_details
        assert result.calculation_details["trade_count"] == 10

    def test_calculate_position_size_with_size_limits(self):
        """サイズ制限の適用テスト"""
        # 最大サイズを超える場合
        gene = PositionSizingGene(
            method=PositionSizingMethod.FIXED_RATIO,
            fixed_ratio=1.0,  # 100%（非現実的な値）
            min_position_size=0.01,
            max_position_size=1.0,  # 最大1.0に制限
        )
        
        result = self.calculator.calculate_position_size(
            gene=gene,
            account_balance=self.account_balance,
            current_price=self.current_price,
            symbol=self.symbol,
        )
        
        assert result.position_size == 1.0  # 最大値に制限される

    def test_calculate_position_size_invalid_inputs(self):
        """無効な入力値のテスト"""
        gene = PositionSizingGene()
        
        # 負の口座残高
        result = self.calculator.calculate_position_size(
            gene=gene,
            account_balance=-1000.0,
            current_price=self.current_price,
            symbol=self.symbol,
        )
        
        assert result.position_size == 0.01  # エラー時の最小サイズ
        assert len(result.warnings) > 0
        assert "error" in result.calculation_details

    def test_calculate_position_size_none_gene(self):
        """遺伝子がNoneの場合のテスト"""
        result = self.calculator.calculate_position_size(
            gene=None,
            account_balance=self.account_balance,
            current_price=self.current_price,
            symbol=self.symbol,
        )
        
        assert result.position_size == 0.01
        assert result.method_used == "unknown"
        assert len(result.warnings) > 0

    def test_risk_metrics_calculation(self):
        """リスクメトリクス計算のテスト"""
        gene = PositionSizingGene(
            method=PositionSizingMethod.FIXED_RATIO,
            fixed_ratio=0.1,
        )
        
        market_data = {"atr_pct": 0.03}
        
        result = self.calculator.calculate_position_size(
            gene=gene,
            account_balance=self.account_balance,
            current_price=self.current_price,
            symbol=self.symbol,
            market_data=market_data,
        )
        
        risk_metrics = result.risk_metrics
        assert "position_value" in risk_metrics
        assert "position_ratio" in risk_metrics
        assert "potential_loss_1atr" in risk_metrics
        assert "potential_loss_ratio" in risk_metrics
        assert "atr_used" in risk_metrics
        
        # 値の妥当性チェック
        expected_position_value = result.position_size * self.current_price
        assert risk_metrics["position_value"] == expected_position_value
        assert risk_metrics["atr_used"] == 0.03

    def test_confidence_score_calculation(self):
        """信頼度スコア計算のテスト"""
        gene = PositionSizingGene()
        
        # 高品質データでのテスト
        market_data = {"atr_source": "real"}
        trade_history = [{"pnl": 100} for _ in range(60)]  # 十分なデータ
        
        result = self.calculator.calculate_position_size(
            gene=gene,
            account_balance=self.account_balance,
            current_price=self.current_price,
            symbol=self.symbol,
            market_data=market_data,
            trade_history=trade_history,
        )
        
        assert 0.0 <= result.confidence_score <= 1.0
        # 高品質データなので高い信頼度が期待される
        assert result.confidence_score > 0.5

    def test_calculation_history(self):
        """計算履歴のテスト"""
        gene = PositionSizingGene()
        
        # 複数回計算を実行
        for i in range(3):
            self.calculator.calculate_position_size(
                gene=gene,
                account_balance=self.account_balance + i * 1000,
                current_price=self.current_price,
                symbol=self.symbol,
            )
        
        history = self.calculator.get_calculation_history()
        assert len(history) == 3
        
        # 履歴の順序確認
        for i, result in enumerate(history):
            assert result.calculation_details["account_balance"] == self.account_balance + i * 1000

    def test_cache_functionality(self):
        """キャッシュ機能のテスト"""
        # 初期状態ではキャッシュなし
        cache_status = self.calculator.get_cache_status()
        assert cache_status["cached"] is False
        
        # キャッシュクリア
        self.calculator.clear_cache()
        cache_status = self.calculator.get_cache_status()
        assert cache_status["cached"] is False

    def test_market_data_cache_expiration(self):
        """市場データキャッシュの期限切れテスト"""
        # 期限切れのキャッシュを作成
        expired_cache = MarketDataCache(
            atr_values={"BTCUSDT": 1000.0},
            volatility_metrics={"volatility": 0.02},
            price_data=None,
            last_updated=datetime.now() - timedelta(minutes=10),
        )
        
        assert expired_cache.is_expired(max_age_minutes=5) is True
        
        # 有効なキャッシュ
        valid_cache = MarketDataCache(
            atr_values={"BTCUSDT": 1000.0},
            volatility_metrics={"volatility": 0.02},
            price_data=None,
            last_updated=datetime.now(),
        )
        
        assert valid_cache.is_expired(max_age_minutes=5) is False

    def test_unknown_method_fallback(self):
        """未知の手法のフォールバックテスト"""
        gene = PositionSizingGene(
            method=PositionSizingMethod.FIXED_RATIO,  # 正常な値で初期化
            fixed_ratio=0.15,
        )
        
        # 手法を無効な値に変更（テスト用）
        gene.method = "unknown_method"  # type: ignore
        
        result = self.calculator.calculate_position_size(
            gene=gene,
            account_balance=self.account_balance,
            current_price=self.current_price,
            symbol=self.symbol,
        )
        
        # 固定比率にフォールバックされることを確認
        assert result.position_size == self.account_balance * 0.15
        assert any("フォールバック" in warning for warning in result.warnings)

    def test_calculation_performance(self):
        """計算パフォーマンスのテスト"""
        gene = PositionSizingGene()
        
        result = self.calculator.calculate_position_size(
            gene=gene,
            account_balance=self.account_balance,
            current_price=self.current_price,
            symbol=self.symbol,
        )
        
        # 計算時間が記録されていることを確認
        assert "calculation_time_seconds" in result.calculation_details
        assert result.calculation_details["calculation_time_seconds"] >= 0

    def test_large_calculation_history_limit(self):
        """大量の計算履歴の制限テスト"""
        gene = PositionSizingGene()
        
        # 1100回計算を実行（制限の1000を超える）
        for i in range(1100):
            self.calculator.calculate_position_size(
                gene=gene,
                account_balance=self.account_balance,
                current_price=self.current_price,
                symbol=self.symbol,
            )
        
        # 履歴が500に制限されることを確認
        history = self.calculator.get_calculation_history()
        assert len(history) == 500


if __name__ == "__main__":
    pytest.main([__file__])
