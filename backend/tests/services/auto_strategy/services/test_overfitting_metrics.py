"""
過学習メトリクス（overfitting_metrics）のテスト

PBO 推定・DSR・テスト窓年数推定の各ロジックをテストします。
"""

from math import isclose

import pytest

from app.services.auto_strategy.services.overfitting_metrics import (
    deflated_sharpe_ratio,
    estimate_test_window_years,
    expected_max_sharpe_ratio,
    probability_of_backtest_overfitting,
)


class TestProbabilityOfBacktestOverfitting:
    """PBO（負けフォールド比率）のテスト"""

    def test_empty_returns_returns_none(self):
        assert probability_of_backtest_overfitting([]) is None

    def test_all_winning_folds(self):
        assert probability_of_backtest_overfitting([0.1, 0.2, 0.3]) == 0.0

    def test_half_losing_folds(self):
        assert probability_of_backtest_overfitting([0.1, -0.1, 0.2, -0.2]) == 0.5

    def test_majority_losing_folds(self):
        assert probability_of_backtest_overfitting([0.1, -0.2, -0.3]) == pytest.approx(
            2 / 3
        )

    def test_zero_return_not_counted_as_losing(self):
        """total_return == 0 は損失ではないため負けフォールドに数えない"""
        assert probability_of_backtest_overfitting([0.0, 0.1]) == 0.0


class TestExpectedMaxSharpeRatio:
    """帰無仮説下の最大期待シャープレシオのテスト"""

    def test_single_trial_is_zero(self):
        assert expected_max_sharpe_ratio(1) == 0.0

    def test_increases_with_trials(self):
        assert expected_max_sharpe_ratio(100) > expected_max_sharpe_ratio(10)

    def test_known_value(self):
        # sigma=1, N=100: Phi^-1(0.99) ≈ 2.326
        assert isclose(
            expected_max_sharpe_ratio(100, sigma_sharpe=1.0), 2.3263, abs_tol=0.01
        )

    def test_scaled_by_sigma(self):
        assert expected_max_sharpe_ratio(100, sigma_sharpe=2.0) == pytest.approx(
            expected_max_sharpe_ratio(100, sigma_sharpe=1.0) * 2.0
        )


class TestDeflatedSharpeRatio:
    """Deflated Sharpe Ratio のテスト"""

    def test_no_information_returns_neutral(self):
        assert deflated_sharpe_ratio(1.0, n_observations=1.0, n_trials=100) == 0.5
        assert deflated_sharpe_ratio(1.0, n_observations=5.0, n_trials=1) == 0.5

    def test_nan_sharpe_returns_neutral(self):
        assert deflated_sharpe_ratio(float("nan"), 5.0, 100) == 0.5

    def test_zero_sigma_returns_neutral(self):
        assert deflated_sharpe_ratio(1.0, 5.0, 100, sigma_sharpe=0.0) == 0.5

    def test_high_sharpe_with_few_trials_passes(self):
        # 試行数が少なければ観測 SR が有意になりやすい
        dsr = deflated_sharpe_ratio(3.0, n_observations=5.0, n_trials=5)
        assert dsr > 0.95

    def test_many_trials_deflate_high_sharpe(self):
        """試行数が増えると同じ SR でも有意性が下がる（多重検定補正）"""
        dsr_few = deflated_sharpe_ratio(3.0, n_observations=5.0, n_trials=5)
        dsr_many = deflated_sharpe_ratio(3.0, n_observations=5.0, n_trials=5000)
        assert dsr_many < dsr_few

    def test_longer_track_record_increases_dsr(self):
        # SR が帰無分布の最大期待値を上回っている場合、
        # トラックレコードが長いほど有意性が上がる
        dsr_short = deflated_sharpe_ratio(3.0, n_observations=2.0, n_trials=100)
        dsr_long = deflated_sharpe_ratio(3.0, n_observations=5.0, n_trials=100)
        assert dsr_long > dsr_short

    def test_negative_sharpe_never_significant(self):
        dsr = deflated_sharpe_ratio(-1.0, n_observations=5.0, n_trials=10)
        assert dsr < 0.5


class TestEstimateTestWindowYears:
    """WFA テスト窓の年数推定のテスト"""

    def test_nominal_case(self):
        # 1年(365日)のデータ、3フォールド、train_ratio=0.5
        # 各テスト窓 = 365 * 0.5 / 3 ≈ 60.8日 ≈ 0.1667年
        years = estimate_test_window_years(
            "2023-01-01", "2024-01-01", n_folds=3, train_ratio=0.5
        )
        assert years is not None
        assert isclose(years, 365 * 0.5 / 3 / 365.0, abs_tol=1e-9)

    def test_invalid_dates_returns_none(self):
        assert estimate_test_window_years("invalid", "2025-01-01", 3, 0.5) is None
        assert estimate_test_window_years("2025-01-01", "2024-01-01", 3, 0.5) is None

    def test_invalid_fold_params_returns_none(self):
        assert estimate_test_window_years("2024-01-01", "2025-01-01", 0, 0.5) is None
        assert estimate_test_window_years("2024-01-01", "2025-01-01", 3, 0.0) is None
        assert estimate_test_window_years("2024-01-01", "2025-01-01", 3, 1.0) is None
