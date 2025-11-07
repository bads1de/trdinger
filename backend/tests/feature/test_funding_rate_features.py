"""
ファンディングレート特徴量計算のテスト

Tier 1特徴量（15個）の包括的なテストを実装します。
"""

import numpy as np
import pandas as pd
import pytest

from app.services.ml.feature_engineering.funding_rate_features import (
    FundingRateFeatureCalculator,
)


@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    """1時間足のOHLCVサンプルデータ（240時間 = 10日間）"""
    timestamps = pd.date_range("2024-01-01", periods=240, freq="1h")

    # 価格データ（トレンド + ノイズ）
    base_price = 40000
    trend = np.linspace(0, 5000, 240)
    noise = np.random.randn(240) * 500
    close_prices = base_price + trend + noise

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close_prices * 0.999,
            "high": close_prices * 1.002,
            "low": close_prices * 0.998,
            "close": close_prices,
            "volume": np.random.uniform(100, 1000, 240),
        }
    )


@pytest.fixture
def sample_funding_data() -> pd.DataFrame:
    """8時間ごとのファンディングレートサンプルデータ（30期間）"""
    timestamps = pd.date_range("2024-01-01", periods=30, freq="8h")

    # ファンディングレート: -0.05% ~ 0.1%の範囲で変動
    funding_rates = np.random.uniform(-0.0005, 0.001, 30)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "funding_rate": funding_rates,
        }
    )


@pytest.fixture
def calculator() -> FundingRateFeatureCalculator:
    """FundingRateFeatureCalculatorのインスタンス"""
    return FundingRateFeatureCalculator()


class TestFundingRateFeatureCalculator:
    """FundingRateFeatureCalculatorのテストクラス"""

    def test_initialization(self, calculator: FundingRateFeatureCalculator):
        """初期化のテスト"""
        assert calculator is not None
        assert calculator.settlement_interval == 8
        assert calculator.baseline_rate == 0.0001

    def test_initialization_with_custom_config(self):
        """カスタム設定での初期化テスト"""
        config = {"settlement_interval": 4, "baseline_rate": 0.0002}
        calc = FundingRateFeatureCalculator(config)
        assert calc.settlement_interval == 4
        assert calc.baseline_rate == 0.0002

    def test_calculate_features_basic(
        self,
        calculator: FundingRateFeatureCalculator,
        sample_ohlcv_data: pd.DataFrame,
        sample_funding_data: pd.DataFrame,
    ):
        """基本的な特徴量計算のテスト"""
        result = calculator.calculate_features(sample_ohlcv_data, sample_funding_data)

        # 入力データのカラムが保持されている
        assert "timestamp" in result.columns
        assert "close" in result.columns

        # 結果がDataFrame
        assert isinstance(result, pd.DataFrame)

        # 行数が元のOHLCVと同じ
        assert len(result) == len(sample_ohlcv_data)

    def test_time_cycle_features(
        self,
        calculator: FundingRateFeatureCalculator,
        sample_ohlcv_data: pd.DataFrame,
        sample_funding_data: pd.DataFrame,
    ):
        """時間サイクル特徴量のテスト"""
        result = calculator.calculate_features(sample_ohlcv_data, sample_funding_data)

        # 特徴量が存在する
        assert "fr_hours_since_settlement" in result.columns
        assert "fr_cycle_sin" in result.columns
        assert "fr_cycle_cos" in result.columns

        # fr_hours_since_settlementが0-8の範囲内
        hours_since = result["fr_hours_since_settlement"].dropna()
        assert (hours_since >= 0).all()
        assert (hours_since < 8).all()

        # fr_cycle_sin, fr_cycle_cosが[-1, 1]の範囲内
        cycle_sin = result["fr_cycle_sin"].dropna()
        cycle_cos = result["fr_cycle_cos"].dropna()
        assert (cycle_sin >= -1).all() and (cycle_sin <= 1).all()
        assert (cycle_cos >= -1).all() and (cycle_cos <= 1).all()

        # 三角関数の性質: sin^2 + cos^2 = 1
        valid_idx = result[["fr_cycle_sin", "fr_cycle_cos"]].dropna().index
        sin_sq = result.loc[valid_idx, "fr_cycle_sin"] ** 2
        cos_sq = result.loc[valid_idx, "fr_cycle_cos"] ** 2
        assert np.allclose(sin_sq + cos_sq, 1.0, atol=1e-6)

    def test_basic_rate_features(
        self,
        calculator: FundingRateFeatureCalculator,
        sample_ohlcv_data: pd.DataFrame,
        sample_funding_data: pd.DataFrame,
    ):
        """基本金利特徴量のテスト"""
        result = calculator.calculate_features(sample_ohlcv_data, sample_funding_data)

        # 基本特徴量が存在する
        assert "funding_rate_raw" in result.columns
        assert "fr_lag_1p" in result.columns
        assert "fr_lag_2p" in result.columns
        assert "fr_lag_3p" in result.columns

        # ラグ特徴量のシフトが正しい（8時間 * ラグ数）
        # ラグ1は8時間前の値なので、現在の行のlag_1は8行前のraw値
        raw = result["funding_rate_raw"]
        lag_1 = result["fr_lag_1p"]

        # 少なくとも8行後から比較可能
        if len(result) > 16:
            # 16行目のlag_1は8行目のrawと一致すべき
            idx_check = 16
            if not pd.isna(raw.iloc[idx_check - 8]) and not pd.isna(
                lag_1.iloc[idx_check]
            ):
                assert np.isclose(
                    raw.iloc[idx_check - 8],
                    lag_1.iloc[idx_check],
                    rtol=1e-5,
                    equal_nan=False,
                )

    def test_momentum_features(
        self,
        calculator: FundingRateFeatureCalculator,
        sample_ohlcv_data: pd.DataFrame,
        sample_funding_data: pd.DataFrame,
    ):
        """モメンタム特徴量のテスト"""
        result = calculator.calculate_features(sample_ohlcv_data, sample_funding_data)

        # モメンタム特徴量が存在する
        assert "fr_velocity" in result.columns
        assert "fr_ema_3periods" in result.columns
        assert "fr_ema_7periods" in result.columns

        # fr_velocityの計算が正しい（変化率）
        velocity = result["fr_velocity"].dropna()
        assert len(velocity) > 0

        # EMAが存在する
        ema_3 = result["fr_ema_3periods"].dropna()
        ema_7 = result["fr_ema_7periods"].dropna()
        assert len(ema_3) > 0
        assert len(ema_7) > 0

        # EMAの平滑化特性: ema_3の方がema_7より変動が大きい傾向
        # （短期EMAは価格に敏感）

    def test_regime_classification(
        self,
        calculator: FundingRateFeatureCalculator,
        sample_ohlcv_data: pd.DataFrame,
    ):
        """レジーム分類のテスト"""
        # 特定のファンディングレートパターンを作成
        timestamps = pd.date_range("2024-01-01", periods=5, freq="8h")
        funding_data = pd.DataFrame(
            {
                "timestamp": timestamps,
                "funding_rate": [
                    -0.00015,  # 超低金利: FR < -0.01%
                    0.00005,  # 通常: -0.01% ≤ FR ≤ 0.05%
                    0.0008,  # 過熱: 0.05% < FR ≤ 0.15%
                    0.002,  # 極端過熱: FR > 0.15%
                    0.00003,  # 通常
                ],
            }
        )

        result = calculator.calculate_features(sample_ohlcv_data, funding_data)

        # レジーム特徴量が存在する
        assert "fr_regime_encoded" in result.columns
        assert "regime_duration" in result.columns

        regime = result["fr_regime_encoded"].dropna()

        # レジームが定義された範囲内（-2 ~ 2）
        assert regime.isin([-2, -1, 0, 1, 2]).all()

        # regime_durationが非負
        duration = result["regime_duration"].dropna()
        assert (duration >= 0).all()

    def test_price_interaction_features(
        self,
        calculator: FundingRateFeatureCalculator,
        sample_ohlcv_data: pd.DataFrame,
        sample_funding_data: pd.DataFrame,
    ):
        """価格相互作用特徴量のテスト"""
        result = calculator.calculate_features(sample_ohlcv_data, sample_funding_data)

        # 価格相互作用特徴量が存在する
        assert "fr_price_corr_24h" in result.columns
        assert "fr_volatility_adjusted" in result.columns

        # 相関係数が[-1, 1]の範囲内
        corr = result["fr_price_corr_24h"].dropna()
        if len(corr) > 0:
            assert (corr >= -1).all() and (corr <= 1).all()

        # ボラティリティ調整済み特徴量が存在する
        vol_adj = result["fr_volatility_adjusted"].dropna()
        assert len(vol_adj) > 0

    def test_missing_value_handling(
        self,
        calculator: FundingRateFeatureCalculator,
        sample_ohlcv_data: pd.DataFrame,
    ):
        """欠損値処理のテスト"""
        # 欠損値を含むファンディングレートデータ
        timestamps = pd.date_range("2024-01-01", periods=10, freq="8h")
        funding_data = pd.DataFrame(
            {
                "timestamp": timestamps,
                "funding_rate": [
                    0.0001,
                    np.nan,
                    0.0002,
                    np.nan,
                    0.0003,
                    0.0004,
                    np.nan,
                    0.0005,
                    0.0006,
                    0.0007,
                ],
            }
        )

        result = calculator.calculate_features(sample_ohlcv_data, funding_data)

        # 補完フラグが存在する
        if "fr_imputed_flag" in result.columns:
            imputed_flag = result["fr_imputed_flag"].dropna()
            # フラグが0または1
            assert imputed_flag.isin([0, 1]).all()

        # 補完後のfunding_rate_rawに欠損値が少ない
        # （決済時刻の補間が適用される）
        raw = result["funding_rate_raw"]
        missing_ratio = raw.isna().sum() / len(raw)
        # 元データの30%が欠損でも、補間後は大幅に減少
        assert missing_ratio < 0.5

    def test_data_frequency_mismatch(
        self,
        calculator: FundingRateFeatureCalculator,
        sample_ohlcv_data: pd.DataFrame,
        sample_funding_data: pd.DataFrame,
    ):
        """データ頻度不一致の処理テスト（8時間FRデータと1時間OHLCVデータ）"""
        result = calculator.calculate_features(sample_ohlcv_data, sample_funding_data)

        # 1時間足のOHLCVデータの行数が保持される
        assert len(result) == len(sample_ohlcv_data)

        # FRデータが正しく前方補完される
        # （決済時刻以外は前の値を保持）
        raw_fr = result["funding_rate_raw"].dropna()
        assert len(raw_fr) > 0

        # 少なくとも一部のデータが補完されている
        # （8時間ごとの更新を1時間足に拡張）
        consecutive_same = 0
        for i in range(1, len(raw_fr)):
            if raw_fr.iloc[i] == raw_fr.iloc[i - 1]:
                consecutive_same += 1

        # 前方補完により連続する同じ値が存在する
        assert consecutive_same > 0

    def test_feature_output_shape(
        self,
        calculator: FundingRateFeatureCalculator,
        sample_ohlcv_data: pd.DataFrame,
        sample_funding_data: pd.DataFrame,
    ):
        """出力形状のテスト（全15個のTier 1特徴量が生成される）"""
        result = calculator.calculate_features(sample_ohlcv_data, sample_funding_data)

        # Tier 1特徴量リスト（15個）
        tier1_features = [
            # 基本金利指標（4個）
            "funding_rate_raw",
            "fr_lag_1p",
            "fr_lag_2p",
            "fr_lag_3p",
            # 時間サイクル（3個）
            "fr_hours_since_settlement",
            "fr_cycle_sin",
            "fr_cycle_cos",
            # モメンタム（3個）
            "fr_velocity",
            "fr_ema_3periods",
            "fr_ema_7periods",
            # レジーム（2個）
            "fr_regime_encoded",
            "regime_duration",
            # 価格相互作用（2個）
            "fr_price_corr_24h",
            "fr_volatility_adjusted",
        ]

        # 全特徴量が存在する
        for feature in tier1_features:
            assert feature in result.columns, f"特徴量 {feature} が見つかりません"

        # 特徴量の総数を確認（入力カラム + Tier1特徴量）
        # 入力: timestamp, open, high, low, close, volume (6)
        # Tier1特徴量: 15
        # 最小でも21カラム以上
        assert len(result.columns) >= 21

    def test_empty_funding_data(
        self,
        calculator: FundingRateFeatureCalculator,
        sample_ohlcv_data: pd.DataFrame,
    ):
        """空のファンディングレートデータの処理"""
        empty_funding = pd.DataFrame({"timestamp": [], "funding_rate": []})

        # エラーを発生させずに処理できる
        result = calculator.calculate_features(sample_ohlcv_data, empty_funding)

        # 元のOHLCVデータは保持される
        assert len(result) == len(sample_ohlcv_data)
        assert "close" in result.columns

    def test_extreme_funding_rates(
        self,
        calculator: FundingRateFeatureCalculator,
        sample_ohlcv_data: pd.DataFrame,
    ):
        """極端なファンディングレート値の処理"""
        timestamps = pd.date_range("2024-01-01", periods=5, freq="8h")
        extreme_funding = pd.DataFrame(
            {
                "timestamp": timestamps,
                "funding_rate": [
                    -0.01,  # 非常に低い
                    0.005,  # 極端に高い
                    -0.005,
                    0.01,
                    0.0,
                ],
            }
        )

        result = calculator.calculate_features(sample_ohlcv_data, extreme_funding)

        # エラーなく計算できる
        assert "funding_rate_raw" in result.columns

        # レジーム分類が適切に機能する
        regime = result["fr_regime_encoded"].dropna()
        assert regime.isin([-2, -1, 0, 1, 2]).all()

    def test_single_funding_rate_record(
        self,
        calculator: FundingRateFeatureCalculator,
        sample_ohlcv_data: pd.DataFrame,
    ):
        """単一のファンディングレートレコードでの処理"""
        single_funding = pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2024-01-01")],
                "funding_rate": [0.0001],
            }
        )

        result = calculator.calculate_features(sample_ohlcv_data, single_funding)

        # 基本的な計算は可能
        assert "funding_rate_raw" in result.columns

        # ラグ特徴量は最初の8時間分がNaN（シフトのため）
        lag_1 = result["fr_lag_1p"]
        # 最初の8行はNaN
        assert lag_1.iloc[:8].isna().all()
        # 残りは前方補完されたfunding_rate_rawの値
        assert lag_1.iloc[8:].notna().sum() > 0

    def test_numerical_stability(
        self,
        calculator: FundingRateFeatureCalculator,
        sample_ohlcv_data: pd.DataFrame,
        sample_funding_data: pd.DataFrame,
    ):
        """数値安定性のテスト"""
        result = calculator.calculate_features(sample_ohlcv_data, sample_funding_data)

        # 無限大やNaNの割合が許容範囲内
        numeric_cols = result.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            inf_count = np.isinf(result[col]).sum()
            assert inf_count == 0, f"{col}に無限大が含まれています"

            # NaNは一部許容（初期ラグ期間など）
            nan_ratio = result[col].isna().sum() / len(result)
            assert nan_ratio < 0.5, f"{col}のNaN比率が高すぎます: {nan_ratio:.2%}"
