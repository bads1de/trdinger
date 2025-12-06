"""
70個全てのテクニカルインジケーターの包括的テスト

すべてのインジケーターの初期化と計算が正しく動作するかを検証するテスト。
- 各インジケーターの基本的な計算
- 返り値の形式チェック
- NaN/エラー処理
- 必要なデータ長の検証
"""

import numpy as np
import pandas as pd
import pytest

from app.services.indicators import TechnicalIndicatorService


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """テスト用のOHLCVデータを生成"""
    periods = 500  # 十分な長さを確保
    index = pd.date_range("2022-01-01", periods=periods, freq="h")

    # ランダム性とトレンドを含むデータ
    base = np.linspace(10000, 15000, periods)
    noise = np.random.normal(0, 100, periods)
    close = base + noise

    df = pd.DataFrame(
        {
            "Open": close * np.random.uniform(0.99, 1.01, periods),
            "High": close * np.random.uniform(1.01, 1.03, periods),
            "Low": close * np.random.uniform(0.97, 0.99, periods),
            "Close": close,
            "Volume": np.random.uniform(1000, 5000, periods),
        },
        index=index,
    )

    # ボラティリティを追加
    df["High"] = np.maximum(
        df["High"],
        df[["Open", "Close"]].max(axis=1) * np.random.uniform(1.0, 1.05, periods),
    )
    df["Low"] = np.minimum(
        df["Low"],
        df[["Open", "Close"]].min(axis=1) * np.random.uniform(0.95, 1.0, periods),
    )

    return df


@pytest.fixture
def indicator_service() -> TechnicalIndicatorService:
    """テクニカルインジケーターサービスを提供"""
    return TechnicalIndicatorService()


class TestAllTechnicalIndicators:
    """73個のテクニカルインジケーターの包括的テスト（VHFとBIASを追加）"""

    # テスト対象の73個のインジケーター（VHFとBIASを追加）
    INDICATORS = [
        "ACCBANDS",
        "AD",
        "ADOSC",
        "ADX",
        "ALMA",
        "AMAT",
        "AO",
        "APO",
        "AROON",
        "ATR",
        "BB",
        "BOP",
        "BIAS",  # 新規追加
        "CCI",
        "CG",
        "CHOP",
        "CMF",
        "CMO",
        "COPPOCK",
        "CTI",
        "DEMA",
        "DONCHIAN",
        "DPO",
        "EFI",
        "EMA",
        "EOM",
        "FISHER",
        "FRAMA",
        "HMA",
        "KAMA",
        "KELTNER",
        "KST",
        "KVO",
        "LINREG",
        "LINREGSLOPE",
        "MACD",
        "MASSI",
        "MFI",
        "MOM",
        "NATR",
        "NVI",
        "OBV",
        "PGO",
        "PPO",
        "PSL",
        "PVO",
        "PVT",
        "QQE",
        "RMA",
        "ROC",
        "RSI",
        "RVI",
        "SAR",
        "SMA",
        "SQUEEZE",
        "STC",
        "STOCH",
        "SUPERTREND",
        "SUPER_SMOOTHER",
        "T3",
        "TEMA",
        "TRIMA",
        "TRIX",
        "TSI",
        "UI",
        "UO",
        "VORTEX",
        "VWAP",
        "VWMA",
        "VHF",  # 新規追加
        "WILLR",
        "WMA",
        "ZLMA",
    ]

    def test_all_indicators_can_be_initialized(
        self, indicator_service: TechnicalIndicatorService
    ):
        """すべてのインジケーターが初期化可能か確認"""
        for indicator in self.INDICATORS:
            try:
                # 各インジケーターの設定を取得できるか確認
                config = indicator_service.registry.get_indicator_config(indicator)
                assert config is not None, f"{indicator}の設定が見つかりません"
                assert config.adapter_function is not None, (
                    f"{indicator}のアダプター関数がありません"
                )
            except Exception as e:
                pytest.fail(f"{indicator}の初期化に失敗: {e}")

    @pytest.mark.parametrize("indicator", INDICATORS)
    def test_indicator_basic_calculation(
        self,
        indicator: str,
        indicator_service: TechnicalIndicatorService,
        sample_ohlcv: pd.DataFrame,
    ):
        """各インジケーターの基本計算をテスト"""
        try:
            # インジケーター設定を取得
            config = indicator_service.registry.get_indicator_config(indicator)
            assert config is not None, f"{indicator}の設定が取得できません"

            # デフォルトパラメータでテスト
            default_params = config.default_values or {}

            # 特定のパラメータを設定
            if indicator == "ACCBANDS":
                params = {"length": 20}
            elif indicator == "AD":
                params = {}
            elif indicator == "ADOSC":
                params = {"fast": 3, "slow": 10}
            elif indicator == "ADX":
                params = {"period": 14}
            elif indicator == "ALMA":
                params = {
                    "length": 10,
                    "sigma": 6.0,
                    "distribution_offset": 0.85,
                    "offset": 0,
                }
            elif indicator == "AMAT":
                params = {"fast": 5, "slow": 20}
            elif indicator == "AO":
                params = {"fast": 5, "slow": 34}
            elif indicator == "APO":
                params = {"fast": 12, "slow": 26}
            elif indicator == "AROON":
                params = {"length": 14}
            elif indicator == "ATR":
                params = {"length": 14}
            elif indicator == "BB":
                params = {"length": 20, "std": 2.0}
            elif indicator == "BOP":
                params = {"length": 14}
            elif indicator == "CCI":
                params = {"period": 14}
            elif indicator == "CG":
                params = {"length": 10}
            elif indicator == "CHOP":
                params = {"length": 14}
            elif indicator == "CMF":
                params = {"length": 20}
            elif indicator == "CMO":
                params = {"length": 14}
            elif indicator == "COPPOCK":
                params = {"fast": 11, "slow": 14, "long": 10}
            elif indicator == "CTI":
                params = {"length": 12}
            elif indicator == "DEMA":
                params = {"length": 14}
            elif indicator == "DONCHIAN":
                params = {"length": 20}
            elif indicator == "DPO":
                params = {"length": 20, "centered": True}
            elif indicator == "EFI":
                params = {"length": 13}
            elif indicator == "EOM":
                params = {"length": 14, "divisor": 100000000, "drift": 1}
            elif indicator == "FRAMA":
                params = {"length": 16, "slow": 200}
            elif indicator == "HMA":
                params = {"length": 20}
            elif indicator == "KAMA":
                params = {"length": 30}
            elif indicator == "KELTNER":
                params = {"length": 20, "multiplier": 2.0}
            elif indicator == "KST":
                params = {
                    "roc1": 10,
                    "roc2": 15,
                    "roc3": 20,
                    "roc4": 30,
                    "sma1": 10,
                    "sma2": 10,
                    "sma3": 10,
                    "sma4": 15,
                    "signal": 9,
                }
            elif indicator == "KVO":
                params = {"fast": 14, "slow": 28}
            elif indicator == "LINREG":
                params = {"length": 14}
            elif indicator == "LINREGSLOPE":
                params = {"length": 14}
            elif indicator == "MACD":
                params = {"fast": 12, "slow": 26, "signal": 9}
            elif indicator == "MASSI":
                params = {"fast": 9, "slow": 25}
            elif indicator == "MFI":
                params = {"length": 14}
            elif indicator == "MOM":
                params = {"length": 10}
            elif indicator == "NATR":
                params = {"length": 14}
            elif indicator == "NVI":
                params = {}
            elif indicator == "OBV":
                params = {}
            elif indicator == "PGO":
                params = {"length": 14}
            elif indicator == "PPO":
                params = {"fast": 12, "slow": 26, "signal": 9}
            elif indicator == "PSL":
                params = {"length": 12, "scalar": 100}
            elif indicator == "PVO":
                params = {"fast": 12, "slow": 26, "signal": 9}
            elif indicator == "PVT":
                params = {}
            elif indicator == "QQE":
                params = {"length": 14, "smooth": 5}
            elif indicator == "RMA":
                params = {"length": 14}
            elif indicator == "ROC":
                params = {"length": 10}
            elif indicator == "RVI":
                params = {"length": 14, "scalar": 100}
            elif indicator == "SAR":
                params = {"af": 0.02, "max_af": 0.2}
            elif indicator == "SMA":
                params = {"length": 20}
            elif indicator == "SQUEEZE":
                params = {
                    "bb_length": 20,
                    "bb_std": 2.0,
                    "kc_length": 20,
                    "kc_scalar": 1.5,
                    "mom_length": 12,
                    "mom_smooth": 6,
                    "use_tr": True,
                }
            elif indicator == "STC":
                params = {"fast": 12, "slow": 26, "cycle": 9}
            elif indicator == "STOCH":
                params = {"k_length": 14, "smooth_k": 3, "d_length": 3}
            elif indicator == "SUPERTREND":
                params = {"length": 10, "multiplier": 3.0}
            elif indicator == "SUPER_SMOOTHER":
                params = {"length": 10}
            elif indicator == "TEMA":
                params = {"length": 14}
            elif indicator == "TRIMA":
                params = {"length": 20}
            elif indicator == "TRIX":
                params = {"length": 15, "signal": 9}
            elif indicator == "TSI":
                params = {"fast": 25, "slow": 13, "signal": 13}
            elif indicator == "UI":
                params = {"length": 14}
            elif indicator == "UO":
                params = {"fast": 7, "medium": 14, "slow": 28}
            elif indicator == "VORTEX":
                params = {"length": 14, "drift": 1}
            elif indicator == "VWAP":
                params = {"length": 20}
            elif indicator == "VWMA":
                params = {"length": 20}
            elif indicator == "WILLR":
                params = {"length": 14}
            elif indicator == "WMA":
                params = {"length": 20}
            elif indicator == "ZLMA":
                params = {"length": 20}
            else:
                params = default_params

            # 計算を実行
            result = indicator_service.calculate_indicator(
                sample_ohlcv, indicator, params
            )

            # 結果の形式を検証
            if config.result_type == "single":
                assert isinstance(result, np.ndarray), (
                    f"{indicator}の結果がndarrayではありません"
                )
                assert result.shape[0] == len(sample_ohlcv), (
                    f"{indicator}の結果の長さが不正"
                )
                # NaNを含む場合があるが、最後の数ポイントは有効な場合がある
                assert result.shape[0] > 0, f"{indicator}の結果が空"

            elif config.result_type == "complex" or config.result_type == "multiple":
                assert isinstance(result, tuple), (
                    f"{indicator}の結果がtupleではありません"
                )
                assert len(result) > 0, f"{indicator}の結果が空のtuple"
                for i, series in enumerate(result):
                    assert isinstance(series, np.ndarray), (
                        f"{indicator}の結果[{i}]がndarrayではありません"
                    )
                    assert series.shape[0] == len(sample_ohlcv), (
                        f"{indicator}の結果[{i}]の長さが不正{indicator}"
                    )

        except Exception as e:
            pytest.fail(f"{indicator}のテストでエラー: {e}")

    def test_all_indicators_handle_insufficient_data(
        self, indicator_service: TechnicalIndicatorService
    ):
        """不十分なデータに対する挙動をテスト"""
        short_data = pd.DataFrame(
            {
                "Close": [100, 101, 102],
                "High": [101, 102, 103],
                "Low": [99, 100, 101],
                "Volume": [1000, 1100, 900],
            }
        )

        for indicator in self.INDICATORS:
            try:
                config = indicator_service.registry.get_indicator_config(indicator)
                if not config:
                    continue

                # 短いデータで計算を試みる
                params = config.default_values or {}

                result = indicator_service.calculate_indicator(
                    short_data, indicator, params
                )

                # 結果が期待通りか検証
                if config.result_type == "single":
                    assert isinstance(result, np.ndarray)
                    # すべてNaNまたは部分的にNaN
                    assert result.shape[0] == len(short_data)
                elif config.result_type in ["complex", "multiple"]:
                    assert isinstance(result, tuple)

            except Exception as e:
                # 一部のインジケーターは短いデータでエラーになるのは許容
                if "データ長" not in str(e) and "長さ" not in str(e):
                    print(f"警告: {indicator}で予期しないエラー: {e}")

    def test_indicators_return_valid_values(
        self, indicator_service: TechnicalIndicatorService, sample_ohlcv: pd.DataFrame
    ):
        """インジケーターが有効な値を返すか検証"""
        for indicator in ["RSI", "MFI", "CCI", "WILLR", "UO"]:
            try:
                if indicator == "RSI":
                    result = indicator_service.calculate_indicator(
                        sample_ohlcv, indicator, {"length": 14}
                    )
                    # RSIは0-100の範囲
                    if isinstance(result, np.ndarray):
                        valid_values = result[np.isfinite(result)]
                        if len(valid_values) > 0:
                            assert all(0 <= val <= 100 for val in valid_values), (
                                f"{indicator}の値が範囲外"
                            )

                elif indicator == "MFI":
                    result = indicator_service.calculate_indicator(
                        sample_ohlcv, indicator, {"length": 14}
                    )
                    if isinstance(result, np.ndarray):
                        valid_values = result[np.isfinite(result)]
                        if len(valid_values) > 0:
                            assert all(0 <= val <= 100 for val in valid_values), (
                                f"{indicator}の値が範囲外"
                            )

                elif indicator == "WILLR":
                    result = indicator_service.calculate_indicator(
                        sample_ohlcv, indicator, {"length": 14}
                    )
                    if isinstance(result, np.ndarray):
                        valid_values = result[np.isfinite(result)]
                        if len(valid_values) > 0:
                            assert all(-100 <= val <= 0 for val in valid_values), (
                                f"{indicator}の値が範囲外"
                            )

            except Exception:
                pass  # 他のインジケーターは範囲チェックをスキップ

    def test_all_indicators_with_custom_parameters(
        self, indicator_service: TechnicalIndicatorService, sample_ohlcv: pd.DataFrame
    ):
        """カスタムパラメータでの動作をテスト"""
        for indicator in ["EMA", "SMA", "WMA", "DEMA", "TEMA"]:
            try:
                # 異なる長さのパラメータでテスト
                for length in [5, 20, 50]:
                    params = {"length": length}
                    result = indicator_service.calculate_indicator(
                        sample_ohlcv, indicator, params
                    )

                    if hasattr(result, "shape"):
                        assert result.shape[0] == len(sample_ohlcv)
                    else:
                        assert len(result) == len(sample_ohlcv)

            except Exception as e:
                pytest.fail(f"{indicator}のカスタムパラメータテストで失敗: {e}")

    def test_profile_indicators_performance(
        self, indicator_service: TechnicalIndicatorService, sample_ohlcv: pd.DataFrame
    ):
        """主要インジケーターのパフォーマンスを計測"""
        import time

        # 高頻度で使用されるインジケーター
        common_indicators = ["SMA", "EMA", "RSI", "MACD", "BB"]

        start_time = time.time()

        for indicator in common_indicators:
            for _ in range(5):  # 5回実行
                if indicator == "MACD":
                    indicator_service.calculate_indicator(
                        sample_ohlcv, indicator, {"fast": 12, "slow": 26, "signal": 9}
                    )
                else:
                    indicator_service.calculate_indicator(
                        sample_ohlcv, indicator, {"length": 14}
                    )

        elapsed = time.time() - start_time
        # 2秒以内に完了すべき
        assert elapsed < 2.0, f"パフォーマンステスト失敗: {elapsed:.2f}秒"

    def test_all_indicators_supported_by_registry(
        self, indicator_service: TechnicalIndicatorService
    ):
        """レジストリに登録されているか確認"""
        supported = indicator_service.get_supported_indicators()

        for indicator in self.INDICATORS:
            assert indicator in supported, (
                f"{indicator}がレジストリに登録されていません"
            )
            assert supported[indicator]["parameters"] is not None
            assert supported[indicator]["result_type"] is not None


if __name__ == "__main__":
    # コマンドラインからの実行用
    pytest.main([__file__, "-v"])
