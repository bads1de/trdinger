"""
パラメータ生成ユーティリティ

指標のパラメータ生成ロジックを共通化し、重複を削除します。
"""

import random
from typing import Dict, Any


class ParameterGenerator:
    """パラメータ生成器"""
    
    @staticmethod
    def generate_period_parameter(min_period: int = 5, max_period: int = 50) -> Dict[str, Any]:
        """期間パラメータを生成"""
        return {"period": random.randint(min_period, max_period)}
    
    @staticmethod
    def generate_fast_slow_periods(
        fast_min: int = 5, fast_max: int = 20,
        slow_min: int = 20, slow_max: int = 50
    ) -> Dict[str, Any]:
        """高速・低速期間パラメータを生成"""
        fast_period = random.randint(fast_min, fast_max)
        slow_period = random.randint(slow_min, slow_max)
        # 高速期間が低速期間より小さくなるように調整
        if fast_period >= slow_period:
            slow_period = fast_period + random.randint(5, 15)
        return {"fast_period": fast_period, "slow_period": slow_period}
    
    @staticmethod
    def generate_macd_parameters() -> Dict[str, Any]:
        """MACDパラメータを生成"""
        fast_period = random.randint(5, 20)
        slow_period = random.randint(20, 50)
        signal_period = random.randint(5, 15)
        return {
            "fast_period": fast_period,
            "slow_period": slow_period,
            "signal_period": signal_period
        }
    
    @staticmethod
    def generate_bollinger_bands_parameters() -> Dict[str, Any]:
        """ボリンジャーバンドパラメータを生成"""
        return {
            "period": random.randint(15, 25),
            "std_dev": random.uniform(1.5, 2.5)
        }
    
    @staticmethod
    def generate_stochastic_parameters() -> Dict[str, Any]:
        """ストキャスティクスパラメータを生成"""
        return {
            "k_period": random.randint(10, 20),
            "d_period": random.randint(3, 7),
            "slow_k_period": random.randint(3, 5)
        }
    
    @staticmethod
    def generate_stochastic_fast_parameters() -> Dict[str, Any]:
        """ストキャスティクス高速パラメータを生成"""
        return {
            "period": random.randint(5, 14),
            "fastd_period": random.randint(3, 5),
            "fastd_matype": 0  # SMA
        }
    
    @staticmethod
    def generate_stochastic_rsi_parameters() -> Dict[str, Any]:
        """ストキャスティクスRSIパラメータを生成"""
        return {
            "period": random.randint(14, 21),
            "fastk_period": random.randint(3, 5),
            "fastd_period": random.randint(3, 5),
            "fastd_matype": 0  # SMA
        }
    
    @staticmethod
    def generate_t3_parameters() -> Dict[str, Any]:
        """T3パラメータを生成"""
        return {
            "period": random.randint(5, 30),
            "vfactor": random.uniform(0.5, 0.9)
        }
    
    @staticmethod
    def generate_mama_parameters() -> Dict[str, Any]:
        """MAMAパラメータを生成"""
        return {
            "fastlimit": random.uniform(0.4, 0.6),
            "slowlimit": random.uniform(0.02, 0.08)
        }
    
    @staticmethod
    def generate_keltner_parameters() -> Dict[str, Any]:
        """ケルトナーチャネルパラメータを生成"""
        return {
            "period": random.randint(14, 20),
            "multiplier": random.uniform(1.5, 2.5)
        }
    
    @staticmethod
    def generate_ultimate_oscillator_parameters() -> Dict[str, Any]:
        """アルティメットオシレーターパラメータを生成"""
        periods = [7, 14, 28]
        return {
            "period1": random.choice([7, 8, 9]),
            "period2": random.choice([14, 15, 16]),
            "period3": random.choice([28, 29, 30])
        }
    
    @staticmethod
    def generate_apo_parameters() -> Dict[str, Any]:
        """APOパラメータを生成"""
        fast_period = random.randint(12, 20)
        slow_period = random.randint(26, 40)
        return {
            "fast_period": fast_period,
            "slow_period": slow_period,
            "matype": random.choice([0, 1])  # SMA or EMA
        }
    
    @staticmethod
    def generate_ppo_parameters() -> Dict[str, Any]:
        """PPOパラメータを生成"""
        fast_period = random.randint(12, 20)
        slow_period = random.randint(26, 40)
        return {
            "fast_period": fast_period,
            "slow_period": slow_period,
            "matype": random.choice([0, 1])  # SMA or EMA
        }
    
    @staticmethod
    def generate_adosc_parameters() -> Dict[str, Any]:
        """ADOSCパラメータを生成"""
        fast_period = random.randint(3, 7)
        slow_period = random.randint(8, 15)
        return {
            "fast_period": fast_period,
            "slow_period": slow_period
        }


class ThresholdGenerator:
    """閾値生成器"""
    
    @staticmethod
    def generate_percentage_threshold(condition_type: str = "entry") -> float:
        """パーセンテージ閾値を生成（0-100）"""
        if condition_type == "entry":
            return random.uniform(20, 80)
        else:
            return random.uniform(30, 70)
    
    @staticmethod
    def generate_oscillator_threshold(condition_type: str = "entry") -> float:
        """オシレーター閾値を生成（-100 to 100）"""
        if condition_type == "entry":
            return random.uniform(-80, 80)
        else:
            return random.uniform(-60, 60)
    
    @staticmethod
    def generate_price_ratio_threshold(condition_type: str = "entry") -> float:
        """価格比率閾値を生成（0.9-1.1）"""
        if condition_type == "entry":
            return random.uniform(0.95, 1.05)
        else:
            return random.uniform(0.98, 1.02)
    
    @staticmethod
    def generate_momentum_threshold(condition_type: str = "entry") -> float:
        """モメンタム閾値を生成"""
        if condition_type == "entry":
            return random.uniform(-10, 10)
        else:
            return random.uniform(-5, 5)
    
    @staticmethod
    def generate_volume_threshold(condition_type: str = "entry") -> float:
        """出来高閾値を生成"""
        if condition_type == "entry":
            return random.uniform(-1000, 1000)
        else:
            return random.uniform(-500, 500)


# 指標タイプ別のパラメータ生成マッピング
PARAMETER_GENERATORS = {
    # 期間のみのパラメータ
    "period_only": [
        "SMA", "EMA", "WMA", "HMA", "KAMA", "TEMA", "DEMA", "ZLEMA", "TRIMA",
        "RSI", "MOMENTUM", "MOM", "ROC", "ROCP", "ROCR", "CCI", "WILLR",
        "ADX", "AROON", "AROONOSC", "MFI", "CMO", "TRIX", "ATR", "NATR",
        "TRANGE", "STDDEV", "DONCHIAN", "VWMA", "MIDPOINT", "MIDPRICE",
        "DX", "ADXR", "PLUS_DI", "MINUS_DI", "EMV"
    ],
    
    # 特別なパラメータ生成が必要な指標
    "special": {
        "MACD": ParameterGenerator.generate_macd_parameters,
        "BB": ParameterGenerator.generate_bollinger_bands_parameters,
        "STOCH": ParameterGenerator.generate_stochastic_parameters,
        "STOCHF": ParameterGenerator.generate_stochastic_fast_parameters,
        "STOCHRSI": ParameterGenerator.generate_stochastic_rsi_parameters,
        "T3": ParameterGenerator.generate_t3_parameters,
        "MAMA": ParameterGenerator.generate_mama_parameters,
        "KELTNER": ParameterGenerator.generate_keltner_parameters,
        "ULTOSC": ParameterGenerator.generate_ultimate_oscillator_parameters,
        "APO": ParameterGenerator.generate_apo_parameters,
        "PPO": ParameterGenerator.generate_ppo_parameters,
        "ADOSC": ParameterGenerator.generate_adosc_parameters,
    },
    
    # パラメータが不要な指標
    "no_params": ["BOP", "PSAR", "OBV", "AD", "VWAP", "PVT", "AVGPRICE", "MEDPRICE", "TYPPRICE", "WCLPRICE"]
}


def generate_indicator_parameters(indicator_type: str) -> Dict[str, Any]:
    """指標タイプに応じたパラメータを生成"""
    if indicator_type in PARAMETER_GENERATORS["no_params"]:
        return {}
    elif indicator_type in PARAMETER_GENERATORS["special"]:
        return PARAMETER_GENERATORS["special"][indicator_type]()
    elif indicator_type in PARAMETER_GENERATORS["period_only"]:
        return ParameterGenerator.generate_period_parameter()
    else:
        # デフォルトは期間パラメータ
        return ParameterGenerator.generate_period_parameter()
