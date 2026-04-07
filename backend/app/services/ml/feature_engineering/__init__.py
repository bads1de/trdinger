"""
Feature Engineering パッケージ

生市場データから機械学習モデル用の特徴量を生成する計算機群を提供します。
価格、テクニカル指標、出来高、ファンディングレート、オープンインタレストなど
多様な特徴量カテゴリをカバーします。

主なコンポーネント:
- feature_engineering_service.py: 特徴量生成の統合サービス
- price_features.py: 価格ベース特徴量
- technical_features.py: テクニカル指標特徴量
- volume_profile_features.py: 出来高プロファイル特徴量
- funding_rate_features.py: ファンディングレート特徴量
- oi_fr_interaction_features.py: OI-FR相互作用特徴量
- crypto_features.py: 暗号通貨固有特徴量
- microstructure_features.py: 市場マイクロストラクチャー特徴量
- multi_timeframe_features.py: 複数時間軸特徴量
- advanced_rolling_stats.py: 高度な移動統計
- volatility_estimators.py: 共通ボラティリティ推定量
- complexity_features.py: 複雑度特徴量
- time_anomaly_features.py: 時間異常特徴量
- data_frequency_manager.py: データ頻度管理
"""

from .advanced_rolling_stats import AdvancedRollingStatsCalculator
from .base_feature_calculator import BaseFeatureCalculator
from .complexity_features import ComplexityFeatureCalculator
from .crypto_features import CryptoFeatureCalculator
from .data_frequency_manager import DataFrequencyManager
from .feature_engineering_service import FeatureEngineeringService
from .funding_rate_features import FundingRateFeatureCalculator
from .market_data_features import MarketDataFeatureCalculator
from .microstructure_features import MicrostructureFeatureCalculator
from .multi_timeframe_features import MultiTimeframeFeatureCalculator
from .oi_fr_interaction_features import OIFRInteractionFeatureCalculator
from .price_features import PriceFeatureCalculator
from .technical_features import TechnicalFeatureCalculator
from .time_anomaly_features import TimeAnomalyFeatures
from .volume_profile_features import VolumeProfileFeatureCalculator

__all__ = [
    "AdvancedRollingStatsCalculator",
    "BaseFeatureCalculator",
    "ComplexityFeatureCalculator",
    "CryptoFeatureCalculator",
    "DataFrequencyManager",
    "FeatureEngineeringService",
    "FundingRateFeatureCalculator",
    "MarketDataFeatureCalculator",
    "MicrostructureFeatureCalculator",
    "MultiTimeframeFeatureCalculator",
    "OIFRInteractionFeatureCalculator",
    "PriceFeatureCalculator",
    "TechnicalFeatureCalculator",
    "TimeAnomalyFeatures",
    "VolumeProfileFeatureCalculator",
]
