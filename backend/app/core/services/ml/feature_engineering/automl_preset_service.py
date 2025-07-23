"""
AutoML設定プリセットサービス

異なる市場条件や戦略に応じた最適化されたAutoML設定プリセットを提供します。
"""

import logging
from typing import Dict, Any, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class MarketCondition(Enum):
    """市場状況"""
    BULL_MARKET = "bull_market"  # 強気市場
    BEAR_MARKET = "bear_market"  # 弱気市場
    SIDEWAYS = "sideways"  # 横ばい市場
    HIGH_VOLATILITY = "high_volatility"  # 高ボラティリティ
    LOW_VOLATILITY = "low_volatility"  # 低ボラティリティ


class TradingStrategy(Enum):
    """取引戦略"""
    SCALPING = "scalping"  # スキャルピング
    DAY_TRADING = "day_trading"  # デイトレード
    SWING_TRADING = "swing_trading"  # スイングトレード
    POSITION_TRADING = "position_trading"  # ポジショントレード
    ARBITRAGE = "arbitrage"  # アービトラージ


class DataSize(Enum):
    """データサイズ"""
    SMALL = "small"  # 小規模（< 1000サンプル）
    MEDIUM = "medium"  # 中規模（1000-10000サンプル）
    LARGE = "large"  # 大規模（> 10000サンプル）


@dataclass
class AutoMLPreset:
    """AutoML設定プリセット"""
    name: str
    description: str
    market_condition: MarketCondition
    trading_strategy: TradingStrategy
    data_size: DataSize
    config: Dict[str, Any]
    performance_notes: str


class AutoMLPresetService:
    """
    AutoML設定プリセットサービス
    
    市場条件、取引戦略、データサイズに応じた最適化されたAutoML設定を提供します。
    """
    
    def __init__(self):
        """初期化"""
        self.presets = self._initialize_presets()
    
    def _initialize_presets(self) -> List[AutoMLPreset]:
        """プリセットを初期化"""
        return [
            # スキャルピング戦略用プリセット
            AutoMLPreset(
                name="scalping_high_freq",
                description="スキャルピング用高頻度特徴量設定",
                market_condition=MarketCondition.HIGH_VOLATILITY,
                trading_strategy=TradingStrategy.SCALPING,
                data_size=DataSize.LARGE,
                config={
                    "tsfresh": {
                        "enabled": True,
                        "feature_selection": True,
                        "fdr_level": 0.001,  # 非常に厳しい選択
                        "feature_count_limit": 50,  # 高速処理のため制限
                        "parallel_jobs": 4,
                        "performance_mode": "fast",
                    },
                    "featuretools": {
                        "enabled": True,
                        "max_depth": 1,  # 浅い深度で高速化
                        "max_features": 20,
                    },
                    "autofeat": {
                        "enabled": False,  # 計算コストが高いため無効
                        "max_features": 0,
                        "feateng_steps": 0,
                        "max_gb": 0.5,
                    },
                },
                performance_notes="高頻度取引に最適化。計算速度を重視し、最も重要な特徴量のみを選択。"
            ),
            
            # デイトレード戦略用プリセット
            AutoMLPreset(
                name="day_trading_balanced",
                description="デイトレード用バランス設定",
                market_condition=MarketCondition.BULL_MARKET,
                trading_strategy=TradingStrategy.DAY_TRADING,
                data_size=DataSize.MEDIUM,
                config={
                    "tsfresh": {
                        "enabled": True,
                        "feature_selection": True,
                        "fdr_level": 0.01,
                        "feature_count_limit": 100,
                        "parallel_jobs": 3,
                        "performance_mode": "balanced",
                    },
                    "featuretools": {
                        "enabled": True,
                        "max_depth": 2,
                        "max_features": 50,
                    },
                    "autofeat": {
                        "enabled": True,
                        "max_features": 30,
                        "feateng_steps": 5,
                        "max_gb": 1.0,
                    },
                },
                performance_notes="デイトレードに適したバランス設定。精度と計算時間のトレードオフを考慮。"
            ),
            
            # スイングトレード戦略用プリセット
            AutoMLPreset(
                name="swing_trading_comprehensive",
                description="スイングトレード用包括的設定",
                market_condition=MarketCondition.SIDEWAYS,
                trading_strategy=TradingStrategy.SWING_TRADING,
                data_size=DataSize.LARGE,
                config={
                    "tsfresh": {
                        "enabled": True,
                        "feature_selection": True,
                        "fdr_level": 0.05,
                        "feature_count_limit": 200,
                        "parallel_jobs": 4,
                        "performance_mode": "comprehensive",
                    },
                    "featuretools": {
                        "enabled": True,
                        "max_depth": 3,
                        "max_features": 100,
                    },
                    "autofeat": {
                        "enabled": True,
                        "max_features": 50,
                        "feateng_steps": 10,
                        "max_gb": 2.0,
                    },
                },
                performance_notes="スイングトレードに最適化。多様な特徴量で中長期トレンドを捉える。"
            ),
            
            # 高ボラティリティ市場用プリセット
            AutoMLPreset(
                name="high_volatility_robust",
                description="高ボラティリティ市場用ロバスト設定",
                market_condition=MarketCondition.HIGH_VOLATILITY,
                trading_strategy=TradingStrategy.DAY_TRADING,
                data_size=DataSize.MEDIUM,
                config={
                    "tsfresh": {
                        "enabled": True,
                        "feature_selection": True,
                        "fdr_level": 0.001,  # ノイズ除去を強化
                        "feature_count_limit": 80,
                        "parallel_jobs": 3,
                        "performance_mode": "robust",
                    },
                    "featuretools": {
                        "enabled": True,
                        "max_depth": 2,
                        "max_features": 40,
                    },
                    "autofeat": {
                        "enabled": True,
                        "max_features": 25,
                        "feateng_steps": 8,
                        "max_gb": 1.5,
                    },
                },
                performance_notes="高ボラティリティ環境でのノイズ耐性を重視。ロバストな特徴量選択。"
            ),
            
            # 低ボラティリティ市場用プリセット
            AutoMLPreset(
                name="low_volatility_sensitive",
                description="低ボラティリティ市場用高感度設定",
                market_condition=MarketCondition.LOW_VOLATILITY,
                trading_strategy=TradingStrategy.SWING_TRADING,
                data_size=DataSize.LARGE,
                config={
                    "tsfresh": {
                        "enabled": True,
                        "feature_selection": True,
                        "fdr_level": 0.1,  # より多くの特徴量を許可
                        "feature_count_limit": 300,
                        "parallel_jobs": 4,
                        "performance_mode": "sensitive",
                    },
                    "featuretools": {
                        "enabled": True,
                        "max_depth": 4,  # より深い特徴量合成
                        "max_features": 150,
                    },
                    "autofeat": {
                        "enabled": True,
                        "max_features": 80,
                        "feateng_steps": 15,
                        "max_gb": 3.0,
                    },
                },
                performance_notes="低ボラティリティ環境での微細な変化を捉える高感度設定。"
            ),
            
            # 小規模データ用プリセット
            AutoMLPreset(
                name="small_data_efficient",
                description="小規模データ用効率的設定",
                market_condition=MarketCondition.SIDEWAYS,
                trading_strategy=TradingStrategy.DAY_TRADING,
                data_size=DataSize.SMALL,
                config={
                    "tsfresh": {
                        "enabled": True,
                        "feature_selection": False,  # 過学習防止
                        "fdr_level": 0.05,
                        "feature_count_limit": 30,
                        "parallel_jobs": 2,
                        "performance_mode": "conservative",
                    },
                    "featuretools": {
                        "enabled": True,
                        "max_depth": 1,
                        "max_features": 15,
                    },
                    "autofeat": {
                        "enabled": False,  # 小規模データでは無効
                        "max_features": 0,
                        "feateng_steps": 0,
                        "max_gb": 0.5,
                    },
                },
                performance_notes="小規模データでの過学習を防ぐ保守的設定。シンプルな特徴量に焦点。"
            ),
            
            # アービトラージ戦略用プリセット
            AutoMLPreset(
                name="arbitrage_precision",
                description="アービトラージ用高精度設定",
                market_condition=MarketCondition.LOW_VOLATILITY,
                trading_strategy=TradingStrategy.ARBITRAGE,
                data_size=DataSize.LARGE,
                config={
                    "tsfresh": {
                        "enabled": True,
                        "feature_selection": True,
                        "fdr_level": 0.001,  # 極めて厳しい選択
                        "feature_count_limit": 150,
                        "parallel_jobs": 4,
                        "performance_mode": "precision",
                    },
                    "featuretools": {
                        "enabled": True,
                        "max_depth": 2,
                        "max_features": 75,
                    },
                    "autofeat": {
                        "enabled": True,
                        "max_features": 40,
                        "feateng_steps": 12,
                        "max_gb": 2.5,
                    },
                },
                performance_notes="アービトラージ機会の精密な検出に最適化。高精度な特徴量選択。"
            ),
        ]
    
    def get_preset_by_name(self, name: str) -> AutoMLPreset:
        """名前でプリセットを取得"""
        for preset in self.presets:
            if preset.name == name:
                return preset
        raise ValueError(f"プリセット '{name}' が見つかりません")
    
    def get_presets_by_strategy(self, strategy: TradingStrategy) -> List[AutoMLPreset]:
        """取引戦略でプリセットをフィルタ"""
        return [p for p in self.presets if p.trading_strategy == strategy]
    
    def get_presets_by_market_condition(self, condition: MarketCondition) -> List[AutoMLPreset]:
        """市場条件でプリセットをフィルタ"""
        return [p for p in self.presets if p.market_condition == condition]
    
    def get_presets_by_data_size(self, size: DataSize) -> List[AutoMLPreset]:
        """データサイズでプリセットをフィルタ"""
        return [p for p in self.presets if p.data_size == size]
    
    def recommend_preset(
        self, 
        market_condition: MarketCondition = None,
        trading_strategy: TradingStrategy = None,
        data_size: DataSize = None
    ) -> AutoMLPreset:
        """条件に基づいてプリセットを推奨"""
        candidates = self.presets.copy()
        
        # 条件でフィルタリング
        if market_condition:
            candidates = [p for p in candidates if p.market_condition == market_condition]
        
        if trading_strategy:
            candidates = [p for p in candidates if p.trading_strategy == trading_strategy]
        
        if data_size:
            candidates = [p for p in candidates if p.data_size == data_size]
        
        if not candidates:
            # 条件に合うプリセットがない場合はデフォルトを返す
            return self.get_preset_by_name("day_trading_balanced")
        
        # 最初の候補を返す（将来的にはスコアリング機能を追加可能）
        return candidates[0]
    
    def get_all_presets(self) -> List[AutoMLPreset]:
        """全プリセットを取得"""
        return self.presets.copy()
    
    def get_preset_summary(self) -> Dict[str, Any]:
        """プリセットサマリーを取得"""
        return {
            "total_presets": len(self.presets),
            "strategies": list(set(p.trading_strategy.value for p in self.presets)),
            "market_conditions": list(set(p.market_condition.value for p in self.presets)),
            "data_sizes": list(set(p.data_size.value for p in self.presets)),
            "preset_names": [p.name for p in self.presets]
        }
