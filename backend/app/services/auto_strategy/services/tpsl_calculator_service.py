"""
統一TP/SL計算サービス

TP/SL計算ロジックを一元化し、異なる計算方式を統一的なインターフェースで提供します。
"""

import logging
from typing import Any, Dict, Optional, Tuple

from ..calculators.risk_reward_calculator import RiskRewardCalculator, RiskRewardConfig
from ..calculators.tpsl_calculator import TPSLCalculator
from ..generators.statistical_tpsl_generator import StatisticalTPSLGenerator, StatisticalConfig
from ..generators.volatility_based_generator import VolatilityBasedGenerator, VolatilityConfig
from ..models.gene_tpsl import TPSLGene, TPSLMethod

logger = logging.getLogger(__name__)


class TPSLCalculatorService:
    """
    統一TP/SL計算サービス
    
    異なるTP/SL計算方式を統一的なインターフェースで提供し、
    計算ロジックの一元化を実現します。
    """

    def __init__(self):
        """サービスを初期化"""
        self.tpsl_calculator = TPSLCalculator()
        self.risk_reward_calculator = RiskRewardCalculator()
        self.statistical_generator = StatisticalTPSLGenerator()
        self.volatility_generator = VolatilityBasedGenerator()

    def calculate_tpsl_prices(
        self,
        current_price: float,
        tpsl_gene: Optional[TPSLGene] = None,
        stop_loss_pct: Optional[float] = None,
        take_profit_pct: Optional[float] = None,
        risk_management: Optional[Dict[str, Any]] = None,
        market_data: Optional[Dict[str, Any]] = None,
        position_direction: float = 1.0,
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        統一的なTP/SL価格計算
        
        Args:
            current_price: 現在価格
            tpsl_gene: TP/SL遺伝子（GA最適化対象）
            stop_loss_pct: ストップロス割合（従来方式）
            take_profit_pct: テイクプロフィット割合（従来方式）
            risk_management: リスク管理設定
            market_data: 市場データ
            position_direction: ポジション方向（1.0=ロング, -1.0=ショート）
            
        Returns:
            (SL価格, TP価格)のタプル
        """
        try:
            # TP/SL遺伝子が利用可能な場合（GA最適化対象）
            if tpsl_gene and tpsl_gene.enabled:
                return self._calculate_from_gene(
                    current_price, tpsl_gene, market_data, position_direction
                )
            
            # 従来方式の場合
            return self.tpsl_calculator.calculate_tpsl_prices(
                current_price=current_price,
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
                risk_management=risk_management or {},
                position_direction=position_direction,
            )
            
        except Exception as e:
            logger.error(f"TP/SL価格計算エラー: {e}")
            # フォールバック: 基本計算
            return self._calculate_fallback(current_price, position_direction)

    def _calculate_from_gene(
        self,
        current_price: float,
        tpsl_gene: TPSLGene,
        market_data: Optional[Dict[str, Any]],
        position_direction: float,
    ) -> Tuple[Optional[float], Optional[float]]:
        """TP/SL遺伝子からTP/SL価格を計算"""
        try:
            if tpsl_gene.method == TPSLMethod.FIXED_PERCENTAGE:
                return self._calculate_fixed_percentage(
                    current_price, tpsl_gene, position_direction
                )
            
            elif tpsl_gene.method == TPSLMethod.RISK_REWARD_RATIO:
                return self._calculate_risk_reward_ratio(
                    current_price, tpsl_gene, position_direction
                )
            
            elif tpsl_gene.method == TPSLMethod.VOLATILITY_BASED:
                return self._calculate_volatility_based(
                    current_price, tpsl_gene, market_data, position_direction
                )
            
            elif tpsl_gene.method == TPSLMethod.STATISTICAL:
                return self._calculate_statistical(
                    current_price, tpsl_gene, market_data, position_direction
                )
            
            elif tpsl_gene.method == TPSLMethod.ADAPTIVE:
                return self._calculate_adaptive(
                    current_price, tpsl_gene, market_data, position_direction
                )
            
            else:
                # 未知の方式の場合はフォールバック
                logger.warning(f"未知のTP/SL方式: {tpsl_gene.method}")
                return self._calculate_fallback(current_price, position_direction)
                
        except Exception as e:
            logger.error(f"遺伝子ベースTP/SL計算エラー: {e}")
            return self._calculate_fallback(current_price, position_direction)

    def _calculate_fixed_percentage(
        self,
        current_price: float,
        tpsl_gene: TPSLGene,
        position_direction: float,
    ) -> Tuple[Optional[float], Optional[float]]:
        """固定パーセンテージ方式"""
        if position_direction > 0:  # ロング
            sl_price = current_price * (1 - tpsl_gene.stop_loss_pct)
            tp_price = current_price * (1 + tpsl_gene.take_profit_pct)
        else:  # ショート
            sl_price = current_price * (1 + tpsl_gene.stop_loss_pct)
            tp_price = current_price * (1 - tpsl_gene.take_profit_pct)
        
        return sl_price, tp_price

    def _calculate_risk_reward_ratio(
        self,
        current_price: float,
        tpsl_gene: TPSLGene,
        position_direction: float,
    ) -> Tuple[Optional[float], Optional[float]]:
        """リスクリワード比方式"""
        try:
            config = RiskRewardConfig(target_ratio=tpsl_gene.risk_reward_ratio)
            result = self.risk_reward_calculator.calculate_take_profit(
                tpsl_gene.base_stop_loss, config
            )
            
            if position_direction > 0:  # ロング
                sl_price = current_price * (1 - tpsl_gene.base_stop_loss)
                tp_price = current_price * (1 + result.take_profit_pct)
            else:  # ショート
                sl_price = current_price * (1 + tpsl_gene.base_stop_loss)
                tp_price = current_price * (1 - result.take_profit_pct)
            
            return sl_price, tp_price
            
        except Exception as e:
            logger.error(f"リスクリワード比計算エラー: {e}")
            return self._calculate_fixed_percentage(current_price, tpsl_gene, position_direction)

    def _calculate_volatility_based(
        self,
        current_price: float,
        tpsl_gene: TPSLGene,
        market_data: Optional[Dict[str, Any]],
        position_direction: float,
    ) -> Tuple[Optional[float], Optional[float]]:
        """ボラティリティベース方式"""
        try:
            config = VolatilityConfig(
                atr_period=tpsl_gene.atr_period,
                atr_multiplier_sl=tpsl_gene.atr_multiplier_sl,
                atr_multiplier_tp=tpsl_gene.atr_multiplier_tp,
            )
            
            result = self.volatility_generator.generate_volatility_based_tpsl(
                market_data or {}, config, current_price
            )
            
            if position_direction > 0:  # ロング
                sl_price = current_price * (1 - result.stop_loss_pct)
                tp_price = current_price * (1 + result.take_profit_pct)
            else:  # ショート
                sl_price = current_price * (1 + result.stop_loss_pct)
                tp_price = current_price * (1 - result.take_profit_pct)
            
            return sl_price, tp_price
            
        except Exception as e:
            logger.error(f"ボラティリティベース計算エラー: {e}")
            return self._calculate_fixed_percentage(current_price, tpsl_gene, position_direction)

    def _calculate_statistical(
        self,
        current_price: float,
        tpsl_gene: TPSLGene,
        market_data: Optional[Dict[str, Any]],
        position_direction: float,
    ) -> Tuple[Optional[float], Optional[float]]:
        """統計的方式"""
        try:
            config = StatisticalConfig(
                lookback_period_days=tpsl_gene.lookback_period,
                confidence_threshold=tpsl_gene.confidence_threshold,
            )
            
            result = self.statistical_generator.generate_statistical_tpsl(
                config, market_conditions=market_data
            )
            
            if position_direction > 0:  # ロング
                sl_price = current_price * (1 - result.stop_loss_pct)
                tp_price = current_price * (1 + result.take_profit_pct)
            else:  # ショート
                sl_price = current_price * (1 + result.stop_loss_pct)
                tp_price = current_price * (1 - result.take_profit_pct)
            
            return sl_price, tp_price
            
        except Exception as e:
            logger.error(f"統計的計算エラー: {e}")
            return self._calculate_fixed_percentage(current_price, tpsl_gene, position_direction)

    def _calculate_adaptive(
        self,
        current_price: float,
        tpsl_gene: TPSLGene,
        market_data: Optional[Dict[str, Any]],
        position_direction: float,
    ) -> Tuple[Optional[float], Optional[float]]:
        """適応的方式（複数方式の組み合わせ）"""
        try:
            # 複数の方式を組み合わせて最適な値を選択
            # 現在は簡易実装として、ボラティリティベースを使用
            return self._calculate_volatility_based(
                current_price, tpsl_gene, market_data, position_direction
            )
            
        except Exception as e:
            logger.error(f"適応的計算エラー: {e}")
            return self._calculate_fixed_percentage(current_price, tpsl_gene, position_direction)

    def _calculate_fallback(
        self,
        current_price: float,
        position_direction: float,
    ) -> Tuple[Optional[float], Optional[float]]:
        """フォールバック計算（デフォルト値）"""
        default_sl_pct = 0.03  # 3%
        default_tp_pct = 0.06  # 6%
        
        if position_direction > 0:  # ロング
            sl_price = current_price * (1 - default_sl_pct)
            tp_price = current_price * (1 + default_tp_pct)
        else:  # ショート
            sl_price = current_price * (1 + default_sl_pct)
            tp_price = current_price * (1 - default_tp_pct)
        
        return sl_price, tp_price
