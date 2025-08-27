from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class ExecutionContext:
    current_price: float
    current_equity: float
    available_cash: float


class OrderExecutionPolicy:
    """
    エントリー時のサイズ調整、買付可能性チェック、TP/SL適用を一元化するポリシー。
    StrategyFactory.next() から呼び出される。
    """

    @staticmethod
    def adjust_position_size_for_backtesting(size: float) -> float:
        if size == 0:
            return 0.0

        abs_size = abs(size)
        sign = 1.0 if size > 0 else -1.0

        # 小数点サイズも許可（暗号通貨では一般的）
        if abs_size < 0.001:  # 最小取引サイズ
            return 0.0

        # サイズをそのまま返す（整数に丸めない）
        return size

    @staticmethod
    def ensure_affordable_size(adjusted_size: float, ctx: ExecutionContext) -> float:
        abs_size = abs(adjusted_size)
        if abs_size == 0:
            return 0.0

        # より安全なポジションサイズ計算
        # 利用可能資金の最大80%まで使用（証拠金不足を防ぐため）
        max_cash_usage = ctx.available_cash * 0.8

        if abs_size < 1:
            # 相対サイズ（資金の割合）
            required_cash = ctx.available_cash * abs_size
            if required_cash > max_cash_usage:
                # 最大使用可能な割合に調整
                safe_ratio = max_cash_usage / ctx.available_cash
                return (adjusted_size / abs_size) * safe_ratio
        else:
            # 絶対サイズ（単位数）
            required_cash = abs_size * ctx.current_price
            if required_cash > max_cash_usage:
                # 購入可能な最大単位数に調整
                max_units = max_cash_usage / ctx.current_price
                if max_units < 0.001:  # 最小取引サイズ
                    return 0.0  # 取引不可
                return (1.0 if adjusted_size > 0 else -1.0) * max_units

        return adjusted_size

    @staticmethod
    def compute_tpsl_prices(
        factory, current_price: float, risk_management, gene, position_direction: float
    ) -> Tuple[Optional[float], Optional[float]]:
        stop_loss_pct = risk_management.get("stop_loss")
        take_profit_pct = risk_management.get("take_profit")
        sl_price, tp_price = factory.tpsl_service.calculate_tpsl_prices(
            current_price=current_price,
            tpsl_gene=gene,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            risk_management=risk_management,
            position_direction=position_direction,
        )
        return sl_price, tp_price

    @staticmethod
    def compute_final_position_size(
        factory,
        gene,
        current_price: float,
        current_equity: float,
        available_cash: float,
        data,
        raw_size: float,
    ) -> float:
        """ファクトリーのサイズ算出を受けて、bt制約と買付可能性まで一括反映して最終サイズを返す。"""
        # backtestingの制約調整
        adjusted_size = OrderExecutionPolicy.adjust_position_size_for_backtesting(
            raw_size
        )
        # 購入可能性チェック
        ctx = ExecutionContext(
            current_price=current_price,
            current_equity=current_equity,
            available_cash=available_cash,
        )
        final = OrderExecutionPolicy.ensure_affordable_size(adjusted_size, ctx)
        return final
