"""
リスク管理価格計算機

ストップロス・テイクプロフィット価格の計算ロジック
"""

from typing import Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


def calculate_sl_tp_prices(
    entry_price: float,
    sl_value: Union[float, None],
    tp_value: Union[float, None],
    is_long: bool = True,
    use_absolute: bool = False,
) -> Tuple[Optional[float], Optional[float]]:
    """
    ストップロス・テイクプロフィット価格を計算

    Args:
        entry_price: エントリー価格
        sl_value: ストップロス値（パーセンテージまたは絶対価格）
        tp_value: テイクプロフィット値（パーセンテージまたは絶対価格）
        is_long: ロングポジションかどうか
        use_absolute: 絶対価格を使用するかどうか

    Returns:
        Tuple[sl_price, tp_price]: 計算されたSL/TP価格

    Examples:
        # パーセンテージベース（ロング）
        >>> calculate_sl_tp_prices(100.0, 0.02, 0.05, is_long=True)
        (98.0, 105.0)

        # 絶対価格ベース（ロング）
        >>> calculate_sl_tp_prices(100.0, 95.0, 110.0, is_long=True, use_absolute=True)
        (95.0, 110.0)

        # パーセンテージベース（ショート）
        >>> calculate_sl_tp_prices(100.0, 0.02, 0.05, is_long=False)
        (102.0, 95.0)
    """
    sl_price = None
    tp_price = None

    try:
        if sl_value is not None:
            if use_absolute:
                sl_price = float(sl_value)
            else:
                if is_long:
                    sl_price = entry_price * (1 - sl_value)
                else:
                    sl_price = entry_price * (1 + sl_value)

        if tp_value is not None:
            if use_absolute:
                tp_price = float(tp_value)
            else:
                if is_long:
                    tp_price = entry_price * (1 + tp_value)
                else:
                    tp_price = entry_price * (1 - tp_value)

        # 価格の妥当性チェック
        if sl_price is not None and sl_price <= 0:
            logger.warning(f"無効なストップロス価格が検出されました: {sl_price}")
            sl_price = None

        if tp_price is not None and tp_price <= 0:
            logger.warning(f"無効なテイクプロフィット価格が検出されました: {tp_price}")
            tp_price = None

        # ロング/ショートの論理チェック
        if is_long:
            if sl_price is not None and sl_price >= entry_price:
                logger.warning(
                    f"ロングポジションのストップロス価格 {sl_price} はエントリー価格 {entry_price} より低く設定する必要があります。"
                )
            if tp_price is not None and tp_price <= entry_price:
                logger.warning(
                    f"ロングポジションのテイクプロフィット価格 {tp_price} はエントリー価格 {entry_price} より高く設定する必要があります。"
                )
        else:
            if sl_price is not None and sl_price <= entry_price:
                logger.warning(
                    f"ショートポジションのストップロス価格 {sl_price} はエントリー価格 {entry_price} より高く設定する必要があります。"
                )
            if tp_price is not None and tp_price >= entry_price:
                logger.warning(
                    f"ショートポジションのテイクプロフィット価格 {tp_price} はエントリー価格 {entry_price} より低く設定する必要があります。"
                )

    except Exception as e:
        logger.error(
            f"ストップロス/テイクプロフィット価格の計算中にエラーが発生しました: {e}"
        )
        return None, None

    return sl_price, tp_price


class RiskCalculator:
    """
    リスク管理計算クラス

    様々なリスク管理計算を提供する統一インターフェース
    """

    def __init__(self, default_sl_pct: float = 0.02, default_tp_pct: float = 0.05):
        """
        初期化

        Args:
            default_sl_pct: デフォルトストップロス率
            default_tp_pct: デフォルトテイクプロフィット率
        """
        self.default_sl_pct = default_sl_pct
        self.default_tp_pct = default_tp_pct

    def calculate_percentage_based(
        self,
        entry_price: float,
        is_long: bool = True,
        sl_pct: Optional[float] = None,
        tp_pct: Optional[float] = None,
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        パーセンテージベースのSL/TP計算

        Args:
            entry_price: エントリー価格
            is_long: ロングポジションかどうか
            sl_pct: ストップロス率（Noneの場合はデフォルト使用）
            tp_pct: テイクプロフィット率（Noneの場合はデフォルト使用）

        Returns:
            Tuple[sl_price, tp_price]: 計算されたSL/TP価格
        """
        sl_pct = sl_pct or self.default_sl_pct
        tp_pct = tp_pct or self.default_tp_pct

        return calculate_sl_tp_prices(
            entry_price, sl_pct, tp_pct, is_long, use_absolute=False
        )

    def calculate_absolute_based(
        self,
        entry_price: float,
        sl_price: Optional[float] = None,
        tp_price: Optional[float] = None,
        is_long: bool = True,
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        絶対価格ベースのSL/TP計算

        Args:
            entry_price: エントリー価格
            sl_price: ストップロス価格
            tp_price: テイクプロフィット価格
            is_long: ロングポジションかどうか

        Returns:
            Tuple[sl_price, tp_price]: 検証済みのSL/TP価格
        """
        return calculate_sl_tp_prices(
            entry_price, sl_price, tp_price, is_long, use_absolute=True
        )

    def calculate_atr_based(
        self,
        entry_price: float,
        atr_value: float,
        sl_multiplier: float = 2.0,
        tp_multiplier: float = 3.0,
        is_long: bool = True,
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        ATRベースのSL/TP計算

        Args:
            entry_price: エントリー価格
            atr_value: ATR値
            sl_multiplier: ストップロス用のATR倍数
            tp_multiplier: テイクプロフィット用のATR倍数
            is_long: ロングポジションかどうか

        Returns:
            Tuple[sl_price, tp_price]: 計算されたSL/TP価格
        """
        try:
            if atr_value <= 0:
                logger.warning(f"無効なATR値が検出されました: {atr_value}")
                return None, None

            if is_long:
                sl_price = entry_price - (atr_value * sl_multiplier)
                tp_price = entry_price + (atr_value * tp_multiplier)
            else:
                sl_price = entry_price + (atr_value * sl_multiplier)
                tp_price = entry_price - (atr_value * tp_multiplier)

            return sl_price, tp_price

        except Exception as e:
            logger.error(f"ATRベースのSL/TP計算中に予期せぬエラーが発生しました: {e}")
            return None, None

    def calculate_risk_reward_ratio(
        self, entry_price: float, sl_price: float, tp_price: float, is_long: bool = True
    ) -> Optional[float]:
        """
        リスクリワード比率を計算

        Args:
            entry_price: エントリー価格
            sl_price: ストップロス価格
            tp_price: テイクプロフィット価格
            is_long: ロングポジションかどうか

        Returns:
            リスクリワード比率（None if 計算不可）
        """
        try:
            if is_long:
                risk = entry_price - sl_price
                reward = tp_price - entry_price
            else:
                risk = sl_price - entry_price
                reward = entry_price - tp_price

            if risk <= 0:
                logger.warning(f"無効なリスク値が検出されました: {risk}")
                return None

            return reward / risk

        except Exception as e:
            logger.error(
                f"リスクリワード比率の計算中に予期せぬエラーが発生しました: {e}"
            )
            return None

    def calculate_kelly_criterion(
        self, win_rate: float, avg_win: float, avg_loss: float
    ) -> Optional[float]:
        """
        Kelly Criterion（ケリー基準）を計算

        Args:
            win_rate: 勝率（0-1の範囲）
            avg_win: 平均利益
            avg_loss: 平均損失（正の値）

        Returns:
            Kelly比率（None if 計算不可）

        Formula:
            K = (bp - q) / b
            where:
            - b = avg_win / avg_loss (odds)
            - p = win_rate
            - q = 1 - p (loss rate)
        """
        try:
            if not (0 <= win_rate <= 1):
                logger.warning(f"無効な勝率が検出されました: {win_rate}")
                return None

            if avg_win <= 0 or avg_loss <= 0:
                logger.warning(
                    f"無効な平均利益または平均損失が検出されました: {avg_win}, {avg_loss}"
                )
                return None

            # オッズ比（平均利益/平均損失）
            b = avg_win / avg_loss
            p = win_rate
            q = 1 - p

            # Kelly比率の計算
            kelly_ratio = (b * p - q) / b

            # Kelly比率は通常0-1の範囲に制限
            kelly_ratio = max(0, min(kelly_ratio, 1))

            return kelly_ratio

        except Exception as e:
            logger.error(f"Kelly Criterionの計算中に予期せぬエラーが発生しました: {e}")
            return None

    def calculate_optimal_position_size(
        self,
        current_equity: float,
        entry_price: float,
        sl_price: float,
        method: str = "fixed_ratio",
        **kwargs,
    ) -> Optional[float]:
        """
        最適なポジションサイズを計算

        Args:
            current_equity: 現在の資産額
            entry_price: エントリー価格
            sl_price: ストップロス価格
            method: 計算方法
            **kwargs: 追加パラメータ

        Returns:
            ポジションサイズ（None if 計算不可）
        """
        try:
            # 入力値の検証
            if current_equity <= 0 or entry_price <= 0:
                logger.warning(
                    f"無効な資産額またはエントリー価格が検出されました: {current_equity}, {entry_price}"
                )
                return None

            if method == "fixed_ratio":
                ratio = kwargs.get("ratio", 0.02)  # 2%
                if ratio <= 0 or ratio >= 1:
                    logger.warning(f"無効な比率が検出されました: {ratio}")
                    return None
                size = (current_equity * ratio) / entry_price
                return size  # 0-99.99%の範囲に制限を撤廃

            elif method == "fixed_risk":
                risk_amount = kwargs.get("risk_amount", current_equity * 0.01)  # 1%
                if sl_price is None:
                    logger.warning("fixed_riskメソッドにはストップロス価格が必須です。")
                    return None

                risk_per_share = abs(entry_price - sl_price)
                if risk_per_share <= 0:
                    logger.warning(
                        f"無効な1株あたりのリスクが検出されました: {risk_per_share}"
                    )
                    return None

                size = risk_amount / risk_per_share / entry_price  # 株数を比率に変換
                return size  # 0-99.99%の範囲に制限を撤廃

            elif method == "kelly":
                win_rate = kwargs.get("win_rate", 0.5)
                avg_win = kwargs.get("avg_win", 0.05)
                avg_loss = kwargs.get("avg_loss", 0.02)

                kelly_ratio = self.calculate_kelly_criterion(
                    win_rate, avg_win, avg_loss
                )
                if kelly_ratio is None or kelly_ratio <= 0:
                    logger.warning(f"無効なKelly比率が検出されました: {kelly_ratio}")
                    return None

                # Kelly比率を保守的に調整（通常は1/2 Kelly）
                conservative_kelly = kelly_ratio * kwargs.get("kelly_fraction", 0.5)
                size = (current_equity * conservative_kelly) / entry_price
                return size  # 0-99.99%の範囲に制限を撤廃

            elif method == "optimal_f":
                # Ralph Vince's Optimal F
                trade_history = kwargs.get("trade_history", [])
                if len(trade_history) < 10:
                    logger.warning("Optimal F計算には十分な取引履歴がありません。")
                    return None

                optimal_f = self.calculate_optimal_f(trade_history)
                if optimal_f is None or optimal_f <= 0:
                    return None

                size = optimal_f
                return size

            elif method == "volatility_based":
                # ボラティリティベースサイジング
                volatility = kwargs.get("volatility", 0.02)  # デフォルト2%
                base_size = kwargs.get("base_size", 0.02)  # ベースサイズ2%
                volatility_target = kwargs.get(
                    "volatility_target", 0.02
                )  # 目標ボラティリティ2%

                if volatility <= 0:
                    return None

                # ボラティリティ調整
                adjusted_size = base_size * (volatility_target / volatility)
                size = (current_equity * adjusted_size) / entry_price
                return size  # 0-99.99%の範囲に制限を撤廃

            elif method == "percent_volatility":
                # Larry Williams式パーセントボラティリティ
                volatility = kwargs.get("volatility", 0.02)
                risk_percent = kwargs.get("risk_percent", 0.01)  # 1%リスク

                if volatility <= 0:
                    return None

                # ボラティリティ調整済みサイズ
                size = risk_percent / volatility
                return size  # 0-99.99%の範囲に制限を撤廃

            elif method == "martingale":
                # マルチンゲール方式
                consecutive_losses = kwargs.get("consecutive_losses", 0)
                base_size = kwargs.get("base_size", 0.02)
                multiplier = kwargs.get("multiplier", 2.0)
                max_multiplier = kwargs.get("max_multiplier", 8.0)

                # 連敗回数に応じてサイズを増加
                adjusted_multiplier = min(
                    multiplier**consecutive_losses, max_multiplier
                )
                size = (current_equity * base_size * adjusted_multiplier) / entry_price
                return size  # 0-99.99%の範囲に制限を撤廃

            elif method == "anti_martingale":
                # アンチマルチンゲール方式
                consecutive_wins = kwargs.get("consecutive_wins", 0)
                base_size = kwargs.get("base_size", 0.02)
                multiplier = kwargs.get("multiplier", 1.5)
                max_multiplier = kwargs.get("max_multiplier", 4.0)

                # 連勝回数に応じてサイズを増加
                adjusted_multiplier = min(multiplier**consecutive_wins, max_multiplier)
                size = (current_equity * base_size * adjusted_multiplier) / entry_price
                return size  # 0-99.99%の範囲に制限を撤廃

            else:
                logger.warning(
                    f"不明なポジションサイジングメソッドが指定されました: {method}"
                )
                return None

        except Exception as e:
            logger.error(
                f"最適なポジションサイズの計算中に予期せぬエラーが発生しました: {e}"
            )
            return None

    def calculate_optimal_f(self, trade_history: list) -> Optional[float]:
        """
        Ralph Vince's Optimal F計算

        Args:
            trade_history: 取引履歴（利益率のリスト）

        Returns:
            Optimal F値（None if 計算不可）
        """
        try:
            if not trade_history or len(trade_history) < 10:
                return None

            # 最大損失を取得
            max_loss = min(trade_history)
            if max_loss >= 0:
                logger.warning("Optimal F計算には取引履歴に損失が含まれていません。")
                return None

            # 最適Fを計算（簡略版）
            # 実際のOptimal Fは複雑な最適化が必要だが、ここでは近似値を使用
            losses = [trade for trade in trade_history if trade < 0]
            wins = [trade for trade in trade_history if trade > 0]

            if not losses or not wins:
                return None

            avg_loss = abs(sum(losses) / len(losses))
            avg_win = sum(wins) / len(wins)
            win_rate = len(wins) / len(trade_history)

            # Kelly基準をベースとした近似Optimal F
            if avg_loss <= 0:
                return None

            b = avg_win / avg_loss
            p = win_rate
            q = 1 - p

            # Optimal Fの計算
            optimal_f = (b * p - q) / b

            # Optimal Fは通常Kelly基準より保守的
            conservative_f = optimal_f * 0.5  # 1/2 ハーフーケリー

            return max(0, min(conservative_f, 0.5))  # 最大50%に制限

        except Exception as e:
            logger.error(f"Optimal Fの計算中に予期せぬエラーが発生しました: {e}")
            return None
