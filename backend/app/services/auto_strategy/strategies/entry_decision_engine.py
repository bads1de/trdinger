"""
エントリー判定と発注実行を担当するモジュール。

UniversalStrategy.next() に集中していた新規エントリーの責務を分離する。
"""

from __future__ import annotations

import logging
import math
from typing import Any, cast

from app.services.backtest.shared import FRACTIONAL_UNIT

from ..config.constants import EntryType
from ..genes import Condition, ConditionGroup, TPSLGene, TPSLMethod

logger = logging.getLogger(__name__)


class EntryDecisionEngine:
    """エントリー方向の決定と注文実行を担当するクラス。"""

    def __init__(self, strategy: Any):
        self.strategy = strategy

    def determine_entry_direction(self) -> float:
        """
        現在バーでエントリーすべき方向を返す。

        優先順位:
        1. 通常ロング
        2. 通常ショート
        3. ステートフル条件
        """
        if self.tools_block_entry():
            return 0.0

        if self.check_entry_conditions(1.0):
            return 1.0
        if self.check_entry_conditions(-1.0):
            return -1.0

        stateful_dir = (
            self.strategy.stateful_conditions_evaluator.get_stateful_entry_direction()
        )
        return 0.0 if stateful_dir is None else stateful_dir

    def execute_entry(self, direction: float) -> bool:
        """
        指定方向での新規エントリーを実行する。

        Returns:
            実際に注文実行または保留注文作成まで進んだ場合 True
        """
        if direction == 0.0:
            return False

        current_price = self.strategy.data.Close[-1]
        sl_price, tp_price = self.calculate_effective_tpsl_prices(
            direction, current_price
        )

        entry_gene = self.strategy._get_effective_entry_gene(direction)
        entry_params = self.strategy.entry_executor.calculate_entry_params(
            entry_gene, current_price, direction
        )
        position_size = self.calculate_position_size()

        is_market = (
            entry_gene is None
            or not entry_gene.enabled
            or entry_gene.entry_type == EntryType.MARKET
        )

        if is_market:
            if direction > 0:
                self.strategy.buy(size=position_size)
            else:
                self.strategy.sell(size=position_size)

            self.strategy.runtime_state.set_open_position(
                entry_price=current_price,
                sl_price=sl_price,
                tp_price=tp_price,
                direction=direction,
            )
            return True

        self.strategy.order_manager.create_pending_order(
            direction=direction,
            size=position_size,
            entry_params=entry_params,
            sl_price=sl_price,
            tp_price=tp_price,
            entry_gene=entry_gene,
            current_bar_index=self.strategy._current_bar_index,
        )
        return True

    def check_entry_conditions(self, direction: float) -> bool:
        """指定された方向のエントリー条件をチェックする。"""
        cached_signal = self._get_cached_entry_signal(direction)
        if cached_signal is not None:
            return bool(cached_signal)

        field_name = (
            "long_entry_conditions" if direction > 0 else "short_entry_conditions"
        )
        conditions = cast(
            list[Condition | ConditionGroup],
            getattr(self.strategy.gene, field_name, []),
        )
        if not conditions:
            return False

        return cast(
            bool,
            self.strategy.condition_evaluator.evaluate_conditions(
                conditions,
                self.strategy,
            ),
        )

    def _get_cached_entry_signal(self, direction: float) -> Any:
        """キャッシュされたエントリーシグナルを安全に取得する。"""
        cached = getattr(self.strategy, "_precomputed_signals", None)
        if not isinstance(cached, dict):
            return None

        signals = cached.get(direction)
        if signals is None:
            return None

        try:
            signal_len = len(signals)
        except TypeError:
            logger.debug(
                "スカラー結果のためキャッシュ済みエントリーシグナルを使いません: direction=%s, type=%s",
                direction,
                type(signals).__name__,
            )
            return None

        idx = len(self.strategy.data) - 1
        if not 0 <= idx < signal_len:
            return None

        try:
            if hasattr(signals, "iloc"):
                return signals.iloc[idx]
            return signals[idx]
        except Exception as e:
            logger.debug("キャッシュ済みエントリーシグナルの取得に失敗しました: %s", e)
            return None

    def calculate_position_size(self) -> float:
        """ポジションサイズを計算する。"""
        try:
            if (
                self.strategy.gene.position_sizing_gene
                and self.strategy.gene.position_sizing_gene.enabled
            ):
                current_price = (
                    self.strategy.data.Close[-1]
                    if hasattr(self.strategy, "data")
                    and len(self.strategy.data.Close) > 0
                    else 50000.0 * FRACTIONAL_UNIT
                )
                # FractionalBacktest により戦略データの価格は FRACTIONAL_UNIT 倍に
                # スケーリングされている。ポジションサイズ計算（数量ベース）は
                # 実価格（USDT）で行う必要があるため、フレーム価格を実価格へ戻す。
                real_price = current_price / FRACTIONAL_UNIT
                account_balance = getattr(self.strategy, "equity", 100000.0)
                # モックの場合はfloatにキャスト
                try:
                    account_balance = float(account_balance)
                except (TypeError, ValueError):
                    account_balance = 100000.0
                market_data = {}
                try:
                    if (
                        hasattr(self.strategy, "_precomputed_atr")
                        and self.strategy._precomputed_atr is not None
                    ):
                        idx = int(len(self.strategy.data)) - 1
                        if 0 <= idx < len(self.strategy._precomputed_atr):
                            atr = self.strategy._precomputed_atr[idx]
                            import numpy as np

                            if not np.isnan(atr) and current_price > 0:
                                market_data["atr_pct"] = atr / current_price
                                # 実価格換算値も提供（VOLATILITY_BASED 計算機が使用）
                                market_data["atr"] = atr / FRACTIONAL_UNIT

                    if "atr_pct" not in market_data:
                        lookback = getattr(
                            self.strategy.gene.position_sizing_gene,
                            "lookback_period",
                            14,
                        )
                        data_length = int(len(self.strategy.data))
                        if data_length > lookback + 1:
                            import numpy as np

                            high = np.array(self.strategy.data.High[-lookback:])
                            low = np.array(self.strategy.data.Low[-lookback:])
                            prev_close = np.array(
                                self.strategy.data.Close[-lookback - 1 : -1]
                            )

                            tr1 = high - low
                            tr2 = np.abs(high - prev_close)
                            tr3 = np.abs(low - prev_close)
                            tr = np.maximum(tr1, np.maximum(tr2, tr3))
                            atr = np.mean(tr)
                            if current_price > 0:
                                market_data["atr_pct"] = atr / current_price
                                market_data["atr"] = atr / FRACTIONAL_UNIT

                    # 直近リターン系列を提供（VOLATILITY_BASED の VaR/ES 制限用）
                    data_length = int(len(self.strategy.data))
                    var_lookback = getattr(
                        self.strategy.gene.position_sizing_gene, "var_lookback", 100
                    )
                    if data_length > var_lookback + 1:
                        import numpy as np

                        closes = np.asarray(
                            self.strategy.data.Close[-(var_lookback + 1) :],
                            dtype=float,
                        )
                        if len(closes) > 1 and np.all(np.isfinite(closes)):
                            with np.errstate(divide="ignore", invalid="ignore"):
                                returns = closes[1:] / closes[:-1] - 1.0
                            market_data["returns"] = returns[
                                np.isfinite(returns)
                            ].tolist()
                except Exception as e:
                    logger.debug("ATR market data calculation error: %s", e)

                position_size = (
                    self.strategy.position_sizing_service.calculate_position_size_fast(
                        gene=self.strategy.gene.position_sizing_gene,
                        account_balance=account_balance,
                        current_price=real_price,
                        market_data=market_data,
                    )
                )
                position_size = float(position_size)
                if not math.isfinite(position_size) or position_size <= 0:
                    return 0.001

                gene = self.strategy.gene.position_sizing_gene
                min_size_limit = max(
                    0.001, float(getattr(gene, "min_position_size", 0.001))
                )
                max_size_limit = float(
                    getattr(gene, "max_position_size", position_size)
                )
                if not math.isfinite(max_size_limit) or max_size_limit < min_size_limit:
                    max_size_limit = min_size_limit

                # 最終的なユニット数（この時点ではまだ小数である可能性がある）
                final_units = max(min_size_limit, min(max_size_limit, position_size))

                # backtesting.py の仕様に合わせて変換
                # 0 < size < 1: 証拠金比率
                # size >= 1: 整数のユニット数
                equity = getattr(self.strategy, "equity", 100000.0)
                # モックの場合はfloatにキャスト
                try:
                    equity = float(equity)
                except (TypeError, ValueError):
                    equity = 100000.0
                if equity > 0:
                    fraction = (final_units * real_price) / equity
                    if fraction < 1.0:
                        # 1未満なら比率（証拠金比率）として返す
                        # 0.001 などの小さな値でも OK
                        return fraction
                    # 1以上（100%以上の証拠金を使用）の要求は、証拠金比率の上限に
                    # クランプする（証拠金超過による注文キャンセルを防ぐ）。
                    # backtesting.py では 0 < size < 1 が証拠金比率、
                    # size >= 1 がユニット数として扱われるため、size == 1.0 は
                    # 1 ユニット（FRACTIONAL_UNIT 分）と解釈され比率として使えない。
                    # 1.0 未満の最大比率で返すことで、ブローカーが約定時に
                    # マージン内の枚数を計算する。
                    return 0.9999

                return 0.001
            return 0.01
        except Exception as e:
            logger.warning("ポジションサイズ計算エラー、フォールバック使用: %s", e)
            return 0.01

    def calculate_effective_tpsl_prices(
        self,
        direction: float,
        current_price: float,
    ) -> tuple[float | None, float | None]:
        """有効なTP/SL価格を計算する。"""
        active_tpsl_gene = cast(
            TPSLGene | None,
            self.strategy._get_effective_tpsl_gene(direction),
        )
        if not active_tpsl_gene:
            return None, None

        market_data = {}
        tpsl_method = active_tpsl_gene.method
        if tpsl_method in (
            TPSLMethod.VOLATILITY_BASED,
            TPSLMethod.ADAPTIVE,
            TPSLMethod.STATISTICAL,
        ):
            atr_period = getattr(active_tpsl_gene, "atr_period", 14)

            if (
                hasattr(self.strategy, "_precomputed_tpsl_atr")
                and atr_period in self.strategy._precomputed_tpsl_atr
            ):
                idx = int(len(self.strategy.data)) - 1
                atr_array = self.strategy._precomputed_tpsl_atr[atr_period]
                if 0 <= idx < len(atr_array):
                    val = atr_array[idx]
                    import numpy as np

                    if not np.isnan(val):
                        market_data["atr"] = val

            if "atr" not in market_data:
                required_slice_size = atr_period + 1
                data_length = int(len(self.strategy.data))
                if data_length > required_slice_size:
                    highs = self.strategy.data.High[-required_slice_size:]
                    lows = self.strategy.data.Low[-required_slice_size:]
                    closes = self.strategy.data.Close[-required_slice_size:]
                    market_data["ohlc_data"] = [
                        {"high": h, "low": low_val, "close": c}
                        for h, low_val, c in zip(highs, lows, closes, strict=False)
                    ]

            # STATISTICAL / ADAPTIVE 計算機向けに、過去価格系列・
            # ボラティリティ・トレンド情報を提供する。
            # これらが無いと統計・適応的ロジックが常にデフォルト値を返し、
            # 対応する遺伝子（lookback_period / confidence_threshold 等）が
            # 探索空間から実質消滅するため。
            if tpsl_method in (
                TPSLMethod.ADAPTIVE,
                TPSLMethod.STATISTICAL,
            ):
                lookback = max(
                    int(getattr(active_tpsl_gene, "lookback_period", 100)), 50
                )
                data_length = int(len(self.strategy.data))
                if data_length > lookback + 1:
                    closes = self.strategy.data.Close[-(lookback + 1) :]
                    market_data["historical_prices"] = [
                        float(c) for c in closes if c is not None
                    ]
                    market_data["historical_data_available"] = True

                atr_value = market_data.get("atr")
                if atr_value is not None and current_price > 0:
                    volatility_pct = float(atr_value) / current_price
                    market_data["volatility"] = volatility_pct
                    # ボラティリティレジーム（文字列）を判定
                    # （ADAPTIVE 計算機の方式選択で使用）
                    if volatility_pct >= 0.05:
                        market_data["volatility_regime"] = "very_high"
                    elif volatility_pct >= 0.03:
                        market_data["volatility_regime"] = "high"
                    elif volatility_pct <= 0.005:
                        market_data["volatility_regime"] = "low"
                    else:
                        market_data["volatility_regime"] = "normal"

                    # トレンドレジームを判定（直近価格変化率ベース）
                    historical_prices = market_data.get("historical_prices", [])
                    if len(historical_prices) >= 20:
                        start = historical_prices[-20]
                        end = historical_prices[-1]
                        if start and start > 0:
                            change = (end - start) / start
                            if change >= 0.02:
                                market_data["trend"] = "strong_up"
                            elif change <= -0.02:
                                market_data["trend"] = "strong_down"
                            else:
                                market_data["trend"] = "neutral"

        return cast(
            tuple[float | None, float | None],
            self.strategy.tpsl_service.calculate_tpsl_prices(
                current_price=current_price,
                tpsl_gene=active_tpsl_gene,
                position_direction=direction,
                market_data=market_data,
            ),
        )

    def tools_block_entry(self) -> bool:
        """ツールがエントリーをブロックするかチェックする。"""
        if not self.strategy.gene:
            return False
        if (
            not hasattr(self.strategy.gene, "tool_genes")
            or not self.strategy.gene.tool_genes
        ):
            return False

        from ..tools import ToolContext, tool_registry

        try:
            current_timestamp = self.strategy.data.index[-1]
            extra: dict[str, Any] = {}
            try:
                eq = float(getattr(self.strategy, "equity", None) or 0)
                mx = float(getattr(self.strategy, "_max_equity_seen", eq) or eq)
                extra["equity"] = eq
                extra["max_equity"] = mx
            except Exception:
                pass
            try:
                extra["closed_trades"] = list(
                    getattr(self.strategy, "closed_trades", []) or []
                )
            except Exception:
                pass
            try:
                idx = int(len(self.strategy.data)) - 1
                pre = getattr(self.strategy, "_precomputed_atr", None)
                if pre is not None and 0 <= idx < len(pre):
                    import numpy as np

                    atr = pre[idx]
                    price = float(self.strategy.data.Close[-1])
                    if not np.isnan(atr) and price > 0:
                        extra["atr"] = float(atr)
                        extra["atr_pct"] = float(atr) / price
            except Exception:
                pass
            context = ToolContext(
                timestamp=current_timestamp,
                current_price=float(self.strategy.data.Close[-1]),
                current_high=float(self.strategy.data.High[-1]),
                current_low=float(self.strategy.data.Low[-1]),
                current_volume=(
                    float(self.strategy.data.Volume[-1])
                    if hasattr(self.strategy.data, "Volume")
                    else 0.0
                ),
                extra_data=extra,
            )

            for tool_gene in self.strategy.gene.tool_genes:
                if not tool_gene.enabled:
                    continue

                tool = tool_registry.get(tool_gene.tool_name)
                if tool is None:
                    logger.warning(
                        "ツール '%s' がレジストリに見つかりません",
                        tool_gene.tool_name,
                    )
                    continue

                if tool.should_skip_entry(context, tool_gene.params):
                    return True

            return False
        except Exception as e:
            logger.warning("ツールフィルターエラー（フェイルセーフ適用）: %s", e)
            return False
