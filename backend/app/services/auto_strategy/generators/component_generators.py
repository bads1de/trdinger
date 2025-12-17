"""
戦略コンポーネント生成器

TP/SL、ポジションサイジング、エントリー設定などの
戦略構成要素を生成するジェネレーター群をまとめたモジュール。
"""

import logging
import random

from ..config.constants import EntryType
from ..genes import (
    EntryGene,
    PositionSizingGene,
    PositionSizingMethod,
    TPSLGene,
    TPSLMethod,
    create_random_position_sizing_gene,
    create_random_tpsl_gene,
)

logger = logging.getLogger(__name__)


class TPSLGenerator:
    """
    TP/SL遺伝子の生成を担当するクラス
    """

    def __init__(self, config: any):
        """
        初期化

        Args:
            config: GA設定オブジェクト
        """
        self.config = config

    def generate_tpsl_gene(self) -> TPSLGene:
        """
        TP/SL遺伝子を生成（GA最適化対象）

        Returns:
            生成されたTP/SL遺伝子
        """
        try:
            # Anyの設定範囲内でランダムなTP/SL遺伝子を生成
            tpsl_gene = create_random_tpsl_gene()

            # Anyの制約を適用（設定されている場合）
            if hasattr(self.config, "tpsl_method_constraints"):
                # 許可されたメソッドのみを使用
                allowed_methods = self.config.tpsl_method_constraints
                if allowed_methods:
                    tpsl_gene.method = random.choice(
                        [TPSLMethod(m) for m in allowed_methods]
                    )

            if (
                hasattr(self.config, "tpsl_sl_range")
                and self.config.tpsl_sl_range is not None
            ):
                sl_min, sl_max = self.config.tpsl_sl_range
                tpsl_gene.stop_loss_pct = random.uniform(sl_min, sl_max)
                tpsl_gene.base_stop_loss = random.uniform(sl_min, sl_max)

            if (
                hasattr(self.config, "tpsl_tp_range")
                and self.config.tpsl_tp_range is not None
            ):
                # TP範囲制約
                tp_min, tp_max = self.config.tpsl_tp_range
                tpsl_gene.take_profit_pct = random.uniform(tp_min, tp_max)

            if (
                hasattr(self.config, "tpsl_rr_range")
                and self.config.tpsl_rr_range is not None
            ):
                # リスクリワード比範囲制約
                rr_min, rr_max = self.config.tpsl_rr_range
                tpsl_gene.risk_reward_ratio = random.uniform(rr_min, rr_max)

            if (
                hasattr(self.config, "tpsl_atr_multiplier_range")
                and self.config.tpsl_atr_multiplier_range is not None
            ):
                # ATR倍率範囲制約
                atr_min, atr_max = self.config.tpsl_atr_multiplier_range
                tpsl_gene.atr_multiplier_sl = random.uniform(atr_min, atr_max)
                tpsl_gene.atr_multiplier_tp = random.uniform(
                    atr_min * 1.5, atr_max * 2.0
                )

            return tpsl_gene

        except (ValueError, TypeError, KeyError, AttributeError) as e:
            logger.error(f"エラー: TP/SL遺伝子生成中に例外が発生しました: {e}")
            # フォールバック: デフォルトのTP/SL遺伝子
            return TPSLGene(
                method=TPSLMethod.RISK_REWARD_RATIO,
                stop_loss_pct=0.03,
                take_profit_pct=0.06,
                risk_reward_ratio=2.0,
                base_stop_loss=0.03,
                enabled=True,
            )


class PositionSizingGenerator:
    """
    ポジションサイジング遺伝子の生成と管理を担当するクラス
    """

    def __init__(self, config: any):
        """
        初期化

        Args:
            config: GA設定オブジェクト
        """
        self.config = config

    def generate_position_sizing_gene(self):
        """
        ポジションサイジング遺伝子を生成

        設定に基づいてランダムなポジションサイジング遺伝子を生成する。
        エラー発生時はデフォルトの遺伝子を返す。

        Returns:
            PositionSizingGene: 生成されたポジションサイジング遺伝子
        """
        try:
            return create_random_position_sizing_gene(self.config)
        except (ValueError, TypeError, KeyError, AttributeError) as e:
            logger.error(f"エラー: ポジションサイジング遺伝子生成に失敗しました: {e}")
            # フォールバック: デフォルト遺伝子を返す
            return PositionSizingGene(
                method=PositionSizingMethod.FIXED_RATIO,
                fixed_ratio=0.1,
                max_position_size=20.0,  # より大きなデフォルト値
                enabled=True,
            )


class EntryGenerator:
    """
    エントリー遺伝子の生成を担当するクラス
    """

    # エントリータイプの出現確率（初期段階では成行注文を多めに設定）
    DEFAULT_ENTRY_TYPE_WEIGHTS = {
        EntryType.MARKET: 0.6,
        EntryType.LIMIT: 0.2,
        EntryType.STOP: 0.15,
        EntryType.STOP_LIMIT: 0.05,
    }

    def __init__(self, config: any):
        """
        初期化

        Args:
            config: GA設定オブジェクト
        """
        self.config = config

    def generate_entry_gene(self) -> EntryGene:
        """
        ランダムなエントリー遺伝子を生成

        Returns:
            生成されたエントリー遺伝子
        """
        try:
            # エントリータイプを重み付きでランダム選択
            entry_type = self._select_entry_type()

            # オフセット値をランダム生成（0.1% ~ 2.0%）
            limit_offset_pct = random.uniform(0.001, 0.02)
            stop_offset_pct = random.uniform(0.001, 0.02)

            # 有効期限をランダム生成（1 ~ 20バー）
            order_validity_bars = random.randint(1, 20)

            return EntryGene(
                entry_type=entry_type,
                limit_offset_pct=limit_offset_pct,
                stop_offset_pct=stop_offset_pct,
                order_validity_bars=order_validity_bars,
                enabled=True,
                priority=random.uniform(0.5, 1.5),
            )

        except (ValueError, TypeError, KeyError, AttributeError) as e:
            logger.error(f"エラー: エントリー遺伝子生成中に例外が発生しました: {e}")
            # フォールバック: 成行注文のデフォルト遺伝子
            return EntryGene(
                entry_type=EntryType.MARKET,
                limit_offset_pct=0.005,
                stop_offset_pct=0.005,
                order_validity_bars=5,
                enabled=True,
            )

    def _select_entry_type(self) -> EntryType:
        """
        重み付きでエントリータイプを選択

        Returns:
            選択されたエントリータイプ
        """
        # 設定から重みを取得、なければデフォルトを使用
        weights = self.DEFAULT_ENTRY_TYPE_WEIGHTS

        if hasattr(self.config, "entry_type_weights"):
            custom_weights = self.config.entry_type_weights
            if custom_weights:
                weights = {EntryType(k): v for k, v in custom_weights.items()}

        # 重み付きランダム選択
        types = list(weights.keys())
        type_weights = list(weights.values())

        return random.choices(types, weights=type_weights, k=1)[0]
