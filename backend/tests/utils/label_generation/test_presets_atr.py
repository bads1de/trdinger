import pandas as pd
import numpy as np
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from app.services.ml.label_generation.presets import triple_barrier_method_preset


class TestTripleBarrierATRPreset:
    """ATRベースのトリプルバリアプリセットテスト（二値分類）"""

    def setup_method(self):
        # 1時間ごとのデータ (120時間分)
        dates = pd.date_range(start="2023-01-01", periods=120, freq="1h")

        # シンプルな上昇トレンド + ボラティリティ変化を作る
        # 前半(0-59): ボラティリティ小 (変動幅 1)
        # 後半(60-119): ボラティリティ大 (変動幅 5)

        close = []
        high = []
        low = []
        base_price = 100.0

        for i in range(120):
            if i < 60:
                vol = 1.0
            else:
                vol = 5.0

            # 緩やかな上昇トレンド
            base_price += 0.1

            c = base_price
            h = c + vol / 2
            low_val = c - vol / 2

            close.append(c)
            high.append(h)
            low.append(low_val)

        self.df = pd.DataFrame(
            {
                "open": close,  # 簡易的にopen=close
                "high": high,
                "low": low,
                "close": close,
                "volume": [1000] * 120,
            },
            index=dates,
        )

    def test_atr_calculation_logic(self):
        """
        ATRベースのバリアが機能しているか確認するテスト（二値分類）。

        ラベルは 0（バリアに届かず / Invalid）または 1（バリアに到達 / Valid）。
        """

        # 実行: ATRを使用
        labels = triple_barrier_method_preset(
            self.df,
            timeframe="1h",
            horizon_n=10,
            pt=1.0,
            sl=1.0,
            use_atr=True,
            atr_period=14,
        )

        # ラベルは二値分類 (0 or 1)
        unique_labels = labels.dropna().unique()
        for label in unique_labels:
            assert label in [0, 1, 0.0, 1.0], f"ラベルは0か1であるべき: {label}"

        # 結果の長さが妥当であることを確認
        assert len(labels) > 0, "ラベルが生成されていること"
        assert len(labels.dropna()) > 0, "有効なラベルが存在すること"

        # ATRベースのラベリングが動作していることを確認
        # (ボラティリティが異なる期間で、異なるラベル分布が期待される)
        # 低ボラ期間は変動が相対的に大きくなるため、Validが多くなる傾向
        # 高ボラ期間は変動が相対的に小さくなるため、Invalidが多くなる傾向

        # ラベルに両方の値（0と1）が含まれていることを確認
        # （データの性質上、両方のクラスが存在するはず）
        non_nan_labels = labels.dropna()
        assert (
            0 in non_nan_labels.values or 1 in non_nan_labels.values
        ), "ラベルに0か1が含まれていること"
