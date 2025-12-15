import pandas as pd
import numpy as np
import pytest
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from app.services.ml.label_generation.presets import triple_barrier_method_preset

class TestTripleBarrierATRPreset:
    """ATRベースのトリプルバリアプリセットテスト"""

    def setup_method(self):
        # 1時間ごとのデータ (100時間分)
        dates = pd.date_range(start="2023-01-01", periods=100, freq="1h")
        
        # シンプルな上昇トレンド + ボラティリティ変化を作る
        # 前半(0-49): ボラティリティ小 (変動幅 1)
        # 後半(50-99): ボラティリティ大 (変動幅 5)
        
        close = []
        high = []
        low = []
        base_price = 100.0
        
        for i in range(100):
            if i < 50:
                vol = 1.0
            else:
                vol = 5.0
            
            # 緩やかな上昇トレンド
            base_price += 0.1 
            
            c = base_price
            h = c + vol/2
            l = c - vol/2
            
            close.append(c)
            high.append(h)
            low.append(l)
            
        self.df = pd.DataFrame({
            'open': close, # 簡易的にopen=close
            'high': high,
            'low': low,
            'close': close,
            'volume': [1000] * 100
        }, index=dates)

    def test_atr_calculation_logic(self):
        """
        ATRベースのバリアが機能しているか確認するテスト。
        
        同じ値幅(例: +3.0)の動きでも、
        ボラティリティが高い時(ATR=5.0) -> バリア(1ATR=5.0)に届かない -> RANGE(0)
        ボラティリティが低い時(ATR=1.0) -> バリア(1ATR=1.0)を超える -> UP(1)
        となることを確認する。
        """
        
        # テスト用の特定の動きを注入
        
        # ケース1: ボラティリティが低い期間 (i=10, ATR約1.0)
        # 4時間後(i=14)に +3.0 上昇させる
        # 1ATR(1.0) < 3.0 なので、PT=1.0なら UP になるはず
        current_price_low_vol = self.df['close'].iloc[10]
        self.df.loc[self.df.index[14], 'close'] = current_price_low_vol + 3.0
        
        # ケース2: ボラティリティが高い期間 (i=60, ATR約5.0)
        # 4時間後(i=64)に +3.0 上昇させる
        # 1ATR(5.0) > 3.0 なので、PT=1.0なら 届かず RANGE になるはず
        current_price_high_vol = self.df['close'].iloc[60]
        self.df.loc[self.df.index[64], 'close'] = current_price_high_vol + 3.0
        
        # 実行
        # use_atr=True を指定
        labels = triple_barrier_method_preset(
            self.df,
            timeframe="1h",
            horizon_n=10,
            pt=1.0,
            sl=1.0,
            use_atr=True,        # 新規パラメータ
            atr_period=14        # ATR期間
        )
        
        # 検証
        
        # 場所を変更: i=20 (低ボラ)
        current_price_low = self.df['close'].iloc[20]
        self.df.loc[self.df.index[24], 'close'] = current_price_low + 3.0
        
        # 場所を変更: i=70 (高ボラ)
        current_price_high = self.df['close'].iloc[70]
        self.df.loc[self.df.index[74], 'close'] = current_price_high + 3.0
        
        # 再実行
        labels = triple_barrier_method_preset(
            self.df,
            timeframe="1h",
            horizon_n=4, # 4時間後を見る
            pt=1.0,
            sl=1.0,
            use_atr=True,
            atr_period=5 # 短めに設定して値を安定させる
        )
        
        # ケース1 (i=20): UP
        # ATRは約1.0。変動+3.0は 3.0倍 > 1.0倍(PT)。よってUP。
        assert labels.iloc[20] == "UP", f"低ボラティリティ時(ATR約1.0)、+3.0変動はUPであるべき。結果: {labels.iloc[20]}"
        
        # ケース2 (i=70): RANGE
        # ATRは約5.0。変動+3.0は 0.6倍 < 1.0倍(PT)。よってRANGE。
        assert labels.iloc[70] == "RANGE", f"高ボラティリティ時(ATR約5.0)、+3.0変動はRANGEであるべき。結果: {labels.iloc[70]}"

