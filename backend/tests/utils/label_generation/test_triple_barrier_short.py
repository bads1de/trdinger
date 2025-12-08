import pandas as pd
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from app.services.ml.label_generation.triple_barrier import TripleBarrier


class TestTripleBarrierShort:
    def setup_method(self):
        # 1 hour data
        dates = pd.date_range(start="2023-01-01", periods=100, freq="1h")
        self.close = pd.Series(100.0, index=dates)
        self.volatility = pd.Series(0.01, index=dates)  # 1% volatility

    def test_short_profit_taking(self):
        """Test PT for Short position (Price Drop)"""
        # Price drops by 3% at index 5
        self.close.iloc[0] = 100.0
        self.close.iloc[1:5] = 100.0
        self.close.iloc[5] = 97.0

        tb = TripleBarrier(num_threads=1)

        vertical_barriers = pd.Series(
            self.close.index + pd.Timedelta(hours=10), index=self.close.index
        )

        # Side = -1 (Short)
        # We want PT at 2% drop.
        side = pd.Series(-1, index=self.close.index)

        events = tb.get_events(
            close=self.close,
            t_events=self.close.index[:10],
            pt_sl=[2.0, 2.0],
            target=self.volatility,
            min_ret=0.001,
            vertical_barrier_times=vertical_barriers,
            side=side,
        )

        labels = tb.get_bins(events, self.close)

        # Short position: Price drop of 3% > Target 1% * 2.0 = 2% drop.
        # So it should hit PT.
        # PT hit -> bin = 1.0 (Profit)

        assert labels.loc[self.close.index[0], "bin"] == 1.0
        # Check event side
        assert events.loc[self.close.index[0], "side"] == "pt"

    def test_short_stop_loss(self):
        """Test SL for Short position (Price Rise)"""
        # Price rises by 3% at index 5
        self.close.iloc[0] = 100.0
        self.close.iloc[1:5] = 100.0
        self.close.iloc[5] = 103.0

        tb = TripleBarrier(num_threads=1)

        vertical_barriers = pd.Series(
            self.close.index + pd.Timedelta(hours=10), index=self.close.index
        )

        side = pd.Series(-1, index=self.close.index)

        events = tb.get_events(
            close=self.close,
            t_events=self.close.index[:10],
            pt_sl=[2.0, 2.0],
            target=self.volatility,
            min_ret=0.001,
            vertical_barrier_times=vertical_barriers,
            side=side,
        )

        labels = tb.get_bins(events, self.close)

        # Short position: Price rise of 3% > Target 1% * 2.0 = 2%.
        # Should hit SL.
        # SL hit -> bin = -1.0 (Loss)

        assert labels.loc[self.close.index[0], "bin"] == -1.0
        assert events.loc[self.close.index[0], "side"] == "sl"
