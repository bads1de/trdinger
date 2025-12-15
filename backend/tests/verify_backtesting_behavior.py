from backtesting import Backtest, Strategy
import pandas as pd
import numpy as np

# Create dummy data
data = pd.DataFrame(
    {
        "Open": np.random.rand(100),
        "High": np.random.rand(100),
        "Low": np.random.rand(100),
        "Close": np.random.rand(100),
        "Volume": np.random.rand(100),
    }
)


class TestStrategy(Strategy):
    def init(self):
        print("Init called")

    def next(self):
        # Print length of data
        if len(self.data) < 5:
            print(f"Len: {len(self.data)}, Close[-1]: {self.data.Close[-1]}")

        # Check if accessing [-1] changes over time
        if len(self.data) == 50:
            print(f"At 50: {self.data.Close[-1]}")


bt = Backtest(data, TestStrategy)
bt.run()


