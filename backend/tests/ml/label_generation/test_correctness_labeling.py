import numpy as np
import pandas as pd
import pytest
from scipy import stats
from app.services.ml.label_generation.triple_barrier import TripleBarrier
from app.services.ml.label_generation.trend_scanning import TrendScanning


def test_trend_scanning_correctness():
    """Trend Scanningの計算結果（t値、係数）がscipyの実装と一致するか確認"""
    np.random.seed(42)
    # ランダムウォークデータ
    n = 100
    close = pd.Series(np.cumprod(1 + np.random.randn(n) * 0.01) * 100)

    # テスト対象
    ts = TrendScanning(min_window=10, max_window=30, step=1)

    # 実際の実装から結果を取得 (t値を返すモードで)
    df_res = ts.get_labels(close, return_t_value=True)

    # 手動検証
    # ランダムにいくつかのポイントを選んで検証
    check_indices = np.random.choice(df_res.index, size=5, replace=False)

    for t0 in check_indices:
        row = df_res.loc[t0]
        if pd.isna(row["t1"]):
            continue

        t1 = row["t1"]
        t0_idx = close.index.get_loc(t0)
        t1_idx = close.index.get_loc(t1)

        # ウィンドウ期間のデータ
        # TrendScanningの実装では Closeの対数 をとって回帰している
        # 該当部分: close_values = np.log(close.values.astype(np.float64))
        y = np.log(close.iloc[t0_idx : t1_idx + 1].values)
        x = np.arange(len(y))

        # Scipyで線形回帰
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        t_value_scipy = slope / std_err if std_err > 0 else 0.0

        # Numba実装の結果
        t_value_numba = row["t_value"]

        # t値が一致するか (浮動小数点の誤差許容)
        # Numba実装では t値を +/- 100 でクリップしている点に注意
        if abs(t_value_scipy) > 100:
            expected = 100.0 * np.sign(t_value_scipy)
        else:
            expected = t_value_scipy

        print(
            f"Index: {t0}, Numba T: {t_value_numba:.4f}, Scipy T: {t_value_scipy:.4f}"
        )

        # かなり近い値であることを期待（計算順序の違いで多少ずれる）
        assert np.isclose(
            t_value_numba, expected, atol=1e-4
        ), f"Mismatch at {t0}: Numba={t_value_numba}, Scipy={expected}"


def test_triple_barrier_correctness():
    """Triple Barrierの判定ロジックがPythonの素直な実装と一致するか確認"""
    np.random.seed(42)
    n = 200
    index = pd.date_range("2024-01-01", periods=n, freq="1h")
    # 単調増加と単調減少、ランダムを含める
    close_vals = np.concatenate(
        [
            np.linspace(100, 110, 50),  # 上昇
            np.linspace(110, 100, 50),  # 下降
            100 + np.random.randn(100).cumsum(),  # ランダム
        ]
    )
    close = pd.Series(close_vals, index=index)

    target = pd.Series(0.01, index=index)  # 1% target

    # 垂直バリアなし
    tb = TripleBarrier(pt=1.0, sl=1.0, min_ret=0.0001)

    # Side = 1 (Long)
    events_long = tb.get_events(
        close, index, [1.0, 1.0], target, 0.0001, side=pd.Series(1, index=index)
    )

    # Side = -1 (Short)
    events_short = tb.get_events(
        close, index, [1.0, 1.0], target, 0.0001, side=pd.Series(-1, index=index)
    )

    # Pythonでの簡易検証ロジック
    def verify_point(events_df, idx_check, side_val):
        t0 = index[idx_check]
        row = events_df.loc[t0]

        # Numbaの結果
        numba_t1 = row["t1"]
        numba_side = row["side"]

        if pd.isna(numba_t1):
            # Numbaがヒットなしとした場合、Pythonでもヒットなしであるべき
            p0 = close.iloc[idx_check]
            upper = p0 * (1 + 0.01)
            lower = p0 * (1 - 0.01)

            # 将来の全データをチェック
            future_prices = close.iloc[idx_check + 1 :]
            if side_val == 1:
                hit_pt = future_prices[future_prices >= upper]
                hit_sl = future_prices[future_prices <= lower]
            else:
                hit_pt = future_prices[future_prices <= lower]  # Short利確は価格下落
                hit_sl = future_prices[future_prices >= upper]  # Short損切りは価格上昇

            assert (
                hit_pt.empty and hit_sl.empty
            ), f"Numba missed a hit at {t0}. Side={side_val}"
            return

        # Numbaがヒットした時刻までのデータで検証
        # Numbaが報告した時刻以前に、別の条件を満たしていないか？

        p0 = close.iloc[idx_check]
        trgt = 0.01

        # t0の次から t1まで
        sub_period = close.loc[t0:numba_t1][1:]  # t0含まず

        if side_val == 1:
            # Long
            # PT条件: ret >= trgt
            # SL条件: ret <= -trgt
            rets = sub_period / p0 - 1

            # 最初に条件を満たしたのが t1 であること
            first_pt = rets[rets >= trgt].index.min()
            first_sl = rets[rets <= -trgt].index.min()

        else:
            # Short
            # PT条件: ret <= -trgt
            # SL条件: ret >= trgt
            rets = sub_period / p0 - 1

            first_pt = rets[rets <= -trgt].index.min()
            first_sl = rets[rets >= trgt].index.min()

        # Pythonで見つけた最初のイベント
        valid_events = [ts for ts in [first_pt, first_sl] if pd.notna(ts)]
        if not valid_events:
            # ここに来るのはおかしい（NumbaはヒットしているのにPythonが見つけられない）
            raise AssertionError(
                f"Python logic found no event, but Numba found {numba_t1}"
            )

        first_event = min(valid_events)

        assert (
            first_event == numba_t1
        ), f"Timing mismatch at {t0}. Numba={numba_t1}, Python={first_event}"

        # Sideの整合性
        if side_val == 1:
            ret_at_hit = close.loc[first_event] / p0 - 1
            expected_type = "pt" if ret_at_hit > 0 else "sl"
        else:
            ret_at_hit = close.loc[first_event] / p0 - 1
            expected_type = "pt" if ret_at_hit < 0 else "sl"  # Shortはリターン負で利確

        assert (
            numba_side == expected_type
        ), f"Side mismatch at {t0}. Numba={numba_side}, Python={expected_type}, Ret={ret_at_hit}"

    # ランダムに検証
    for i in np.random.choice(range(len(index) - 1), 10, replace=False):
        verify_point(events_long, i, 1)
        verify_point(events_short, i, -1)

    print("\nCorrectness verification passed!")


if __name__ == "__main__":
    test_trend_scanning_correctness()
    test_triple_barrier_correctness()
