import sys
import pandas as pd
import numpy as np

# パス設定
sys.path.append("c:/Users/buti3/trading/backend")

from app.services.ml.feature_engineering.feature_engineering_service import (
    FeatureEngineeringService,
)


def test_feature_limit_removal():
    """FeatureEngineeringServiceのデータ制限(200k行)が撤廃されたか確認"""
    service = FeatureEngineeringService()

    # 20万行を超えるダミーデータ作成 (200,005行)
    # 軽量化のためカラムは最小限にする
    rows = 200005
    dates = pd.date_range(start="2020-01-01", periods=rows, freq="1min")
    df = pd.DataFrame(
        {
            "open": np.random.rand(rows) * 100,
            "high": np.random.rand(rows) * 100,
            "low": np.random.rand(rows) * 100,
            "close": np.random.rand(rows) * 100,
            "volume": np.random.rand(rows) * 1000,
        },
        index=dates,
    )

    # 特徴量計算実行
    # 非常に重くなるので、計算負荷の低い設定で実行したいが、
    # calculate_advanced_features は FeatureCalculator を呼び出すので、
    # モックするか、あるいは単に行数が減っていないかだけを確認するために
    # 内部ロジックをバイパスしたいところだが、今回は統合テストとして実行する。
    # ただし時間がかかりすぎるのを防ぐため、lookback_periodsを最小にする。

    # ここではデータがカットされていないか（行数が維持されているか）だけが重要。
    # 重い計算を避けるため、内部の calculator をモックするのも手だが、
    # 実際のメソッドを実行した方が確実。

    # エラー回避のため、最低限のCalculatorのみ実行されるようにハックするか、
    # あるいはタイムアウトを許容するか。
    # 20万行でも単純な計算ならそこまで遅くないはず（Numbaなど最適化されているため）。
    # 重い VolumeProfileなどはデフォルトで計算される可能性がある。

    # テストの実行時間を短縮するため、簡易的なチェックで済ませる
    # calculate_advanced_features の冒頭部分でのカットがなくなったかを確認したい。

    # 方法: FeatureEngineeringServiceを継承して、計算部分をオーバーライドし、
    # データを受け取った時点での行数を確認する。

    class TestableFeatureEngineeringService(FeatureEngineeringService):
        def __init__(self):
            super().__init__()
            self.calculated_length = 0

        def _save_to_cache(self, key, data):
            pass  # キャッシュ無効化

        # _get_from_cache をオーバーライドして常にNoneを返す
        def _get_from_cache(self, key):
            return None

        # price_calculator などをダミーに差し替える
        # しかし calculate_advanced_features メソッド内で self.price_calculator を呼んでいるので
        # インスタンス生成後に差し替えが可能。

    service = TestableFeatureEngineeringService()

    # 各 Calculator の mock
    class DummyCalculator:
        def calculate_features(self, df, *args, **kwargs):
            return df

        def create_crypto_features(self, df, *args, **kwargs):
            return df

        def calculate_interaction_features(self, df, *args, **kwargs):
            return df

    # 全てのCalculatorをダミーに置換
    service.price_calculator = DummyCalculator()
    service.market_data_calculator = DummyCalculator()
    service.technical_calculator = DummyCalculator()
    service.interaction_calculator = DummyCalculator()
    service.microstructure_calculator = DummyCalculator()
    service.volume_profile_calculator = DummyCalculator()
    service.oi_fr_interaction_calculator = DummyCalculator()
    service.advanced_stats_calculator = DummyCalculator()
    service.multi_timeframe_calculator = DummyCalculator()
    service.crypto_features = DummyCalculator()  # Noneチェックがあるので注意

    # 実行
    result_df = service.calculate_advanced_features(df)

    print(f"Original length: {rows}")
    print(f"Result length: {len(result_df)}")

    assert (
        len(result_df) == rows
    ), f"データ行数が減少しています: {len(result_df)} (期待値: {rows})"
    assert len(result_df) > 200000, "200,000行以上のデータが保持されていません"


if __name__ == "__main__":
    try:
        test_feature_limit_removal()
        print("✅ Feature limit removal test passed")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)




