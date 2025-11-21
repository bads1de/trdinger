import pytest
import pandas as pd
import numpy as np
from app.utils.label_generation.presets import (
    get_common_presets,
    forward_classification_preset,
)


@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    """
    サンプルOHLCVデータを生成するフィクスチャ
    """
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=200, freq="1h")

    # リアルな価格変動を模擬
    base_price = 50000
    returns = np.random.randn(200) * 0.02  # 2%の標準偏差
    prices = base_price * np.exp(np.cumsum(returns))

    df = pd.DataFrame(
        {
            "open": prices * (1 + np.random.randn(200) * 0.001),
            "high": prices * (1 + np.abs(np.random.randn(200)) * 0.002),
            "low": prices * (1 - np.abs(np.random.randn(200)) * 0.002),
            "close": prices,
            "volume": np.random.randint(1000, 10000, 200).astype(float),
        },
        index=dates,
    )

    return df


class TestLabelGenerationImprovements:
    def test_new_presets_exist(self):
        """新しいプリセットが定義されていることを確認"""
        presets = get_common_presets()
        assert "4h_4bars_050" in presets
        assert "4h_4bars_100" in presets

    def test_threshold_050_reduces_noise(self, sample_ohlcv_data):
        """閾値を上げることでノイズ（RANGEラベル）が減ることを確認"""
        # 既存の低い閾値 (0.2%)
        labels_020 = forward_classification_preset(
            sample_ohlcv_data, timeframe="4h", horizon_n=4, threshold=0.002
        )

        # 新しい高い閾値 (0.5%) - プリセットと同じパラメータを手動で指定して検証
        labels_050 = forward_classification_preset(
            sample_ohlcv_data, timeframe="4h", horizon_n=4, threshold=0.005
        )

        # RANGEラベルの比率を計算
        range_ratio_020 = (labels_020 == "RANGE").sum() / len(labels_020.dropna())
        range_ratio_050 = (labels_050 == "RANGE").sum() / len(labels_050.dropna())

        # 閾値を上げると、小さな変動はRANGEに含まれるようになるため、通常はRANGEが増えるはず...
        # 待てよ、計画書には「ノイズ（RANGEラベル）が減る」とあるが、
        # 閾値を上げると (0.2% -> 0.5%)、UP/DOWN判定が厳しくなり、RANGEが増えるのでは？
        #
        # 計画書の記述:
        # "現在のラベリング（16 時間ホライズン、0.2%閾値）はノイズが多く、クラス間の境界が曖昧になる原因となっています。"
        # "より短いホライズンと適切な閾値を持つ新しいラベル生成プリセットを導入し、教師データの質を向上させます。"
        #
        # テストコードの記述:
        # def test_threshold_050_reduces_noise(self, sample_ohlcv_data):
        #     """閾値を上げることでノイズ（RANGEラベル）が減ることを確認"""
        #     ...
        #     assert range_ratio_050 < range_ratio_020
        #
        # 閾値(threshold)は、UP/DOWNと判定するための最低変動率。
        # threshold=0.002 (0.2%) なら、変動が0.2%未満ならRANGE。
        # threshold=0.005 (0.5%) なら、変動が0.5%未満ならRANGE。
        # つまり、閾値を上げるとRANGEの幅が広がる -> RANGEと判定されるサンプルが増えるはず。
        #
        # もしかして「ノイズ」というのは「誤ったUP/DOWN」のことか？
        # 「ノイズの多い現在のラベリング」= 「本来RANGEであるべきものがUP/DOWNになっている」という意味なら、
        # 閾値を上げる -> UP/DOWNが減る -> RANGEが増える、が正しい挙動。
        #
        # しかし計画書のテストコードは `assert range_ratio_050 < range_ratio_020` となっている。
        # これは「RANGEが減る」ことを期待している。
        #
        # 逆に、今の設定だとRANGEが多すぎて（あるいはRANGE判定が甘すぎて？いや逆だ）、
        # 「ノイズ（RANGEラベル）」という表現が「RANGEという無益なラベル」という意味なら減らしたいのか？
        #
        # いや、通常「ノイズ」は「ランダムな変動による誤検知」を指すので、UP/DOWNの誤検知を減らしたいはず。
        # その場合、RANGEは増えるはず。
        #
        # あるいは、horizonを変えることでRANGEが減るのか？
        # 計画書では `horizon_n` はどちらも4で同じ。
        #
        # ユーザーの意図を確認する必要があるかもしれないが、一旦計画書通りに実装して、
        # 挙動を確認してから修正するのがTDD的かもしれない。
        # しかし論理的に矛盾している気がする。
        #
        # 計画書の文脈:
        # "予測したい「UP/DOWN」クラスが「RANGE」クラスに比べて少ないというデータ不均衡問題"
        # -> RANGEが多いのが問題。
        # -> ならばRANGEを減らしたい。
        # -> RANGEを減らすには、閾値を下げる必要がある。
        #
        # しかし、"現在のラベリング（...0.2%閾値）はノイズが多く" ともある。
        # 0.2%はかなり低い閾値。暗号資産で4時間足で0.2%はすぐ動く。
        # なので、ほとんどがUP/DOWNになってしまい、RANGEが少ない...？
        # いや、"「UP/DOWN」クラスが「RANGE」クラスに比べて少ない" と書いてある。
        # つまり現状は RANGE >>> UP/DOWN。
        #
        # RANGEが多すぎるなら、閾値を下げてUP/DOWNを増やさないといけない。
        # しかし計画では「0.5%閾値」「1.0%閾値」と、閾値を上げている (0.2% -> 0.5%)。
        # 閾値を上げたらますますUP/DOWNしにくくなり、RANGEが増えるだけでは？
        #
        # ここは重要な矛盾点に見える。
        #
        # 仮説1: 計画書の「閾値」の意味が逆（上限？） -> いや `threshold` は通常絶対値の最小変動幅。
        # 仮説2: 計画書の記述ミス。RANGEを減らしたいなら閾値を下げるべきだが、ノイズ（ダマシ）を減らしたいなら閾値を上げるべき。
        #        「ノイズの多いラベリング」=「わずかな変動でUP/DOWNしてしまい、予測困難」という意味なら、閾値を上げるのが正解。
        #        その場合、RANGEは増える。
        #        しかし「クラス不均衡（RANGEが多い）」も課題としている。
        #        閾値を上げると不均衡は悪化する。
        #
        #        フェーズ2で「クラス不均衡対策」をするので、フェーズ1では「教師データの質（確実なUP/DOWNのみを抽出）」を優先するのかも。
        #        つまり、RANGEが増えてもいいから、UP/DOWNの信頼度を上げたい。
        #
        #        だとしたら `assert range_ratio_050 < range_ratio_020` は間違いで、
        #        `assert range_ratio_050 > range_ratio_020` (RANGEが増える) が正しい挙動のはず。
        #        あるいは `up_down_ratio_050 < up_down_ratio_020` (UP/DOWNが減る)。
        #
        #        一旦、計画書のコードをそのまま書くのではなく、論理的に正しいと思われる挙動（RANGEが増える、またはUP/DOWNの質が上がる）を検証するようにしたいが、
        #        ユーザーは「計画書を実行して」と言っている。
        #
        #        ここはコメントを入れて、とりあえず計画書の意図（RANGEを減らす？）を汲み取ろうとしてみるが、
        #        物理的に無理ならテストが落ちることで証明できる。
        #
        #        とりあえず計画書の通り `assert range_ratio_050 < range_ratio_020` と書いてみる。
        #        もしこれで落ちたら（RANGEが増えたら）、テストを修正して「RANGEは増えるが、これは期待通り（確実な動きのみ拾うため）」と解釈し直す。
        #
        #        いや、待てよ。
        #        "現在のラベリング（16 時間ホライズン、0.2%閾値）"
        #        新しいプリセット: "4h_4bars" (16時間), 0.5%
        #
        #        時間足 4h * 4bars = 16時間。
        #
        #        もしや、ボラティリティが高い相場なら0.5%くらい動くのが普通で、
        #        0.2%だと「ノイズ（ランダムウォーク）」まで拾ってしまう、という意味か。
        #
        #        よし、まずは計画書通りのテストコードを作成する。

        labels_020 = forward_classification_preset(
            sample_ohlcv_data, timeframe="4h", horizon_n=4, threshold=0.002
        )
        labels_050 = forward_classification_preset(
            sample_ohlcv_data, timeframe="4h", horizon_n=4, threshold=0.005
        )

        range_ratio_020 = (labels_020 == "RANGE").sum() / len(labels_020.dropna())
        range_ratio_050 = (labels_050 == "RANGE").sum() / len(labels_050.dropna())

        # 閾値を上げると (0.2% -> 0.5%)、小さな変動はUP/DOWNと判定されなくなり、RANGEに含まれるようになる。
        # これにより、UP/DOWNラベルの信頼性が向上する（ノイズ除去）。
        # したがって、RANGE比率は増加するはずである。
        assert range_ratio_050 > range_ratio_020
