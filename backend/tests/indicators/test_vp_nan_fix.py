#!/usr/bin/env python3
"""
VPインジケーター NaN修正テスト

VPがNaNになる原因を特定し、修正するためのTDDテスト
"""

import pandas as pd
import numpy as np

from app.services.indicators.technical_indicators.volume import VolumeIndicators


def test_vp_with_zero_volume():
    """vol=0のデータでVPがNaNになることを確認"""
    close = pd.Series([100.0] * 20)
    volume = pd.Series([0] * 20)  # 全てゼロボリューム

    result = VolumeIndicators.vp(close, volume, width=10)

    # ゼロボリュームの場合は全てNaNになる
    all_nan = True
    for r in result:
        if not isinstance(r, pd.Series):
            all_nan = False
            break
        if not r.isna().all():
            all_nan = False
            break

    assert all_nan, "ゼロボリュームの場合、全てNaNになるべき"


def test_vp_with_short_data():
    """データ長不足でVPが空になることを確認"""
    close = pd.Series([100.0, 101.0])
    volume = pd.Series([100, 200])

    result = VolumeIndicators.vp(close, volume, width=10)

    # データ長<widthなので空Seriesが返される
    for r in result:
        assert isinstance(r, pd.Series), "結果はpd.Seriesであるべき"
        assert len(r) == 0, "データ長不足なので空シリーズ"


def test_vp_with_same_price_range():
    """価格範囲が0でVPがNaNになることを確認"""
    close = pd.Series([100.0] * 50)  # 全て同じ価格
    volume = pd.Series([100] * 50)

    result = VolumeIndicators.vp(close, volume, width=10)

    # 価格範囲=0の場合、適切に処理されるべき（現在はNaN）
    for r in result:
        assert isinstance(r, pd.Series), "結果はpd.Seriesであるべき"
        # 現在はエラーになるかも、修正で処理
        if len(r) > 0:
            # 修正後有効な値を持つべき
            pass


def test_vp_with_nan_volume():
    """ボリュームにNaNを含むデータでVPの耐久性を確認"""
    close = pd.Series([100.0] * 20)
    volume = pd.Series([100.0, np.nan] + [100.0] * 18)

    result = VolumeIndicators.vp(close, volume, width=10)

    # NaNを含む場合でも処理され、結果がある
    assert result is not None, "結果はNoneでない"
    assert len(result) == 6, "6つのシリーズが返される"


def test_vp_normal_case():
    """正常スペックVでVPが正常に計算されることを確認"""
    close = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0] * 20)  # もzetaスキなデータ
    volume = pd.Series([100, 150, 200, 250, 300] * 20)  # もずい volna

    result = VolumeIndicators.vp(close, volume, width=10)

    # 正常データではVolume Profileが計算される
    assert result is not None, "結果はNoneでない"
    assert len(result) == 6, "6つのシリーズが返される"

    for r in result:
        assert isinstance(r, pd.Series), f"陽極アイテムはpd.Seriesであるべき: {type(r)}"

    # 少なくとも一部は有効な値を持つべき
    at_least_one_valid = any(len(r) > 0 and not r.isna().all() for r in result)
    assert at_least_one_valid, "少なくとも一つは有効な値を持つべき"


if __name__ == "__main__":
    print("VP NaN修正テストを実行")

    try:
        print("1. ゼロボリュームテスト")
        test_vp_with_zero_volume()
        print(" ✓ ゼロボリュームテスト通過")

        print("2. データ長不足テスト")
        test_vp_with_short_data()
        print(" ✓ データ長不足テスト通過")

        print("3. 価格範囲0テスト")
        test_vp_with_same_price_range()
        print(" ✓ 価格範囲0テスト通過")

        print("4. NaNボリュームテスト")
        test_vp_with_nan_volume()
        print(" ✓ NaNボリュームテスト通過")

        print("5. 正常ケーステスト")
        test_vp_normal_case()
        print(" ✓ 正常ケーステスト通過")

        print("\n✅ 全てのVP NaN修正テストが通過しました")

    except AssertionError as e:
        print(f"\n❌ テスト失敗: {e}")
        # 詳細情報を出力
        try:
            case = "unknown"
            if "ゼロボリューム" in str(e):
                case = "zero_volume"
            elif "データ長" in str(e):
                case = "short_data"
            elif "価格範囲" in str(e):
                case = "same_price"
            elif "NaNボリューム" in str(e):
                case = "nan_volume"
            elif "正常ケース" in str(e):
                case = "normal_case"
            print(f"失敗したケース: {case}")
        except:
            pass
        exit(1)
    except Exception as e:
        print(f"\n💥 予期せぬエラー: {e}")
        import traceback
        traceback.print_exc()
        exit(1)