"""
ボラティリティ系指標のテスト

NATR、TRANGE指標のテストを実行します。
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import talib

# TALibAdapterクラスのコードを直接コピーしてテスト
class TALibCalculationError(Exception):
    """TA-Lib計算エラー"""
    pass

class TALibAdapter:
    """TA-Libと既存システムの橋渡しクラス"""

    @staticmethod
    def _validate_input(data: pd.Series, period: int) -> None:
        """入力データとパラメータの検証"""
        if data is None or len(data) == 0:
            raise TALibCalculationError("入力データが空です")

        if period <= 0:
            raise TALibCalculationError(f"期間は正の整数である必要があります: {period}")

        if len(data) < period:
            raise TALibCalculationError(
                f"データ長({len(data)})が期間({period})より短いです"
            )

    @staticmethod
    def _safe_talib_calculation(func, *args, **kwargs) -> np.ndarray:
        """TA-Lib計算の安全な実行"""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            raise TALibCalculationError(f"TA-Lib計算エラー: {e}")

    @staticmethod
    def natr(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
        """Normalized Average True Range (NATR) を計算"""
        if not (len(high) == len(low) == len(close)):
            raise TALibCalculationError("高値、安値、終値のデータ長が一致しません")

        TALibAdapter._validate_input(close, period)

        try:
            result = TALibAdapter._safe_talib_calculation(
                talib.NATR, high.values, low.values, close.values, timeperiod=period
            )
            return pd.Series(result, index=close.index, name=f"NATR_{period}")
        except TALibCalculationError:
            raise
        except Exception as e:
            raise TALibCalculationError(f"NATR計算失敗: {e}")

    @staticmethod
    def trange(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """True Range (TRANGE) を計算"""
        if not (len(high) == len(low) == len(close)):
            raise TALibCalculationError("高値、安値、終値のデータ長が一致しません")

        if len(close) == 0:
            raise TALibCalculationError("入力データが空です")

        try:
            result = TALibAdapter._safe_talib_calculation(
                talib.TRANGE, high.values, low.values, close.values
            )
            return pd.Series(result, index=close.index, name="TRANGE")
        except TALibCalculationError:
            raise
        except Exception as e:
            raise TALibCalculationError(f"TRANGE計算失敗: {e}")

    @staticmethod
    def atr(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
        """Average True Range (ATR) を計算（比較用）"""
        if not (len(high) == len(low) == len(close)):
            raise TALibCalculationError("高値、安値、終値のデータ長が一致しません")

        TALibAdapter._validate_input(close, period)

        try:
            result = TALibAdapter._safe_talib_calculation(
                talib.ATR, high.values, low.values, close.values, timeperiod=period
            )
            return pd.Series(result, index=close.index, name=f"ATR_{period}")
        except TALibCalculationError:
            raise
        except Exception as e:
            raise TALibCalculationError(f"ATR計算失敗: {e}")


def test_volatility_indicators():
    """ボラティリティ系指標のテスト"""
    print("=== ボラティリティ系TA-Lib指標のテスト ===")
    
    # テストデータの準備（ボラティリティの変化を含む）
    np.random.seed(42)
    sample_size = 100
    dates = pd.date_range('2023-01-01', periods=sample_size, freq='D')
    
    # ボラティリティが変化する価格データを生成
    base_price = 100
    trend = np.linspace(0, 5, sample_size)
    
    # 前半は低ボラティリティ、後半は高ボラティリティ
    volatility = np.concatenate([
        np.full(50, 0.5),  # 低ボラティリティ期間
        np.full(50, 2.0)   # 高ボラティリティ期間
    ])
    
    noise = np.random.normal(0, volatility)
    close_prices = base_price + trend + noise
    
    # OHLC データを生成
    high_prices = close_prices + np.random.uniform(0, volatility * 0.5, sample_size)
    low_prices = close_prices - np.random.uniform(0, volatility * 0.5, sample_size)
    
    high = pd.Series(high_prices, index=dates)
    low = pd.Series(low_prices, index=dates)
    close = pd.Series(close_prices, index=dates)
    
    print(f"テストデータ準備完了: {sample_size}日分")
    print(f"価格範囲: {close.min():.2f} - {close.max():.2f}")
    print(f"前半ボラティリティ: 低（0.5）、後半ボラティリティ: 高（2.0）")
    
    # NATRテスト
    print("\n--- NATR (Normalized Average True Range) ---")
    try:
        natr = TALibAdapter.natr(high, low, close, 14)
        valid_natr = natr.dropna()
        print(f"✅ NATR計算成功")
        print(f"   有効値数: {len(valid_natr)}/{len(natr)}")
        if len(valid_natr) > 0:
            print(f"   値域: {valid_natr.min():.3f}% - {valid_natr.max():.3f}%")
            latest_natr = natr.iloc[-1]
            latest_str = f"{latest_natr:.3f}%" if not pd.isna(latest_natr) else "NaN"
            print(f"   最新値: {latest_str}")
            
            # 前半と後半のNATR平均を比較
            if len(valid_natr) > 30:
                mid_point = len(natr) // 2
                first_half = natr.iloc[:mid_point].dropna()
                second_half = natr.iloc[mid_point:].dropna()
                
                if len(first_half) > 0 and len(second_half) > 0:
                    print(f"   前半平均: {first_half.mean():.3f}%")
                    print(f"   後半平均: {second_half.mean():.3f}%")
                    print(f"   ボラティリティ変化検出: {'✅' if second_half.mean() > first_half.mean() else '❌'}")
                    
    except Exception as e:
        print(f"❌ NATRエラー: {e}")
    
    # TRANGEテスト
    print("\n--- TRANGE (True Range) ---")
    try:
        trange = TALibAdapter.trange(high, low, close)
        valid_trange = trange.dropna()
        print(f"✅ TRANGE計算成功")
        print(f"   有効値数: {len(valid_trange)}/{len(trange)}")
        if len(valid_trange) > 0:
            print(f"   値域: {valid_trange.min():.3f} - {valid_trange.max():.3f}")
            latest_trange = trange.iloc[-1]
            latest_str = f"{latest_trange:.3f}" if not pd.isna(latest_trange) else "NaN"
            print(f"   最新値: {latest_str}")
            
            # 前半と後半のTRANGE平均を比較
            mid_point = len(trange) // 2
            first_half = trange.iloc[:mid_point].dropna()
            second_half = trange.iloc[mid_point:].dropna()
            
            if len(first_half) > 0 and len(second_half) > 0:
                print(f"   前半平均: {first_half.mean():.3f}")
                print(f"   後半平均: {second_half.mean():.3f}")
                print(f"   ボラティリティ変化検出: {'✅' if second_half.mean() > first_half.mean() else '❌'}")
                
    except Exception as e:
        print(f"❌ TRANGEエラー: {e}")
    
    # エラーハンドリングテスト
    print("\n--- エラーハンドリングテスト ---")
    try:
        # データ長不一致テスト（NATR）
        try:
            TALibAdapter.natr(high[:50], low, close, 14)
            print("❌ NATRデータ長不一致エラーが検出されませんでした")
        except TALibCalculationError:
            print("✅ NATRデータ長不一致エラーが正しく検出されました")
        
        # 期間過大テスト（NATR）
        try:
            TALibAdapter.natr(high[:10], low[:10], close[:10], 14)
            print("❌ NATR期間過大エラーが検出されませんでした")
        except TALibCalculationError:
            print("✅ NATR期間過大エラーが正しく検出されました")
        
        # データ長不一致テスト（TRANGE）
        try:
            TALibAdapter.trange(high[:50], low, close)
            print("❌ TRANGEデータ長不一致エラーが検出されませんでした")
        except TALibCalculationError:
            print("✅ TRANGEデータ長不一致エラーが正しく検出されました")
            
    except Exception as e:
        print(f"❌ エラーハンドリングテストエラー: {e}")
    
    # ATRとNATRの比較テスト
    print("\n--- ATR vs NATR 比較テスト ---")
    try:
        atr = TALibAdapter.atr(high, low, close, 14)
        natr_comp = TALibAdapter.natr(high, low, close, 14)
        
        # 有効なデータのみで比較
        comparison_data = pd.DataFrame({
            'close': close,
            'atr': atr,
            'natr': natr_comp
        }).dropna()
        
        if len(comparison_data) > 10:
            print(f"✅ ATR vs NATR比較計算成功（{len(comparison_data)}データポイント）")
            
            # 最新値での比較
            latest = comparison_data.iloc[-1]
            print(f"   最新価格: {latest['close']:.2f}")
            print(f"   ATR: {latest['atr']:.3f}")
            print(f"   NATR: {latest['natr']:.3f}%")
            
            # 理論的関係の確認: NATR ≈ (ATR / Close) * 100
            theoretical_natr = (latest['atr'] / latest['close']) * 100
            print(f"   理論NATR: {theoretical_natr:.3f}%")
            print(f"   差異: {abs(latest['natr'] - theoretical_natr):.3f}%")
            
            # 相関係数
            correlation = comparison_data['atr'].corr(comparison_data['natr'])
            print(f"   ATR-NATR相関: {correlation:.3f}")
            
        else:
            print("⚠️ 比較計算に十分なデータがありません")
            
    except Exception as e:
        print(f"❌ 比較テストエラー: {e}")
    
    # ボラティリティ変化の検出性能テスト
    print("\n--- ボラティリティ変化検出性能テスト ---")
    try:
        natr_result = TALibAdapter.natr(high, low, close, 14)
        trange_result = TALibAdapter.trange(high, low, close)
        atr_result = TALibAdapter.atr(high, low, close, 14)
        
        # 前半と後半で分割
        mid_point = len(close) // 2
        
        indicators = {
            'NATR': natr_result,
            'TRANGE': trange_result,
            'ATR': atr_result
        }
        
        print("   ボラティリティ変化検出結果:")
        for name, indicator in indicators.items():
            first_half = indicator.iloc[:mid_point].dropna()
            second_half = indicator.iloc[mid_point:].dropna()
            
            if len(first_half) > 5 and len(second_half) > 5:
                ratio = second_half.mean() / first_half.mean()
                print(f"   {name}: 後半/前半比 = {ratio:.2f} ({'✅' if ratio > 1.5 else '❌'})")
            else:
                print(f"   {name}: データ不足")
                
    except Exception as e:
        print(f"❌ 変化検出テストエラー: {e}")
    
    print("\n=== テスト完了 ===")


if __name__ == "__main__":
    test_volatility_indicators()
