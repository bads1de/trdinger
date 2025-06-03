"""
高度なトレンド系指標のテスト

KAMA、T3、TEMA、DEMA指標のテストを実行します。
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
    def kama(data: pd.Series, period: int = 30) -> pd.Series:
        """Kaufman Adaptive Moving Average (KAMA) を計算"""
        TALibAdapter._validate_input(data, period)

        try:
            result = TALibAdapter._safe_talib_calculation(
                talib.KAMA, data.values, timeperiod=period
            )
            return pd.Series(result, index=data.index, name=f"KAMA_{period}")
        except TALibCalculationError:
            raise
        except Exception as e:
            raise TALibCalculationError(f"KAMA計算失敗: {e}")

    @staticmethod
    def t3(data: pd.Series, period: int = 5, vfactor: float = 0.7) -> pd.Series:
        """Triple Exponential Moving Average (T3) を計算"""
        TALibAdapter._validate_input(data, period)

        try:
            result = TALibAdapter._safe_talib_calculation(
                talib.T3, data.values, timeperiod=period, vfactor=vfactor
            )
            return pd.Series(result, index=data.index, name=f"T3_{period}")
        except TALibCalculationError:
            raise
        except Exception as e:
            raise TALibCalculationError(f"T3計算失敗: {e}")

    @staticmethod
    def tema(data: pd.Series, period: int = 30) -> pd.Series:
        """Triple Exponential Moving Average (TEMA) を計算"""
        TALibAdapter._validate_input(data, period)

        try:
            result = TALibAdapter._safe_talib_calculation(
                talib.TEMA, data.values, timeperiod=period
            )
            return pd.Series(result, index=data.index, name=f"TEMA_{period}")
        except TALibCalculationError:
            raise
        except Exception as e:
            raise TALibCalculationError(f"TEMA計算失敗: {e}")

    @staticmethod
    def dema(data: pd.Series, period: int = 30) -> pd.Series:
        """Double Exponential Moving Average (DEMA) を計算"""
        TALibAdapter._validate_input(data, period)

        try:
            result = TALibAdapter._safe_talib_calculation(
                talib.DEMA, data.values, timeperiod=period
            )
            return pd.Series(result, index=data.index, name=f"DEMA_{period}")
        except TALibCalculationError:
            raise
        except Exception as e:
            raise TALibCalculationError(f"DEMA計算失敗: {e}")


def test_advanced_trend_indicators():
    """高度なトレンド系指標のテスト"""
    print("=== 高度なトレンド系TA-Lib指標のテスト ===")
    
    # テストデータの準備
    np.random.seed(42)
    sample_size = 100
    dates = pd.date_range('2023-01-01', periods=sample_size, freq='D')
    
    # より現実的な価格データを生成（トレンドあり）
    base_price = 100
    trend = np.linspace(0, 10, sample_size)  # 上昇トレンド
    noise = np.random.normal(0, 1, sample_size)
    close_prices = base_price + trend + noise
    
    close = pd.Series(close_prices, index=dates)
    
    print(f"テストデータ準備完了: {sample_size}日分")
    print(f"価格範囲: {close.min():.2f} - {close.max():.2f}")
    print(f"トレンド: 上昇（{close.iloc[0]:.2f} → {close.iloc[-1]:.2f}）")
    
    # KAMAテスト
    print("\n--- KAMA (Kaufman Adaptive Moving Average) ---")
    try:
        kama = TALibAdapter.kama(close, 30)
        valid_kama = kama.dropna()
        print(f"✅ KAMA計算成功")
        print(f"   有効値数: {len(valid_kama)}/{len(kama)}")
        if len(valid_kama) > 0:
            print(f"   値域: {valid_kama.min():.2f} - {valid_kama.max():.2f}")
            latest_kama = kama.iloc[-1]
            latest_str = f"{latest_kama:.2f}" if not pd.isna(latest_kama) else "NaN"
            print(f"   最新値: {latest_str}")
            print(f"   価格との差: {abs(close.iloc[-1] - latest_kama):.2f}" if not pd.isna(latest_kama) else "N/A")
    except Exception as e:
        print(f"❌ KAMAエラー: {e}")
    
    # T3テスト
    print("\n--- T3 (Triple Exponential Moving Average) ---")
    try:
        t3 = TALibAdapter.t3(close, 14, 0.7)
        valid_t3 = t3.dropna()
        print(f"✅ T3計算成功")
        print(f"   有効値数: {len(valid_t3)}/{len(t3)}")
        if len(valid_t3) > 0:
            print(f"   値域: {valid_t3.min():.2f} - {valid_t3.max():.2f}")
            latest_t3 = t3.iloc[-1]
            latest_str = f"{latest_t3:.2f}" if not pd.isna(latest_t3) else "NaN"
            print(f"   最新値: {latest_str}")
            print(f"   価格との差: {abs(close.iloc[-1] - latest_t3):.2f}" if not pd.isna(latest_t3) else "N/A")
    except Exception as e:
        print(f"❌ T3エラー: {e}")
    
    # TEMAテスト
    print("\n--- TEMA (Triple Exponential Moving Average) ---")
    try:
        tema = TALibAdapter.tema(close, 21)
        valid_tema = tema.dropna()
        print(f"✅ TEMA計算成功")
        print(f"   有効値数: {len(valid_tema)}/{len(tema)}")
        if len(valid_tema) > 0:
            print(f"   値域: {valid_tema.min():.2f} - {valid_tema.max():.2f}")
            latest_tema = tema.iloc[-1]
            latest_str = f"{latest_tema:.2f}" if not pd.isna(latest_tema) else "NaN"
            print(f"   最新値: {latest_str}")
            print(f"   価格との差: {abs(close.iloc[-1] - latest_tema):.2f}" if not pd.isna(latest_tema) else "N/A")
    except Exception as e:
        print(f"❌ TEMAエラー: {e}")
    
    # DEMAテスト
    print("\n--- DEMA (Double Exponential Moving Average) ---")
    try:
        dema = TALibAdapter.dema(close, 21)
        valid_dema = dema.dropna()
        print(f"✅ DEMA計算成功")
        print(f"   有効値数: {len(valid_dema)}/{len(dema)}")
        if len(valid_dema) > 0:
            print(f"   値域: {valid_dema.min():.2f} - {valid_dema.max():.2f}")
            latest_dema = dema.iloc[-1]
            latest_str = f"{latest_dema:.2f}" if not pd.isna(latest_dema) else "NaN"
            print(f"   最新値: {latest_str}")
            print(f"   価格との差: {abs(close.iloc[-1] - latest_dema):.2f}" if not pd.isna(latest_dema) else "N/A")
    except Exception as e:
        print(f"❌ DEMAエラー: {e}")
    
    # エラーハンドリングテスト
    print("\n--- エラーハンドリングテスト ---")
    try:
        # 期間過大テスト（KAMA）
        try:
            TALibAdapter.kama(close[:10], 30)
            print("❌ KAMA期間過大エラーが検出されませんでした")
        except TALibCalculationError:
            print("✅ KAMA期間過大エラーが正しく検出されました")
        
        # 期間過大テスト（TEMA）
        try:
            TALibAdapter.tema(close[:10], 21)
            print("❌ TEMA期間過大エラーが検出されませんでした")
        except TALibCalculationError:
            print("✅ TEMA期間過大エラーが正しく検出されました")
            
    except Exception as e:
        print(f"❌ エラーハンドリングテストエラー: {e}")
    
    # 移動平均の比較テスト
    print("\n--- 移動平均の応答性比較テスト ---")
    try:
        # 同じ期間で各移動平均を計算
        period = 14
        kama_14 = TALibAdapter.kama(close, period)
        t3_14 = TALibAdapter.t3(close, period, 0.7)
        tema_14 = TALibAdapter.tema(close, period)
        dema_14 = TALibAdapter.dema(close, period)
        
        # 有効なデータのみで比較
        comparison_data = pd.DataFrame({
            'close': close,
            'kama': kama_14,
            't3': t3_14,
            'tema': tema_14,
            'dema': dema_14
        }).dropna()
        
        if len(comparison_data) > 10:
            print(f"✅ 移動平均比較計算成功（{len(comparison_data)}データポイント）")
            
            # 最新値での比較
            latest = comparison_data.iloc[-1]
            print(f"   最新価格: {latest['close']:.2f}")
            print(f"   KAMA: {latest['kama']:.2f} (差: {abs(latest['close'] - latest['kama']):.2f})")
            print(f"   T3: {latest['t3']:.2f} (差: {abs(latest['close'] - latest['t3']):.2f})")
            print(f"   TEMA: {latest['tema']:.2f} (差: {abs(latest['close'] - latest['tema']):.2f})")
            print(f"   DEMA: {latest['dema']:.2f} (差: {abs(latest['close'] - latest['dema']):.2f})")
            
            # 価格追従性の評価（標準偏差）
            price_diff = {
                'KAMA': np.std(comparison_data['close'] - comparison_data['kama']),
                'T3': np.std(comparison_data['close'] - comparison_data['t3']),
                'TEMA': np.std(comparison_data['close'] - comparison_data['tema']),
                'DEMA': np.std(comparison_data['close'] - comparison_data['dema'])
            }
            
            print("\n   価格追従性（標準偏差、小さいほど良い）:")
            for name, std in sorted(price_diff.items(), key=lambda x: x[1]):
                print(f"   {name}: {std:.3f}")
                
        else:
            print("⚠️ 比較計算に十分なデータがありません")
            
    except Exception as e:
        print(f"❌ 比較テストエラー: {e}")
    
    print("\n=== テスト完了 ===")


if __name__ == "__main__":
    test_advanced_trend_indicators()
