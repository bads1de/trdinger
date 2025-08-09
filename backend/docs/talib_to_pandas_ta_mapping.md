# TA-lib から pandas-ta への関数対応表

このドキュメントは、現在のコードベースで使用されているTA-lib関数とpandas-taの対応関数のマッピングを示します。

## 基本的な使用方法の違い

### TA-lib
```python
import talib
import numpy as np

# numpy配列を使用
close = np.array([...])
sma = talib.SMA(close, timeperiod=20)
```

### pandas-ta
```python
import pandas_ta as ta
import pandas as pd

# pandas DataFrameまたはSeriesを使用
df = pd.DataFrame({'Close': [...]})
sma = ta.sma(df['Close'], length=20)
# または
df.ta.sma(length=20, append=True)  # DataFrameに直接追加
```

## トレンド系指標 (Trend Indicators)

| TA-lib関数 | pandas-ta関数 | パラメータ対応 | 備考 |
|-----------|---------------|---------------|------|
| `talib.SMA(data, timeperiod)` | `ta.sma(data, length)` | timeperiod → length | 単純移動平均 |
| `talib.EMA(data, timeperiod)` | `ta.ema(data, length)` | timeperiod → length | 指数移動平均 |
| `talib.TEMA(data, timeperiod)` | `ta.tema(data, length)` | timeperiod → length | 三重指数移動平均 |
| `talib.DEMA(data, timeperiod)` | `ta.dema(data, length)` | timeperiod → length | 二重指数移動平均 |
| `talib.WMA(data, timeperiod)` | `ta.wma(data, length)` | timeperiod → length | 加重移動平均 |
| `talib.TRIMA(data, timeperiod)` | `ta.trima(data, length)` | timeperiod → length | 三角移動平均 |
| `talib.KAMA(data, timeperiod)` | `ta.kama(data, length)` | timeperiod → length | 適応移動平均 |
| `talib.MAMA(data, fastlimit, slowlimit)` | `ta.mama(data, fastlimit, slowlimit)` | 同じ | MESA適応移動平均 |
| `talib.T3(data, timeperiod, vfactor)` | `ta.t3(data, length, vfactor)` | timeperiod → length | T3移動平均 |
| `talib.SAR(high, low, acceleration, maximum)` | `ta.psar(high, low, af0, af, max_af)` | パラメータ名変更 | パラボリックSAR |
| `talib.SAREXT(high, low, ...)` | `ta.psar(high, low, ...)` | 拡張パラメータ対応 | 拡張パラボリックSAR |
| `talib.HT_TRENDLINE(data)` | `ta.ht_trendline(data)` | 同じ | ヒルベルト変換トレンドライン |
| `talib.MA(data, timeperiod, matype)` | 複数関数で対応 | matypeに応じて分岐 | 移動平均（タイプ指定） |

## モメンタム系指標 (Momentum Indicators)

| TA-lib関数 | pandas-ta関数 | パラメータ対応 | 備考 |
|-----------|---------------|---------------|------|
| `talib.RSI(data, timeperiod)` | `ta.rsi(data, length)` | timeperiod → length | 相対力指数 |
| `talib.MACD(data, fastperiod, slowperiod, signalperiod)` | `ta.macd(data, fast, slow, signal)` | パラメータ名変更 | MACD |
| `talib.STOCH(high, low, close, fastk_period, slowk_period, slowk_matype, slowd_period, slowd_matype)` | `ta.stoch(high, low, close, k, d, smooth_k)` | パラメータ簡略化 | ストキャスティクス |
| `talib.STOCHF(high, low, close, fastk_period, fastd_period, fastd_matype)` | `ta.stochf(high, low, close, fastk_period, fastd_period)` | 同じ | 高速ストキャスティクス |
| `talib.STOCHRSI(data, timeperiod, fastk_period, fastd_period, fastd_matype)` | `ta.stochrsi(data, length, rsi_length, k, d)` | パラメータ名変更 | ストキャスティクスRSI |
| `talib.CCI(high, low, close, timeperiod)` | `ta.cci(high, low, close, length)` | timeperiod → length | 商品チャネル指数 |
| `talib.CMO(data, timeperiod)` | `ta.cmo(data, length)` | timeperiod → length | チャンデモメンタムオシレーター |
| `talib.MOM(data, timeperiod)` | `ta.mom(data, length)` | timeperiod → length | モメンタム |
| `talib.ADX(high, low, close, timeperiod)` | `ta.adx(high, low, close, length)` | timeperiod → length | 平均方向性指数 |
| `talib.ADXR(high, low, close, timeperiod)` | `ta.adx(high, low, close, length)` | ADXの一部として計算 | ADX評価 |
| `talib.APO(data, fastperiod, slowperiod, matype)` | `ta.apo(data, fast, slow)` | パラメータ名変更 | 絶対価格オシレーター |
| `talib.AROON(high, low, timeperiod)` | `ta.aroon(high, low, length)` | timeperiod → length | アルーン |
| `talib.AROONOSC(high, low, timeperiod)` | `ta.aroon(high, low, length)` | アルーンの一部として計算 | アルーンオシレーター |
| `talib.BOP(open, high, low, close)` | `ta.bop(open, high, low, close)` | 同じ | バランスオブパワー |
| `talib.DX(high, low, close, timeperiod)` | `ta.adx(high, low, close, length)` | ADXの一部として計算 | 方向性指数 |
| `talib.MFI(high, low, close, volume, timeperiod)` | `ta.mfi(high, low, close, volume, length)` | timeperiod → length | マネーフローインデックス |
| `talib.MINUS_DI(high, low, close, timeperiod)` | `ta.adx(high, low, close, length)` | ADXの一部として計算 | マイナス方向性指標 |
| `talib.MINUS_DM(high, low, timeperiod)` | `ta.adx(high, low, close, length)` | ADXの一部として計算 | マイナス方向性移動 |
| `talib.PLUS_DI(high, low, close, timeperiod)` | `ta.adx(high, low, close, length)` | ADXの一部として計算 | プラス方向性指標 |
| `talib.PLUS_DM(high, low, timeperiod)` | `ta.adx(high, low, close, length)` | ADXの一部として計算 | プラス方向性移動 |
| `talib.PPO(data, fastperiod, slowperiod, matype)` | `ta.ppo(data, fast, slow)` | パラメータ名変更 | パーセント価格オシレーター |
| `talib.ROC(data, timeperiod)` | `ta.roc(data, length)` | timeperiod → length | 変化率 |
| `talib.ROCP(data, timeperiod)` | `ta.roc(data, length)` | 同じ計算 | 変化率（パーセント） |
| `talib.ROCR(data, timeperiod)` | `ta.roc(data, length)` | 計算方法調整 | 変化率（比率） |
| `talib.ROCR100(data, timeperiod)` | `ta.roc(data, length)` | 計算方法調整 | 変化率（比率×100） |
| `talib.RSI(data, timeperiod)` | `ta.rsi(data, length)` | timeperiod → length | 相対力指数 |
| `talib.TRIX(data, timeperiod)` | `ta.trix(data, length)` | timeperiod → length | TRIX |
| `talib.ULTOSC(high, low, close, timeperiod1, timeperiod2, timeperiod3)` | `ta.uo(high, low, close, fast, medium, slow)` | パラメータ名変更 | アルティメットオシレーター |
| `talib.WILLR(high, low, close, timeperiod)` | `ta.willr(high, low, close, length)` | timeperiod → length | ウィリアムズ%R |

## ボラティリティ系指標 (Volatility Indicators)

| TA-lib関数 | pandas-ta関数 | パラメータ対応 | 備考 |
|-----------|---------------|---------------|------|
| `talib.ATR(high, low, close, timeperiod)` | `ta.atr(high, low, close, length)` | timeperiod → length | 平均真の値幅 |
| `talib.NATR(high, low, close, timeperiod)` | `ta.natr(high, low, close, length)` | timeperiod → length | 正規化ATR |
| `talib.TRANGE(high, low, close)` | `ta.true_range(high, low, close)` | 同じ | 真の値幅 |
| `talib.BBANDS(data, timeperiod, nbdevup, nbdevdn, matype)` | `ta.bbands(data, length, std)` | パラメータ簡略化 | ボリンジャーバンド |
| `talib.STDDEV(data, timeperiod, nbdev)` | `ta.stdev(data, length)` | timeperiod → length | 標準偏差 |
| `talib.VAR(data, timeperiod, nbdev)` | `ta.variance(data, length)` | timeperiod → length | 分散 |

## 出来高系指標 (Volume Indicators)

| TA-lib関数 | pandas-ta関数 | パラメータ対応 | 備考 |
|-----------|---------------|---------------|------|
| `talib.AD(high, low, close, volume)` | `ta.ad(high, low, close, volume)` | 同じ | 蓄積/分配ライン |
| `talib.ADOSC(high, low, close, volume, fastperiod, slowperiod)` | `ta.adosc(high, low, close, volume, fast, slow)` | パラメータ名変更 | チャイキンA/Dオシレーター |
| `talib.OBV(close, volume)` | `ta.obv(close, volume)` | 同じ | オンバランスボリューム |

## サイクル系指標 (Cycle Indicators)

| TA-lib関数 | pandas-ta関数 | パラメータ対応 | 備考 |
|-----------|---------------|---------------|------|
| `talib.HT_DCPERIOD(data)` | `ta.ht_dcperiod(data)` | 同じ | ヒルベルト変換支配的サイクル期間 |
| `talib.HT_DCPHASE(data)` | `ta.ht_dcphase(data)` | 同じ | ヒルベルト変換支配的サイクル位相 |
| `talib.HT_PHASOR(data)` | `ta.ht_phasor(data)` | 同じ | ヒルベルト変換フェーザー成分 |
| `talib.HT_SINE(data)` | `ta.ht_sine(data)` | 同じ | ヒルベルト変換サイン波 |
| `talib.HT_TRENDMODE(data)` | `ta.ht_trendmode(data)` | 同じ | ヒルベルト変換トレンドモード |

## 統計系指標 (Statistics Indicators)

| TA-lib関数 | pandas-ta関数 | パラメータ対応 | 備考 |
|-----------|---------------|---------------|------|
| `talib.BETA(high, low, timeperiod)` | `ta.beta(high, low, length)` | timeperiod → length | ベータ |
| `talib.CORREL(high, low, timeperiod)` | `ta.correl(high, low, length)` | timeperiod → length | ピアソン相関係数 |
| `talib.LINEARREG(data, timeperiod)` | `ta.linreg(data, length)` | timeperiod → length | 線形回帰 |
| `talib.LINEARREG_ANGLE(data, timeperiod)` | `ta.linreg(data, length)` | 線形回帰の一部として計算 | 線形回帰角度 |
| `talib.LINEARREG_INTERCEPT(data, timeperiod)` | `ta.linreg(data, length)` | 線形回帰の一部として計算 | 線形回帰切片 |
| `talib.LINEARREG_SLOPE(data, timeperiod)` | `ta.linreg(data, length)` | 線形回帰の一部として計算 | 線形回帰傾き |
| `talib.TSF(data, timeperiod)` | `ta.tsf(data, length)` | timeperiod → length | 時系列予測 |

## 数学変換系指標 (Math Transform)

| TA-lib関数 | pandas-ta関数 | 代替方法 | 備考 |
|-----------|---------------|----------|------|
| `talib.ACOS(data)` | `np.arccos(data)` | NumPy使用 | 逆余弦 |
| `talib.ASIN(data)` | `np.arcsin(data)` | NumPy使用 | 逆正弦 |
| `talib.ATAN(data)` | `np.arctan(data)` | NumPy使用 | 逆正接 |
| `talib.CEIL(data)` | `np.ceil(data)` | NumPy使用 | 天井関数 |
| `talib.COS(data)` | `np.cos(data)` | NumPy使用 | 余弦 |
| `talib.COSH(data)` | `np.cosh(data)` | NumPy使用 | 双曲線余弦 |
| `talib.EXP(data)` | `np.exp(data)` | NumPy使用 | 指数関数 |
| `talib.FLOOR(data)` | `np.floor(data)` | NumPy使用 | 床関数 |
| `talib.LN(data)` | `np.log(data)` | NumPy使用 | 自然対数 |
| `talib.LOG10(data)` | `np.log10(data)` | NumPy使用 | 常用対数 |
| `talib.SIN(data)` | `np.sin(data)` | NumPy使用 | 正弦 |
| `talib.SINH(data)` | `np.sinh(data)` | NumPy使用 | 双曲線正弦 |
| `talib.SQRT(data)` | `np.sqrt(data)` | NumPy使用 | 平方根 |
| `talib.TAN(data)` | `np.tan(data)` | NumPy使用 | 正接 |
| `talib.TANH(data)` | `np.tanh(data)` | NumPy使用 | 双曲線正接 |

## 数学演算子系指標 (Math Operators)

| TA-lib関数 | pandas-ta関数 | 代替方法 | 備考 |
|-----------|---------------|----------|------|
| `talib.ADD(data0, data1)` | `data0 + data1` | NumPy/pandas演算 | 加算 |
| `talib.SUB(data0, data1)` | `data0 - data1` | NumPy/pandas演算 | 減算 |
| `talib.MULT(data0, data1)` | `data0 * data1` | NumPy/pandas演算 | 乗算 |
| `talib.DIV(data0, data1)` | `data0 / data1` | NumPy/pandas演算 | 除算 |
| `talib.MAX(data, timeperiod)` | `ta.max(data, length)` | pandas rolling max | 最大値 |
| `talib.MAXINDEX(data, timeperiod)` | `ta.max(data, length)` | pandas rolling idxmax | 最大値インデックス |
| `talib.MIN(data, timeperiod)` | `ta.min(data, length)` | pandas rolling min | 最小値 |
| `talib.MININDEX(data, timeperiod)` | `ta.min(data, length)` | pandas rolling idxmin | 最小値インデックス |
| `talib.MINMAX(data, timeperiod)` | 複数関数組み合わせ | min/max組み合わせ | 最小最大値 |
| `talib.MINMAXINDEX(data, timeperiod)` | 複数関数組み合わせ | idxmin/idxmax組み合わせ | 最小最大値インデックス |
| `talib.SUM(data, timeperiod)` | `ta.sum(data, length)` | pandas rolling sum | 合計 |

## パターン認識系指標 (Pattern Recognition)

pandas-taにはパターン認識関数が限定的なため、多くはTA-libを継続使用するか、独自実装が必要です。

| TA-lib関数 | pandas-ta関数 | 代替方法 | 備考 |
|-----------|---------------|----------|------|
| `talib.CDLDOJI(open, high, low, close)` | 独自実装 | カスタム関数 | 十字線 |
| `talib.CDLHAMMER(open, high, low, close)` | 独自実装 | カスタム関数 | ハンマー |
| `talib.CDLSHOOTINGSTAR(open, high, low, close)` | 独自実装 | カスタム関数 | 流れ星 |
| その他多数のパターン | 独自実装 | カスタム関数 | 各種ローソク足パターン |

## 移行時の注意点

1. **パラメータ名の変更**: `timeperiod` → `length` など
2. **戻り値の形式**: TA-libはnumpy配列、pandas-taはpandas Series
3. **NaN処理**: pandas-taの方が一貫したNaN処理
4. **複数戻り値**: MACDなどは複数列のDataFrameとして返される
5. **パフォーマンス**: 大量データではTA-libの方が高速な場合がある

## 移行戦略

1. **段階的移行**: 基本指標から順次移行
2. **テスト重視**: 各指標の移行前後で結果の一致性を確認
3. **互換性維持**: backtesting.pyとの互換性を保つためnumpy配列変換を実装
4. **エラーハンドリング**: pandas-ta用の新しいエラーハンドリングを実装
