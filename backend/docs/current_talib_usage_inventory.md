# 現在のコードベースにおけるTA-lib使用状況の詳細調査

このドキュメントは、現在のコードベースで実際に使用されているTA-lib関数の完全なリストです。

## 使用箇所の概要

### 1. テクニカル指標モジュール (`backend/app/services/indicators/technical_indicators/`)

#### trend.py - トレンド系指標
- `talib.SMA(data, timeperiod=period)` - 単純移動平均
- `talib.EMA(data, timeperiod=period)` - 指数移動平均
- `talib.TEMA(data, timeperiod=period)` - 三重指数移動平均
- `talib.DEMA(data, timeperiod=period)` - 二重指数移動平均
- `talib.WMA(data, timeperiod=period)` - 加重移動平均
- `talib.TRIMA(data, timeperiod=period)` - 三角移動平均
- `talib.KAMA(data, timeperiod=period)` - 適応移動平均
- `talib.MAMA(data, fastlimit=fastlimit, slowlimit=slowlimit)` - MESA適応移動平均
- `talib.T3(data, timeperiod=period, vfactor=vfactor)` - T3移動平均
- `talib.SAR(high, low, acceleration=acceleration, maximum=maximum)` - パラボリックSAR
- `talib.SAREXT(high, low, **params)` - 拡張パラボリックSAR
- `talib.HT_TRENDLINE(data)` - ヒルベルト変換トレンドライン
- `talib.MA(data, timeperiod=period, matype=matype)` - 移動平均（タイプ指定）

#### momentum.py - モメンタム系指標
- `talib.RSI(data, timeperiod=period)` - 相対力指数
- `talib.MACD(data, fastperiod=fast, slowperiod=slow, signalperiod=signal)` - MACD
- `talib.STOCH(high, low, close, fastk_period=fastk, slowk_period=slowk, slowk_matype=slowk_matype, slowd_period=slowd, slowd_matype=slowd_matype)` - ストキャスティクス
- `talib.STOCHF(high, low, close, fastk_period=fastk, fastd_period=fastd, fastd_matype=fastd_matype)` - 高速ストキャスティクス
- `talib.STOCHRSI(data, timeperiod=period, fastk_period=fastk, fastd_period=fastd, fastd_matype=fastd_matype)` - ストキャスティクスRSI
- `talib.CCI(high, low, close, timeperiod=period)` - 商品チャネル指数
- `talib.CMO(data, timeperiod=period)` - チャンデモメンタムオシレーター
- `talib.MOM(data, timeperiod=period)` - モメンタム
- `talib.ADX(high, low, close, timeperiod=period)` - 平均方向性指数
- `talib.ADXR(high, low, close, timeperiod=period)` - ADX評価
- `talib.APO(data, fastperiod=fast, slowperiod=slow, matype=matype)` - 絶対価格オシレーター
- `talib.AROON(high, low, timeperiod=period)` - アルーン
- `talib.AROONOSC(high, low, timeperiod=period)` - アルーンオシレーター
- `talib.BOP(open, high, low, close)` - バランスオブパワー
- `talib.DX(high, low, close, timeperiod=period)` - 方向性指数
- `talib.MFI(high, low, close, volume, timeperiod=period)` - マネーフローインデックス
- `talib.MINUS_DI(high, low, close, timeperiod=period)` - マイナス方向性指標
- `talib.MINUS_DM(high, low, timeperiod=period)` - マイナス方向性移動
- `talib.PLUS_DI(high, low, close, timeperiod=period)` - プラス方向性指標
- `talib.PLUS_DM(high, low, timeperiod=period)` - プラス方向性移動
- `talib.PPO(data, fastperiod=fast, slowperiod=slow, matype=matype)` - パーセント価格オシレーター
- `talib.ROC(data, timeperiod=period)` - 変化率
- `talib.ROCP(data, timeperiod=period)` - 変化率（パーセント）
- `talib.ROCR(data, timeperiod=period)` - 変化率（比率）
- `talib.ROCR100(data, timeperiod=period)` - 変化率（比率×100）
- `talib.TRIX(data, timeperiod=period)` - TRIX
- `talib.ULTOSC(high, low, close, timeperiod1=period1, timeperiod2=period2, timeperiod3=period3)` - アルティメットオシレーター
- `talib.WILLR(high, low, close, timeperiod=period)` - ウィリアムズ%R

#### volatility.py - ボラティリティ系指標
- `talib.ATR(high, low, close, timeperiod=period)` - 平均真の値幅
- `talib.NATR(high, low, close, timeperiod=period)` - 正規化ATR
- `talib.TRANGE(high, low, close)` - 真の値幅
- `talib.BBANDS(data, timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev, matype=matype)` - ボリンジャーバンド
- `talib.STDDEV(data, timeperiod=period, nbdev=nbdev)` - 標準偏差
- `talib.VAR(data, timeperiod=period, nbdev=nbdev)` - 分散

#### volume.py - 出来高系指標
- `talib.AD(high, low, close, volume)` - 蓄積/分配ライン
- `talib.ADOSC(high, low, close, volume, fastperiod=fast, slowperiod=slow)` - チャイキンA/Dオシレーター
- `talib.OBV(close, volume)` - オンバランスボリューム

#### cycle.py - サイクル系指標
- `talib.HT_DCPERIOD(data)` - ヒルベルト変換支配的サイクル期間
- `talib.HT_DCPHASE(data)` - ヒルベルト変換支配的サイクル位相
- `talib.HT_PHASOR(data)` - ヒルベルト変換フェーザー成分
- `talib.HT_SINE(data)` - ヒルベルト変換サイン波
- `talib.HT_TRENDMODE(data)` - ヒルベルト変換トレンドモード

#### statistics.py - 統計系指標
- `talib.BETA(high, low, timeperiod=period)` - ベータ
- `talib.CORREL(high, low, timeperiod=period)` - ピアソン相関係数
- `talib.LINEARREG(data, timeperiod=period)` - 線形回帰
- `talib.LINEARREG_ANGLE(data, timeperiod=period)` - 線形回帰角度
- `talib.LINEARREG_INTERCEPT(data, timeperiod=period)` - 線形回帰切片
- `talib.LINEARREG_SLOPE(data, timeperiod=period)` - 線形回帰傾き
- `talib.TSF(data, timeperiod=period)` - 時系列予測

#### math_transform.py - 数学変換系指標
- `talib.ACOS(data)` - 逆余弦
- `talib.ASIN(data)` - 逆正弦
- `talib.ATAN(data)` - 逆正接
- `talib.CEIL(data)` - 天井関数
- `talib.COS(data)` - 余弦
- `talib.COSH(data)` - 双曲線余弦
- `talib.EXP(data)` - 指数関数
- `talib.FLOOR(data)` - 床関数
- `talib.LN(data)` - 自然対数
- `talib.LOG10(data)` - 常用対数
- `talib.SIN(data)` - 正弦
- `talib.SINH(data)` - 双曲線正弦
- `talib.SQRT(data)` - 平方根
- `talib.TAN(data)` - 正接
- `talib.TANH(data)` - 双曲線正接

#### math_operators.py - 数学演算子系指標
- `talib.ADD(data0, data1)` - 加算
- `talib.SUB(data0, data1)` - 減算
- `talib.MULT(data0, data1)` - 乗算
- `talib.DIV(data0, data1)` - 除算
- `talib.MAX(data, timeperiod=period)` - 最大値
- `talib.MAXINDEX(data, timeperiod=period)` - 最大値インデックス
- `talib.MIN(data, timeperiod=period)` - 最小値
- `talib.MININDEX(data, timeperiod=period)` - 最小値インデックス
- `talib.MINMAX(data, timeperiod=period)` - 最小最大値
- `talib.MINMAXINDEX(data, timeperiod=period)` - 最小最大値インデックス
- `talib.SUM(data, timeperiod=period)` - 合計

#### price_transform.py - 価格変換系指標
- `talib.AVGPRICE(open, high, low, close)` - 平均価格
- `talib.MEDPRICE(high, low)` - 中央価格
- `talib.TYPPRICE(high, low, close)` - 典型価格
- `talib.WCLPRICE(high, low, close)` - 加重終値価格

#### pattern_recognition.py - パターン認識系指標
- `talib.CDLDOJI(open, high, low, close)` - 十字線
- `talib.CDLHAMMER(open, high, low, close)` - ハンマー
- `talib.CDLSHOOTINGSTAR(open, high, low, close)` - 流れ星
- その他多数のローソク足パターン認識関数

### 2. 機械学習特徴量エンジニアリング (`backend/app/services/ml/feature_engineering/`)

#### technical_features.py
- `talib.SMA(close_vals, timeperiod=short_ma)` - 短期移動平均
- `talib.SMA(close_vals, timeperiod=long_ma)` - 長期移動平均
- `talib.RSI(close_values, timeperiod=14)` - RSI
- `talib.MACD(close_values, fastperiod=12, slowperiod=26, signalperiod=9)` - MACD
- `talib.LINEARREG_SLOPE(close_values, timeperiod=10)` - 線形回帰傾き
- `talib.LINEARREG_SLOPE(rsi_values, timeperiod=10)` - RSI線形回帰傾き

#### advanced_features.py
- `talib.OBV(close, volume)` - オンバランスボリューム
- `talib.AD(high, low, close, volume)` - 蓄積/分配ライン
- `talib.ADOSC(high, low, close, volume)` - チャイキンA/Dオシレーター
- `talib.CDLDOJI(open, high, low, close)` - 十字線パターン
- `talib.CDLHAMMER(open, high, low, close)` - ハンマーパターン
- `talib.CDLSHOOTINGSTAR(open, high, low, close)` - 流れ星パターン

#### enhanced_crypto_features.py
- `talib.RSI(close_values, timeperiod=period)` - RSI（複数期間）

### 3. 適応学習モジュール (`backend/app/services/ml/adaptive_learning/`)

#### market_regime_detector.py
- `talib.BBANDS(close_values, timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev, matype=0)` - ボリンジャーバンド

## 移行優先度の分類

### 高優先度（基本指標）
1. **SMA, EMA** - 最も基本的な移動平均
2. **RSI** - 広く使用されるモメンタム指標
3. **MACD** - 重要なトレンド・モメンタム指標
4. **ATR** - ボラティリティ測定の基本
5. **BBANDS** - ボリンジャーバンド

### 中優先度（複雑指標）
1. **Stochastic系** - STOCH, STOCHF, STOCHRSI
2. **ADX系** - ADX, ADXR, DX, PLUS_DI, MINUS_DI, PLUS_DM, MINUS_DM
3. **出来高系** - OBV, AD, ADOSC
4. **その他モメンタム** - CCI, CMO, MOM, MFI, WILLR

### 低優先度（特殊指標）
1. **数学変換系** - ACOS, ASIN, SQRT等（NumPyで代替可能）
2. **数学演算子系** - ADD, SUB, MULT, DIV等（NumPy/pandasで代替可能）
3. **パターン認識系** - CDLDOJI等（独自実装が必要）
4. **ヒルベルト変換系** - HT_*系関数

## 移行時の技術的考慮事項

### 1. 戻り値の形式変更
- **TA-lib**: numpy.ndarray
- **pandas-ta**: pandas.Series または pandas.DataFrame

### 2. パラメータ名の変更
- `timeperiod` → `length`
- `fastperiod/slowperiod` → `fast/slow`
- `nbdevup/nbdevdn` → `std`

### 3. 複数戻り値の処理
- **MACD**: 3つの値（MACD, Signal, Histogram）
- **BBANDS**: 3つの値（Upper, Middle, Lower）
- **STOCH**: 2つの値（%K, %D）

### 4. backtesting.py互換性
- numpy配列への変換が必要
- `.values`プロパティを使用

### 5. エラーハンドリング
- pandas-ta用の新しいエラーハンドリング実装が必要
- NaN処理の統一

## 推定作業量

- **基本指標移行**: 約20関数 - 2-3日
- **複雑指標移行**: 約30関数 - 3-4日
- **数学系移行**: 約25関数 - 1-2日
- **テスト・検証**: 全体 - 2-3日
- **ドキュメント更新**: 1日

**総計**: 約9-13日の作業量
