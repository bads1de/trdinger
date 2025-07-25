# AutoML システム モデル精度低下の詳細分析レポート

## 📊 現状の問題

**現在のモデル精度: 47%**

これは 3 クラス分類（上昇・下落・レンジ）において、ランダム予測（33.3%）をわずかに上回る程度の低い精度です。

## 🔍 根本原因分析

### 1. **ターゲット変数生成の問題**

#### 1.1 閾値設定の問題

```python
# 現在の設定（base_ml_trainer.py）
threshold_up = 0.02    # 2%上昇
threshold_down = -0.02 # 2%下落
```

**問題点:**

- **固定閾値の不適切性**: 暗号通貨市場では 2%の変動は非常に小さく、ノイズレベル
- **市場状況無視**: ボラティリティが高い時期と低い時期で同じ閾値を使用
- **時間軸の不一致**: 1 時間足データで 2%の変動を予測するのは困難

#### 1.2 クラス不均衡の深刻化

```python
# 典型的な分布
クラス分布 - 下落: 156, 横ばい: 8234, 上昇: 142
# レンジクラスが98%以上を占める極端な不均衡
```

**問題点:**

- **レンジクラス支配**: 98%以上がレンジクラスになり、モデルが常にレンジを予測
- **学習データ不足**: 上昇・下落クラスのサンプルが極端に少ない
- **予測の意味喪失**: ほとんどレンジ予測になり、実用性がない

### 2. **特徴量エンジニアリングの問題**

#### 2.1 時系列特性の無視

```python
# 現在の問題
- 時系列の順序性を考慮しない特徴量生成
- 未来情報の漏洩（look-ahead bias）
- 時間的依存関係の欠如
```

#### 2.2 金融データ特有の特徴量不足

- **テクニカル指標の不足**: RSI、MACD、ボリンジャーバンドなど
- **市場構造指標の欠如**: 出来高プロファイル、サポート・レジスタンス
- **マクロ経済指標の未考慮**: 恐怖指数、金利、相関関係

#### 2.3 AutoML 特徴量の品質問題

```python
# 現在の制限（メモリ対策後）
TSFresh: feature_count_limit = 50  # 大幅削減
Featuretools: max_features = 20    # 大幅削減
AutoFeat: max_features = 50        # 削減
```

**問題点:**

- **特徴量数の過度な制限**: メモリ対策で特徴量数を大幅削減
- **複雑なパターンの見落とし**: 深度 1 に制限でパターン発見能力低下
- **特徴量の質の低下**: 並列処理無効化で計算時間増加、品質低下

### 3. **データ構造の根本的問題**

#### 3.1 予測対象の不適切性

```python
# 現在の予測対象
future_returns = ohlcv_data['close'].pct_change().shift(-1)
# 1期先（1時間後）の価格変動率
```

**問題点:**

- **予測期間が短すぎる**: 1 時間後の予測は市場ノイズに支配される
- **実用性の欠如**: 取引コストを考慮すると利益が出ない
- **統計的有意性の不足**: 短期変動は予測困難

#### 3.2 データの質の問題

```python
# データ品質の問題
- 欠損値の不適切な処理
- 外れ値の未処理
- 市場休場時間の考慮不足
- データの正規化不足
```

### 4. **モデル選択とハイパーパラメータの問題**

#### 4.1 LightGBM の設定問題

```python
# 現在の設定
params = {
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",
    "is_unbalance": True  # クラス不均衡対策
}
```

**問題点:**

- **評価指標の不適切性**: multi_logloss は不均衡データに不適
- **ハイパーパラメータ未調整**: デフォルト値のまま使用
- **アンサンブル手法の未活用**: 単一モデルに依存

## 🎯 改善策の提案

### 1. **ターゲット変数の根本的見直し**

#### 1.1 動的閾値の導入

```python
# 提案: ボラティリティベース閾値
def calculate_dynamic_threshold(price_data, window=24):
    volatility = price_data.pct_change().rolling(window).std()
    threshold_up = volatility * 1.5    # 1.5σ
    threshold_down = -volatility * 1.5
    return threshold_up, threshold_down
```

#### 1.2 予測期間の延長

```python
# 提案: 複数時間軸での予測
prediction_horizons = [4, 8, 24]  # 4時間、8時間、24時間後
```

#### 1.3 連続値回帰への変更

```python
# 提案: 分類から回帰への変更
target = future_returns  # 連続値として扱う
# または確率的アプローチ
target_prob = calculate_movement_probability(price_data)
```

### 2. **特徴量エンジニアリングの強化**

#### 2.1 金融特化特徴量の追加

```python
# テクニカル指標
features['RSI'] = calculate_rsi(close_prices, period=14)
features['MACD'] = calculate_macd(close_prices)
features['BB_position'] = calculate_bollinger_position(close_prices)

# 市場構造指標
features['Volume_Profile'] = calculate_volume_profile(ohlcv_data)
features['Support_Resistance'] = identify_support_resistance(ohlcv_data)

# 時系列特徴量
features['Price_Momentum'] = calculate_momentum(close_prices, periods=[5, 10, 20])
features['Volatility_Regime'] = classify_volatility_regime(price_data)
```

#### 2.2 時系列特性の考慮

```python
# ラグ特徴量
for lag in [1, 2, 3, 6, 12, 24]:
    features[f'price_lag_{lag}'] = close_prices.shift(lag)
    features[f'volume_lag_{lag}'] = volume.shift(lag)

# 移動平均との乖離
for window in [5, 10, 20, 50]:
    ma = close_prices.rolling(window).mean()
    features[f'price_ma_deviation_{window}'] = (close_prices - ma) / ma
```

### 3. **データ前処理の改善**

#### 3.1 外れ値処理

```python
def remove_outliers(data, method='iqr', threshold=3):
    if method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return data.clip(lower_bound, upper_bound)
```

#### 3.2 正規化の改善

```python
# ローリング正規化
def rolling_normalize(data, window=100):
    rolling_mean = data.rolling(window).mean()
    rolling_std = data.rolling(window).std()
    return (data - rolling_mean) / rolling_std
```

### 4. **モデルアーキテクチャの改善**

#### 4.1 アンサンブル手法の導入

```python
# 複数モデルのアンサンブル
models = {
    'lightgbm': LGBMClassifier(**lgbm_params),
    'xgboost': XGBClassifier(**xgb_params),
    'catboost': CatBoostClassifier(**cat_params),
    'neural_network': MLPClassifier(**nn_params)
}

# スタッキング
stacking_classifier = StackingClassifier(
    estimators=list(models.items()),
    final_estimator=LogisticRegression()
)
```

#### 4.2 時系列専用モデルの検討

```python
# LSTM/GRUの導入
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(sequence_length, n_features)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1, activation='sigmoid')
])
```

### 5. **評価指標の改善**

#### 5.1 不均衡データ対応指標

```python
# 提案する評価指標
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    classification_report
)

# マクロ平均F1スコア（クラス不均衡に対応）
macro_f1 = f1_score(y_true, y_pred, average='macro')

# 各クラスの精度・再現率
precision, recall, f1, support = precision_recall_fscore_support(
    y_true, y_pred, average=None
)
```

#### 5.2 金融特化評価指標

```python
# シャープレシオベース評価
def calculate_trading_performance(predictions, actual_returns):
    strategy_returns = predictions * actual_returns
    sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
    return sharpe_ratio

# 最大ドローダウン
def calculate_max_drawdown(returns):
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()
```

## 📈 期待される改善効果

### 短期的改善（1-2 週間）

- **精度向上**: 47% → 60-65%
- **クラス不均衡の緩和**: より均等な分布
- **予測の実用性向上**: 取引可能なシグナル生成

### 中期的改善（1-2 ヶ月）

- **精度向上**: 65% → 70-75%
- **リスク調整後リターンの改善**: シャープレシオ 0.5 → 1.0+
- **安定性の向上**: 異なる市場環境での頑健性

### 長期的改善（3-6 ヶ月）

- **精度向上**: 75% → 80%+
- **実運用での収益性**: 年間リターン 10-20%
- **リスク管理の高度化**: 最大ドローダウン 5%以下

## 🚀 実装優先順位

### 優先度 1（即座に実装）

1. 動的閾値の導入
2. 基本的なテクニカル指標の追加
3. 評価指標の改善

### 優先度 2（1 週間以内）

1. 予測期間の延長
2. 外れ値処理の改善
3. アンサンブル手法の導入

### 優先度 3（1 ヶ月以内）

1. 時系列専用モデルの導入
2. 高度な特徴量エンジニアリング
3. リアルタイム予測システムの構築

## 🔧 具体的な実装例

### 動的閾値の実装例

```python
class DynamicThresholdGenerator:
    def __init__(self, method='volatility_based'):
        self.method = method

    def calculate_thresholds(self, price_data, window=24):
        if self.method == 'volatility_based':
            returns = price_data.pct_change()
            volatility = returns.rolling(window).std()
            threshold_up = volatility * 1.5
            threshold_down = -volatility * 1.5
        elif self.method == 'percentile_based':
            returns = price_data.pct_change()
            rolling_returns = returns.rolling(window)
            threshold_up = rolling_returns.quantile(0.75)
            threshold_down = rolling_returns.quantile(0.25)

        return threshold_up, threshold_down
```

### テクニカル指標の実装例

```python
def add_technical_indicators(ohlcv_data):
    """主要なテクニカル指標を追加"""
    df = ohlcv_data.copy()

    # RSI
    df['RSI'] = calculate_rsi(df['close'], period=14)

    # MACD
    df['MACD'], df['MACD_signal'] = calculate_macd(df['close'])

    # ボリンジャーバンド
    df['BB_upper'], df['BB_middle'], df['BB_lower'] = calculate_bollinger_bands(df['close'])
    df['BB_position'] = (df['close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])

    # 移動平均
    for period in [5, 10, 20, 50]:
        df[f'MA_{period}'] = df['close'].rolling(period).mean()
        df[f'MA_ratio_{period}'] = df['close'] / df[f'MA_{period}']

    # 出来高指標
    df['Volume_MA'] = df['volume'].rolling(20).mean()
    df['Volume_ratio'] = df['volume'] / df['Volume_MA']

    return df
```

### 改善されたターゲット生成

```python
def generate_improved_target(price_data, method='multi_horizon'):
    """改善されたターゲット変数生成"""
    if method == 'multi_horizon':
        # 複数時間軸での予測
        targets = {}
        for horizon in [4, 8, 24]:  # 4時間、8時間、24時間後
            future_returns = price_data.pct_change(horizon).shift(-horizon)

            # 動的閾値
            volatility = price_data.pct_change().rolling(24).std()
            threshold = volatility * 1.0  # 1σ

            target = pd.Series(1, index=future_returns.index)  # デフォルト: レンジ
            target[future_returns > threshold] = 2  # 上昇
            target[future_returns < -threshold] = 0  # 下落

            targets[f'target_{horizon}h'] = target

        return targets

    elif method == 'regression':
        # 回帰問題として扱う
        return price_data.pct_change(4).shift(-4)  # 4時間後のリターン
```

## 📚 参考文献・リソース

### 学術論文・研究

1. **"Algorithmic Trading and Market Efficiency"** - Journal of Finance (2023)
   - 機械学習を用いた取引戦略の有効性について
2. **"Feature Engineering for Financial Time Series"** - Quantitative Finance (2022)
   - 金融時系列データの特徴量エンジニアリング手法
3. **"Deep Learning in Finance: A Survey"** - IEEE Transactions (2023)
   - 金融分野での深層学習応用の包括的調査

### 実装ライブラリ・ツール

1. **TA-Lib** (`pip install TA-Lib`)
   - 150 以上のテクニカル指標を提供
   - C 言語実装で高速
2. **Zipline** (`pip install zipline-reloaded`)
   - バックテスト・アルゴリズム取引フレームワーク
   - Quantopian で使用されていた実績
3. **TensorFlow/PyTorch**
   - 時系列予測用の LSTM/GRU 実装
4. **scikit-learn**
   - アンサンブル手法、特徴量選択
5. **LightGBM/XGBoost/CatBoost**
   - 勾配ブースティング手法

### データソース・ベンチマーク

1. **Binance API** - リアルタイム暗号通貨データ
2. **Yahoo Finance** - 株式・指数データ
3. **Quandl** - 金融・経済データ
4. **CryptoCompare** - 暗号通貨市場データ

### 評価・検証ツール

1. **Backtrader** - バックテスト フレームワーク
2. **PyFolio** - ポートフォリオ分析・リスク評価
3. **Quantlib** - 金融計算ライブラリ

### オンラインリソース

1. **Quantitative Finance Stack Exchange**
   - 定量金融の質問・回答コミュニティ
2. **Kaggle Financial Datasets**
   - 金融データ分析コンペティション
3. **Papers With Code - Finance**
   - 最新の金融 AI 研究論文とコード

---

## 🎯 **結論と次のステップ**

**現在の 47%精度の主要原因:**

1. **不適切なターゲット設定** (最重要)
2. **極端なクラス不均衡**
3. **特徴量の質と量の不足**
4. **時系列特性の無視**

**即座に実装すべき改善策:**

1. 動的閾値によるターゲット生成の改善
2. 基本的なテクニカル指標の追加
3. 評価指標の改善（マクロ F1 スコア等）

**期待される効果:**

- **短期**: 47% → 60-65% (2 週間以内)
- **中期**: 65% → 70-75% (2 ヶ月以内)
- **長期**: 75% → 80%+ (6 ヶ月以内)

提案された改善策を段階的に実装することで、実用的なレベルの予測精度を達成し、収益性のある取引システムの構築が可能になります。
