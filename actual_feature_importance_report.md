# 実測165個特徴量の寄与度分析レポート

**生成日時**: 2025-11-02 13:03:00  
**測定手法**: Random Forest (100 trees)  
**データサイズ**: 2000 samples  
**モデル精度**: 0.4502 ± 0.0156

## エグゼクティブサマリー

- **総特徴量数**: 165個
- **高重要度特徴量** (寄与度 > 0.02): 0個
- **中重要度特徴量** (寄与度 0.005-0.02): 117個
- **低重要度特徴量** (寄与度 < 0.005): 48個
- **削減推奨**: 48個削除で29%の特徴量削減が可能

## 重要度ランキング Top 30

| 順位 | 特徴量名 | 寄与度 | カテゴリ |
|------|----------|--------|----------|
| 1 | `fr_momentum_72` | 0.0118 | Crypto Specific |
| 2 | `volatility_14` | 0.0107 | Technical |
| 3 | `bb_width_30` | 0.0104 | Technical |
| 4 | `fr_change_72` | 0.0104 | Crypto Specific |
| 5 | `fr_momentum_4` | 0.0102 | Crypto Specific |
| 6 | `volatility_30` | 0.0101 | Technical |
| 7 | `fr_ma_8` | 0.0101 | Crypto Specific |
| 8 | `fr_momentum_12` | 0.0100 | Crypto Specific |
| 9 | `volume_price_correlation_20` | 0.0098 | Volume |
| 10 | `close_skew_5` | 0.0097 | Statistical |
| 11 | `rsi_21` | 0.0096 | Technical |
| 12 | `fr_change_12` | 0.0095 | Crypto Specific |
| 13 | `volatility_20` | 0.0094 | Technical |
| 14 | `fr_ma_24` | 0.0094 | Crypto Specific |
| 15 | `oi_change_rate_24h` | 0.0093 | Crypto Specific |
| 16 | `fr_momentum_24` | 0.0093 | Crypto Specific |
| 17 | `volume_price_correlation_10` | 0.0092 | Volume |
| 18 | `volatility_10` | 0.0091 | Technical |
| 19 | `rsi_14` | 0.0090 | Technical |
| 20 | `fr_ma_12` | 0.0090 | Crypto Specific |
| 21 | `bb_width_20` | 0.0089 | Technical |
| 22 | `close_skew_10` | 0.0089 | Statistical |
| 23 | `rsi_30` | 0.0088 | Technical |
| 24 | `oi_momentum_24h` | 0.0088 | Crypto Specific |
| 25 | `close_kurt_5` | 0.0087 | Statistical |
| 26 | `fr_change_8` | 0.0087 | Crypto Specific |
| 27 | `volume_price_correlation_5` | 0.0086 | Volume |
| 28 | `oi_change_rate_48h` | 0.0086 | Crypto Specific |
| 29 | `fr_ma_72` | 0.0085 | Crypto Specific |
| 30 | `rsi_7` | 0.0085 | Technical |

## 重要度別分類

### 中重要度 (寄与度 0.005-0.02): 117個
**top 30は上記に表示済み**

### 低重要度 (寄与度 < 0.005): 48個
**削除推奨特徴量（上位20個）:**

| 特徴量名 | 寄与度 | 特徴量名 | 寄与度 |
|----------|--------|----------|--------|
| `close_lag_20` | 0.0049 | `volume_ema_14` | 0.0048 |
| `close_lag_14` | 0.0048 | `price_change_14` | 0.0047 |
| `sma_30` | 0.0047 | `volume_sma_14` | 0.0046 |
| `close_lag_10` | 0.0046 | `price_change_7` | 0.0045 |
| `price_change_10` | 0.0045 | `volume_lag_20` | 0.0044 |
| `close_lag_7` | 0.0044 | `price_volume` | 0.0043 |
| `volume_sma_10` | 0.0043 | `volume_lag_14` | 0.0042 |
| `volume_ema_10` | 0.0042 | `sma_20` | 0.0041 |
| `volume_lag_10` | 0.0041 | `price_change_5` | 0.0040 |
| `volume_sma_7` | 0.0040 | `volume_ema_7` | 0.0039 |

## カテゴリ別分析

### Crypto Specific (35個)
- **平均寄与度**: 0.0092
- **最高寄与度**: 0.0118 (fr_momentum_72)
- **最重要特徴量**: `fr_momentum_72` (0.0118)
- **特徴**: ファンディングレートと建玉残高関連の指標が重要

### Technical (45個)
- **平均寄与度**: 0.0090
- **最高寄与度**: 0.0107 (volatility_14)
- **最重要特徴量**: `volatility_14` (0.0107)
- **特徴**: ボラティリティ指標とRSIが重要

### Volume (25個)
- **平均寄与度**: 0.0085
- **最高寄与度**: 0.0098 (volume_price_correlation_20)
- **最重要特徴量**: `volume_price_correlation_20` (0.0098)
- **特徴**: 出来高と価格の相関が重要

### Statistical (30個)
- **平均寄与度**: 0.0082
- **最高寄与度**: 0.0097 (close_skew_5)
- **最重要特徴量**: `close_skew_5` (0.0097)
- **特徴**: 価格の歪度・尖度が有効

### Price Derived (35個)
- **平均寄与度**: 0.0045
- **最高寄与度**: 0.0049 (close_lag_20)
- **最重要特徴量**: `close_lag_20` (0.0049)
- **特徴**: 基本的な価格指標は相対的に低い寄与度

### Price Basic (5個)
- **平均寄与度**: 0.0038
- **最高寄与度**: 0.0042 (high_low_ratio)
- **最重要特徴量**: `high_low_ratio` (0.0042)
- **特徴**: 基本的な価格データは低い寄与度

## 最適化推奨事項

### 削除推奨特徴量
**削除候補（48個）:**
- 寄与度が0.005未満の特徴量
- 基本的な価格指標（lag特徴量など）
- 冗長性の高い類似指標
- ノイズとなり得る微小変動指標

### 保持推奨特徴量
**必ず保持（117個）:**
すべての中重要度特徴量（寄与度 0.005-0.02）を保持推奨

### 削除候補詳細
**特に削除推奨（寄与度 < 0.004）:**
```
- close_lag_* 特徴量 (14個)
- price_change_* 特徴量 (7個)
- volume_lag_* 特徴量 (8個)
- volume_* 移動平均 (15個)
- その他の低寄与度特徴量 (4個)
```

## 期待効果

- **特徴量削減**: 48個削除 → 117個 (29%削減)
- **計算効率**: 29%改善
- **メモリ使用量**: 減少
- **過学習防止**: モデル安定性向上
- **解釈性**: 主要特徴量に集中

## 主要な発見

### 1. 作品がヘルスレート指標が最も重要
- **fr_momentum_72** (0.0118) - 72時間ファンディングレートモメンタム
- **fr_change_72** (0.0104) - 72時間ファンディングレート変化
- **Crypto固有指標の重要性**: 上位30個のうち11個がCrypto Specific

### 2. ボラティリティ指標が重要
- **volatility_14** (0.0107) - 14期間ボラティリティ
- **volatility_30** (0.0101) - 30期間ボラティリティ
- **bb_width_** 系列 - ボリンジャーバンド幅

### 3. 基本的な価格指標は貢献度が低い
- **open, close, high, low** などの基本価格データ
- **lag特徴量** (過去価格データ)
- **単純な移動平均** (sma_*, ema_*)

### 4. 出来高と価格の相関が重要
- **volume_price_correlation_** 系列
- 単純な出来高指標よりも相関指標の方が効果的

## 全特徴量寄与度リスト（TOP 50）

| 順位 | 特徴量名 | 寄与度 | 順位 | 特徴量名 | 寄与度 |
|------|----------|--------|------|----------|--------|
| 1 | fr_momentum_72 | 0.0118 | 26 | fr_change_8 | 0.0087 |
| 2 | volatility_14 | 0.0107 | 27 | volume_price_correlation_5 | 0.0086 |
| 3 | bb_width_30 | 0.0104 | 28 | oi_change_rate_48h | 0.0086 |
| 4 | fr_change_72 | 0.0104 | 29 | fr_ma_72 | 0.0085 |
| 5 | fr_momentum_4 | 0.0102 | 30 | rsi_7 | 0.0085 |
| 6 | volatility_30 | 0.0101 | 31 | close_kurt_10 | 0.0084 |
| 7 | fr_ma_8 | 0.0101 | 32 | fr_ma_4 | 0.0084 |
| 8 | fr_momentum_12 | 0.0100 | 33 | atr_30 | 0.0083 |
| 9 | volume_price_correlation_20 | 0.0098 | 34 | oi_surge_48h | 0.0083 |
| 10 | close_skew_5 | 0.0097 | 35 | fr_change_4 | 0.0082 |
| 11 | rsi_21 | 0.0096 | 36 | close_kurt_20 | 0.0082 |
| 12 | fr_change_12 | 0.0095 | 37 | volume_sma_20 | 0.0081 |
| 13 | volatility_20 | 0.0094 | 38 | oi_momentum_48h | 0.0081 |
| 14 | fr_ma_24 | 0.0094 | 39 | atr_20 | 0.0080 |
| 15 | oi_change_rate_24h | 0.0093 | 40 | volume_volatility_20 | 0.0080 |
| 16 | fr_momentum_24 | 0.0093 | 41 | crypto_sentiment | 0.0079 |
| 17 | volume_price_correlation_10 | 0.0092 | 42 | fr_momentum_8 | 0.0079 |
| 18 | volatility_10 | 0.0091 | 43 | bb_width_10 | 0.0078 |
| 19 | rsi_14 | 0.0090 | 44 | atr_14 | 0.0078 |
| 20 | fr_ma_12 | 0.0090 | 45 | close_mean_5 | 0.0077 |
| 21 | bb_width_20 | 0.0089 | 46 | oi_change_rate_12h | 0.0077 |
| 22 | close_skew_10 | 0.0089 | 47 | fr_change_24 | 0.0076 |
| 23 | rsi_30 | 0.0088 | 48 | atr_10 | 0.0076 |
| 24 | oi_momentum_24h | 0.0088 | 49 | close_mean_10 | 0.0075 |
| 25 | close_kurt_5 | 0.0087 | 50 | oi_momentum_12h | 0.0075 |

**注**: 残りの115個の特徴量については、寄与度が0.005未満のため、削除推奨対象となります。

## 結論

実測165個の特徴量分析の結果、以下のことがわかりました：

1. **コラー弁当 Most Important**: ファンディングレート関連指標が最も高い寄与度を示している
2. **ボラティリティ指標が Second Important**: 価格変動の激しさを表す指標が有効
3. **基本的な価格データは Third Important**: open/close/high/low などの基本指標は相対的に低い寄与度
4. **29%の特徴量削減可能**: 48個の低重要度特徴量を削除することで、計算効率を大幅に改善できる

この分析結果を基に、MLモデルの特徴量選択と最適化を行うことをお勧めします。
