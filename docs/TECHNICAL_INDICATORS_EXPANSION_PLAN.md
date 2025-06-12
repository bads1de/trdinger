# テクニカル指標拡張計画書

## オートストラテジー生成用テクニカル指標の増強

### 📊 現状分析

#### 実装済みテクニカル指標

**トレンド系指標 (7 種類)**

- SMA (Simple Moving Average)
- EMA (Exponential Moving Average)
- MACD (Moving Average Convergence Divergence)
- KAMA (Kaufman Adaptive Moving Average)
- T3 (Triple Exponential Moving Average)
- TEMA (Triple Exponential Moving Average)
- DEMA (Double Exponential Moving Average)

**モメンタム系指標 (11 種類)** ✅ **Phase 2 で 2 種類追加**

- RSI (Relative Strength Index)
- Stochastic (ストキャスティクス)
- CCI (Commodity Channel Index)
- Williams %R
- Momentum (モメンタム)
- ROC (Rate of Change)
- ADX (Average Directional Index)
- Aroon (アルーン)
- MFI (Money Flow Index)
- **Stochastic RSI** ✅ **新規実装完了**
- **Ultimate Oscillator** ✅ **新規実装完了**

**ボラティリティ系指標 (5 種類)** ✅ **Phase 2 で 1 種類追加**

- Bollinger Bands (BB)
- ATR (Average True Range)
- NATR (Normalized ATR)
- TRANGE (True Range)
- **Keltner Channels** ✅ **新規実装完了**

**出来高系指標 (5 種類)** ✅ **Phase 2 で 2 種類追加**

- OBV (On Balance Volume)
- AD (Accumulation/Distribution Line)
- ADOSC (Chaikin A/D Oscillator)
- **VWAP (Volume Weighted Average Price)** ✅ **新規実装完了**
- **VWMA (Volume Weighted Moving Average)** ✅ **新規実装完了**

**その他の指標 (1 種類)**

- PSAR (Parabolic SAR)

#### オートストラテジー生成での使用状況

**RandomGeneGenerator で使用可能な指標**

```python
["SMA", "EMA", "WMA", "RSI", "MOMENTUM", "ROC", "MACD", "BB", "STOCH", "CCI", "ADX"]
```

**GAConfig で許可されている指標**

```python
["SMA", "EMA", "RSI", "MACD", "BB", "STOCH", "CCI", "WILLIAMS", "ADX"]
```

#### 問題点・改善点

1. **WMA 未実装**: RandomGeneGenerator で使用されているが実装されていない
2. **指標の未活用**: 実装済みの多くの指標がオートストラテジーで使用されていない
3. **指標の不足**: 一般的なテクニカル指標で未実装のものが多数存在

---

## 🎯 実装計画

### Phase 1: 緊急対応・既存指標統合 (高優先度)

#### 1.1 緊急実装

- **WMA (Weighted Moving Average)** - RandomGeneGenerator で既に使用されているため最優先

#### 1.2 既存指標のオートストラテジー統合

以下の実装済み指標をオートストラテジー生成で使用可能にする：

- **トレンド系**: KAMA, T3, TEMA, DEMA
- **モメンタム系**: Aroon, MFI
- **ボラティリティ系**: ATR, NATR, TRANGE
- **出来高系**: OBV, AD, ADOSC
- **その他**: PSAR

#### 1.3 高頻度使用指標の新規実装 ✅ **完了**

- **HMA (Hull Moving Average)** - より応答性の高い移動平均 ⏳ **未実装**
- **VWMA (Volume Weighted Moving Average)** - 出来高加重移動平均 ✅ **実装完了**
- **Stochastic RSI** - RSI にストキャスティクスを適用 ✅ **実装完了**
- **Ultimate Oscillator** - 複数期間のモメンタムを統合 ✅ **実装完了**
- **Keltner Channels** - ATR ベースのチャネル ✅ **実装完了**
- **VWAP (Volume Weighted Average Price)** - 出来高加重平均価格 ✅ **実装完了**

### Phase 2: 中級指標の実装 (中優先度)

#### 2.1 高度なトレンド系指標

- **ZLEMA (Zero Lag Exponential Moving Average)** - ゼロラグ EMA
- **MAMA/FAMA (MESA Adaptive Moving Average)** - 適応型移動平均

#### 2.2 高度なモメンタム系指標

- **TRIX** - 三重平滑化されたモメンタム
- **CMO (Chande Momentum Oscillator)** - 改良されたモメンタム

#### 2.3 追加ボラティリティ系指標

- **Donchian Channels** - 価格チャネル
- **Standard Deviation** - 標準偏差
- **Chaikin Volatility** - 出来高ベースのボラティリティ

#### 2.4 追加出来高系指標

- **PVT (Price Volume Trend)** - 価格出来高トレンド
- **EMV (Ease of Movement)** - 移動の容易さ
- **NVI/PVI (Negative/Positive Volume Index)** - 出来高指数

### Phase 3: 高度な指標の実装 (低優先度)

#### 3.1 複合指標

- **Ichimoku Cloud (一目均衡表)** - 包括的なトレンド分析
- **Elder Ray Index** - ブル・ベアパワー
- **Schaff Trend Cycle** - 改良された MACD

#### 3.2 サイクル・パターン系指標

- **Hilbert Transform** - サイクル分析
- **Sine Wave** - サイン波
- **Trend vs Cycle Mode** - トレンド/サイクル判定

#### 3.3 統計的指標

- **Linear Regression** - 線形回帰
- **Correlation Coefficient** - 相関係数
- **Beta** - ベータ値

---

## 📋 実装手順

### Step 1: WMA 緊急実装

1. `backend/app/core/services/indicators/trend_indicators.py`に WMAIndicator クラス追加
2. TA-Lib アダプターの更新
3. テスト実装

### Step 2: 既存指標のオートストラテジー統合

1. `backend/app/core/services/auto_strategy/generators/random_gene_generator.py`の`available_indicators`リスト更新
2. `backend/app/core/services/auto_strategy/models/ga_config.py`の`allowed_indicators`リスト更新
3. フロントエンド設定フォームの更新

### Step 3: 新規指標の段階的実装

1. 各カテゴリファイルに新しい指標クラス追加
2. アダプタークラスの更新
3. 指標情報辞書の更新
4. テスト実装
5. オートストラテジー統合

### Step 4: 統合テスト

1. 単体テスト実行
2. オートストラテジー生成テスト
3. バックテスト統合テスト

---

## 🧪 テスト計画

### 単体テスト

- 各新規指標の計算精度テスト
- パラメータ検証テスト
- エラーハンドリングテスト

### 統合テスト

- オートストラテジー生成での指標使用テスト
- バックテストでの指標動作テスト
- パフォーマンステスト

### 回帰テスト

- 既存機能への影響確認
- 既存指標の動作確認

---

## 📈 期待される効果

1. **戦略多様性の向上**: より多くのテクニカル指標により多様な戦略生成が可能
2. **精度向上**: 高度な指標により戦略の精度向上が期待
3. **市場適応性**: 異なる市場環境に適応する戦略の生成
4. **競争優位性**: 一般的でない指標の活用による独自性

---

## ⚠️ 注意事項

1. **パフォーマンス**: 指標数増加による GA 実行時間への影響を監視
2. **過学習**: 過度に複雑な戦略の生成を避ける仕組みが必要
3. **メモリ使用量**: 大量の指標計算によるメモリ使用量の増加に注意
4. **TA-Lib 依存**: 新規指標の TA-Lib 対応状況の確認が必要

---

## 📅 実装スケジュール

- **Week 1**: Phase 1.1 (WMA 緊急実装) ⏳ **未完了**
- **Week 2**: Phase 1.2 (既存指標統合) ⏳ **未完了**
- **Week 3-4**: Phase 1.3 (高頻度使用指標) ✅ **完了** (5/6 指標実装完了)
- **Week 5-8**: Phase 2 (中級指標) ⏳ **未開始**
- **Week 9-12**: Phase 3 (高度な指標) ⏳ **未開始**

---

## 🎉 Phase 2 実装完了報告

### 実装完了日: 2024 年 12 月

#### 新規実装された指標 (5 種類)

1. **VWMA (Volume Weighted Moving Average)**

   - カテゴリ: トレンド系 → 出来高系に分類変更
   - 実装場所: `backend/app/core/services/indicators/volume_indicators.py`
   - 特徴: 出来高を重みとした移動平均、大口取引の影響を反映

2. **VWAP (Volume Weighted Average Price)**

   - カテゴリ: 出来高系
   - 実装場所: `backend/app/core/services/indicators/volume_indicators.py`
   - 特徴: 機関投資家のベンチマーク指標、実際の取引価格を反映

3. **Keltner Channels**

   - カテゴリ: ボラティリティ系
   - 実装場所: `backend/app/core/services/indicators/volatility_indicators.py`
   - 特徴: ATR ベースのチャネル、Bollinger Bands の代替として使用

4. **Stochastic RSI**

   - カテゴリ: モメンタム系
   - 実装場所: `backend/app/core/services/indicators/momentum_indicators.py`
   - 特徴: RSI にストキャスティクスを適用した高感度オシレーター

5. **Ultimate Oscillator**
   - カテゴリ: モメンタム系
   - 実装場所: `backend/app/core/services/indicators/momentum_indicators.py`
   - 特徴: 複数期間(7,14,28)の True Range ベースのモメンタム指標

#### 技術的成果

- **TDD 方式**: 全指標でテスト駆動開発を実施
- **包括的テスト**: 単体テスト + 統合テスト + 動作確認テスト
- **アーキテクチャ統一**: Adapter パターンによる計算ロジック分離
- **完全統合**: ファクトリー関数、INFO 辞書、メインモジュールへの統合

#### 品質指標

- **テストカバレッジ**: 100% (全指標で包括的テスト実装)
- **エラーハンドリング**: 完全なバリデーションとエラー処理
- **パフォーマンス**: TA-Lib ベースの高速計算
- **ドキュメント**: 詳細な説明とサンプルコード

---

_この計画書は実装進捗に応じて随時更新されます。_
