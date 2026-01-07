# 王道戦略シード注入設計 (Hybrid Initialization)

## 1. 目的

GA（遺伝的アルゴリズム）の初期収束を助け、「ゴミを磨く」状態を回避するために、歴史的に有効性が証明されている「王道戦略」を初期集団に一定割合だけ注入する。
これにより、**「多様性の確保（90%のランダム）」と「最低限の性能保証（10%のシード）」のハイブリッド構成**を実現する。

## 2. アーキテクチャ

### 2.1 構成比率

- **全個体数**: $N$ (例: 100)
- **シード注入率**: 10% ($0.1 \times N$)
- **ランダム生成**: 90% ($0.9 \times N$)

### 2.2 新規コンポーネント

- `SeedStrategyFactory`: 定石戦略の `StrategyGene` オブジェクトを生成する工場クラス。

## 3. 初期投入する王道戦略定義 (The Seeds)

以下の 5 つの戦略をシードとして定義する。これらは極端に複雑ではなく、GA がパラメータ調整（最適化）や条件追加（フィルタリング）を行いやすい素直なロジックとする。

### Strategy A: DMI Extreme Trend (High Momentum)

単純な SMA クロスではなく、ADX/DMI 指標を用いて極めて強いトレンド発生（DI > 45 かつ ADX > 45）を狙う戦略。方向性とトレンド強度の両方が極まったポイントを狙う。

- **指標**:
  - ADX/DMI: Period=14
- **エントリー条件 (Long)**:
  - `DMP > 45` (Plus DI が強い)
  - **AND** `ADX > 45` (トレンド自体も強い)
- **エントリー条件 (Short)**:
  - `DMN > 45` (Minus DI が強い)
  - **AND** `ADX > 45` (トレンド自体も強い)
- **TP/SL**:
  - TP: 5%, SL: 3%

### Strategy B: RSI Momentum (Trend Follow)

暗号資産市場の強いトレンド性を考慮した、RSI を使った順張り戦略。
逆張り（Reversal）ではなく、RSI が中立ゾーンを抜けた方向へついていく。

- **指標**:
  - RSI: Period=14
- **エントリー条件 (Long)**:
  - `RSI > 75` (強力な上昇モメンタム)
- **エントリー条件 (Short)**:
  - `RSI < 25` (強力な下落モメンタム)
- **TP/SL**:
  - TP: 5%, SL: 3% (トレンドに乗るため RR 比はプラスに設定)

### Strategy C: Bollinger Breakout (Volatility Breakout)

ボラティリティの拡大（スクイーズからのエクスパンション）につく順張り戦略。

- **指標**:
  - BB Upper: Period=20, StdDev=2.0
  - BB Lower: Period=20, StdDev=2.0
- **エントリー条件 (Long)**:
  - `Close > BB Upper`
- **エントリー条件 (Short)**:
  - `Close < BB Lower`
- **TP/SL**:
  - TP: 8%, SL: 4% (RR 比 2.0) - 大きなトレンド狙い

### Strategy D: KAMA-ADX Hybrid

Kaufman Adaptive Moving Average (KAMA) を用いたトレンド判定と、ADX による強いトレンドフィルターを組み合わせた複合戦略。ノイズの多い相場でダマシを防ぐことを重視。

- **指標構成**:
  - `KAMA_MACD`: KAMA(12, 2, 30) をベースにした MACD ライン
  - `Signal`: KAMA_MACD の EMA(9)
  - `ADX`: Period=13
- **エントリー条件 (Long)**:
  - `KAMA_MACD > Signal` (KAMA ベースのゴールデンクロス)
  - **AND** `DMP > DMN` (上昇トレンド方向)
  - **AND** `DMP > 40` (上昇トレンド強度十分)
  - **AND** `ADX > 20` (トレンド発生中)
- **エントリー条件 (Short)**:
  - `KAMA_MACD < Signal` (KAMA ベースのデッドクロス)
  - **AND** `DMN > DMP` (下落トレンド方向)
  - **AND** `DMN > 40` (下落トレンド強度十分)
  - **AND** `ADX > 20` (トレンド発生中)
- **TP/SL**:
  - TP: 5%, SL: 3%

### Strategy E: WAE (Waddah Attar Explosion)

「DeadZone」を活用した強力なトレンドフィルター戦略。MACD の変化率でトレンド方向を見て、ボリンジャーバンド幅が DeadZone（ATR ベースの閾値）を超えた時のみ「爆発」としてエントリーする。

- **指標構成 (GA での再現)**:
  - `MACD_Delta`: MACD の変化率 `(MACD - MACD[1]) * Sensitivity`
  - `BB_Width`: ボリンジャーバンド幅 `Upper - Lower`
  - `Dead_Zone`: `ATR(100) * 3.7` (基準ボラティリティ)
- **エントリー条件 (Long)**:
  - `BB_Width > Dead_Zone` (ボラティリティ爆発)
  - **AND** `MACD_Delta > 0` (上昇トレンド加速)
- **エントリー条件 (Short)**:
  - `BB_Width > Dead_Zone`
  - **AND** `MACD_Delta < 0` (下落トレンド加速)
- **TP/SL**:
  - TP: 6%, SL: 3%

### Strategy F: Trendilo (ALMA Momentum)

**T3 Moving Average** による長期トレンド判定と、**ALMA (Arnaud Legoux MA)** を用いたスムースなモメンタムトリガーを組み合わせた戦略。

- **指標**:
  - `T3_MA`: Period=300, Factor=0.7 (長期トレンド)
  - `ADX`: Period=14 (勢いフィルター)
  - `Trendilo`: `ALMA(Change(Close), 20, 0.85, 6)` (エントリートリガー)
- **エントリー条件 (Long)**:
  - `Close > T3_MA` (長期上昇トレンド)
  - **AND** `ADX > 20` (勢いあり)
  - **AND** `Trendilo > 0` (CrossOver)
- **エントリー条件 (Short)**:
  - `Close < T3_MA` (長期下落トレンド)
  - **AND** `ADX > 20`
  - **AND** `Trendilo < 0` (CrossUnder)
- **TP/SL**:
  - TP: ATR(14) \* 4.0
  - SL: ATR(14) \* 2.0

## 4. 実装計画

1. `backend/app/services/auto_strategy/generators/seed_strategy_factory.py` の作成
   - 上記 5 つの戦略を `StrategyGene` として組み立てて返すメソッドを実装。
2. `GAConfig` への設定追加 (`backend/app/services/auto_strategy/config/ga.py`)
   - `use_seed_strategies: bool = True`
   - `seed_injection_rate: float = 0.1`
3. `GAEngine` の修正 (`backend/app/services/auto_strategy/core/ga_engine.py`)
   - 初期集団生成後に `SeedStrategyFactory` を呼び出して個体を置換するロジックを追加。

## 5. 期待される進化の方向性

1. **パラメータ最適化**: シードの「Period=14」や「TP=5%」とい言った数値が、現在の市場に合わせて「18」や「4.2%」などに微調整される。
2. **フィルタリング進化**: 「Strategy A (Golden Cross)」に「RSI < 70」という条件がランダムに追加され、「高値掴みを避けるゴールデンクロス」へと進化する。
3. **ハイブリッド化**: 「MACD のエントリー条件」と「ボリンジャーバンドの決済条件」が交叉によって融合する。
