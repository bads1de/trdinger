# Scientific Meta-Labeling Implementation Plan

## 🎯 目的

Marcos Lopez de Prado 氏の提唱する「Advances in Financial Machine Learning」に基づき、科学的・統計的アプローチによるダマシ検知（Meta-Labeling）モデルを構築する。
従来の主観的なテクニカル指標（BB 等）への依存を脱却し、**CUSUM フィルター**と**動的ボラティリティ**を活用することで、学習データの質と量を劇的に改善し、実用的な Precision（勝率）の達成を目指す。

## 📉 現状の課題と解決策

| 課題                        | 原因                                                | 解決策 (Scientific Approach)                                              |
| --------------------------- | --------------------------------------------------- | ------------------------------------------------------------------------- |
| **学習データ不足**          | 厳しすぎるフィルタ（出来高 2 倍等）でイベントが枯渇 | **Symmetric CUSUM Filter** を採用し、統計的に有意な全変動をイベント化     |
| **低 Precision (21%)**      | ベースシグナル（BB 等）の素の勝率が低すぎる         | 主観的指標を廃止し、**Volatility-based** な客観的イベントのみを対象とする |
| **Triple Barrier 機能不全** | 固定的な PT/SL 幅が相場に合っていない               | **Daily Volatility** に連動した動的な PT/SL 幅を設定する                  |

## 🛠 実装フェーズ

### Phase 1: Symmetric CUSUM Filter の実装 (Event Detection)

価格変動の累積和（CUSUM）を用いて、統計的に有意なイベントを検出する。

- **理論**: $S_t = \max(0, S_{t-1} + y_t - E_{t-1}[y_t])$
- **実装**: `CusumSignalGenerator` クラスの作成
  - 日次ボラティリティ（または ATR）を閾値として使用
  - 上方変動・下方変動の両方を検知
  - これにより、数千〜数万件の「意味のある変動」を ML に供給する

### Phase 2: Dynamic Triple Barrier の適正化 (Labeling)

イベント発生時のボラティリティに基づいて、利確・損切幅を動的に決定する。

- **変更点**:
  - 固定幅（例: 1%）を廃止
  - `width = volatility * factor` で動的に設定
  - `LabelGenerationService` の改修
- **Vertical Barrier**:
  - 時間切れ時の扱いを明確化（リターンが閾値未満なら `0`、それ以外は符号で判定など）

### Phase 3: Microstructure Features の強化 (Feature Engineering)

Kaggle 上位解法を参考に、需給の不均衡（ダマシの予兆）を捉える特徴量を追加する。

- **追加特徴量**:
  - **VPIN (Volume-Synchronized Probability of Informed Trading)** 簡易版: 出来高の不均衡度
  - **OI Flow**: 建玉の変動フロー（価格変動との乖離）
  - **Amihud Illiquidity**: 流動性の枯渇度
  - **Roll Measure**: 実効スプレッドの推定

### Phase 4: Probability Calibration & Bet Sizing (Evaluation)

単なる 0/1 予測ではなく、確率に基づいたポジションサイジングを行う。

- **評価指標**:
  - Precision (Win Rate)
  - **Probabilistic Sharpe Ratio (PSR)**: 確率的シャープレシオ
- **実装**:
  - 予測確率の分布分析
  - 確率に応じたベットサイズ（ポジション量）のシミュレーション

## 📅 作業手順

1. **`CusumSignalGenerator` の実装**

   - `backend/app/services/ml/label_generation/cusum_generator.py` を作成
   - テストコード作成と検証

2. **`LabelGenerationService` の改修**

   - CUSUM フィルターとの統合
   - 動的ボラティリティによるバリア幅設定の実装

3. **特徴量エンジニアリング**

   - `MicrostructureFeatureCalculator` の実装
   - VPIN, OI Flow 等の計算ロジック追加

4. **ML パイプラインの再実行と検証**
   - データ量を増やして学習（Limit: 50,000 件〜）
   - Precision 55%超えを目指す

## 📚 参考文献

- Marcos Lopez de Prado, "Advances in Financial Machine Learning", Wiley, 2018.
- Kaggle Competitions: G-Research Crypto Forecasting, Jane Street Market Prediction.
