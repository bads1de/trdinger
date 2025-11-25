# Trdinger ML Implementation Summary

このドキュメントは、Trdinger プロジェクトにおける機械学習パイプラインの実装状況、技術的決定、および成果を包括的にまとめたものです。他の LLM や開発者がプロジェクトの文脈を迅速に理解し、開発を継続するための資料として機能します。

## 1. プロジェクト概要

**Trdinger** は、ビットコイン無期限先物市場（BTC/USDT）を対象とした、高精度な定量的取引戦略を構築するための AI プラットフォームです。

- **目標**: レンジ相場での損失（往復ビンタ）を回避しつつ、トレンド発生時に高確度でエントリーするモデルの構築。
- **現状の到達点 (2025-11-24)**:
  - **Range 回避率**: **96.9%** (レンジ相場の誤検知をほぼ排除)
  - **Trend 確度**: **86.6%** (エントリー時の勝率)
  - **Trend 検出率**: **10.0%** (全トレンドのうち捕捉できる割合。改善傾向にあるが、さらなる向上が課題)

## 2. モデルアーキテクチャ (Stacking Ensemble)

多様性（Diversity）を確保するため、性質の異なる複数のモデルを組み合わせた **2 層スタッキング構成** を採用しています。

### Level-1: Base Models

以下の 5 つのモデルを並列に学習させます。

1. **LightGBM (GBDT)**: 高速かつ高精度。テーブルデータのパターン認識に優れる。
2. **XGBoost (GBDT)**: 安定性が高く、欠損値処理に優れる（`missing=np.inf` 設定済み）。
3. **CatBoost (GBDT)**: 時系列データやカテゴリカル変数に強く、過学習しにくい。今回の検証で最も高いパフォーマンスを示した。
4. **GRU (RNN)**: ゲート付き回帰型ユニット。時系列の短期〜中期の文脈（流れ）を学習する。
5. **LSTM (RNN)**: 長短期記憶ネットワーク。GRU より複雑な長期依存関係を捉える。

- **入力**:
  - GBDT 系: 生の特徴量（数値特徴量はクリッピング済み）。
  - DL 系 (GRU/LSTM): `StandardScaler` で正規化された特徴量。

### Level-2: Meta Learner

1. **Ridge Regression (NNLS)**:
   - 各 Level-1 モデルの **OOF (Out-Of-Fold)** 予測値を入力とする線形回帰モデル。
   - **Non-negative Least Squares (NNLS)** 制約 (`positive=True`) を適用し、負の重みを排除することで、モデルの解釈性と安定性を高めている。
   - これにより、各モデルの「信頼度」に基づいた最適な重み付けを自動学習する。

## 3. 特徴量エンジニアリング

ビットコイン市場特有の力学を捉えるため、独自の特徴量を多数実装しています（計 136 個）。

### 3.1 Funding Rate (FR) の高度化

生の FR データ（ratio）は数値が小さく 0 に張り付いているため、以下のように加工して学習効率を向上させました。

- **bps 変換**: 値を 10,000 倍し、ベーシスポイント単位に変換。
- **Deviation (乖離)**: ベースライン（0.01%）からの乖離量 (`fr_deviation_bps`) を計算。通常時は 0 になり、異常値を検出しやすい。
- **市場歪み指標**:
  - `fr_cumulative_deviation`: 乖離の累積エネルギー（マグマの蓄積）。
  - `fr_zscore`: 移動平均からの乖離（標準化）。
  - `OI_Weighted_FR`: 建玉残高（OI）で加重した FR。市場全体のコスト負担を表す。

### 3.2 RANGE 検出特化特徴量

レンジ相場を回避するために特化した特徴量群。

- `Price_Range_Normalized`: 価格レンジ幅を現在価格で正規化。
- `CHOP` (Choppiness Index): トレンドの不在を数値化。
- `BBW_Squeeze`: ボリンジャーバンドの収縮（スクイーズ）を検知。
- `Low_Volatility_Flag`: 過去のボラティリティ分布に基づく低ボラティリティ判定。

### 3.3 Multi-Timeframe (MTF)

- 1 時間足だけでなく、**4 時間足**、**24 時間足** にリサンプリングしたデータのテクニカル指標（RSI, ADX, BBW）を追加。
- 上位足のトレンド方向やボラティリティ環境を考慮可能にした。
- **実装上の注意**: 未来情報のリークを防ぐため、リサンプリング後に `reindex` と `ffill` を用いて慎重に結合している。

## 4. 検証手法: Purged K-Fold Cross Validation

金融時系列データにおける「情報のリーク」を防ぐため、厳密な交差検証手法を実装しました。

- **Purging (パージ)**: テスト期間と重複する、あるいは近接するトレーニングデータを削除する。
- **Embargo (エンバーゴ)**: テスト期間の直後のデータも、情報の遅延浸透を考慮してトレーニングから除外する（テスト期間の 1%）。
- **OOF 予測**: メタモデルの学習には、この Purged K-Fold を用いて生成された OOF 予測値のみを使用する。これにより、メタモデルの過学習を防ぎ、未知のデータに対する汎化性能を担保している。

## 5. 実装履歴と効果

| フェーズ    | 実装内容                       | 効果・結果                                                              |
| :---------- | :----------------------------- | :---------------------------------------------------------------------- |
| **Phase 1** | LightGBM/XGBoost 単体          | Range 回避率 79%。過学習の疑いあり。                                    |
| **Phase 2** | **CatBoost, GRU, LSTM 追加**   | モデル多様性が向上。DL モデルが一部の指標で GBDT を上回る。             |
| **Phase 3** | **FR 特徴量加工 & RANGE 指標** | Range 回避率が **100%** 近くまで向上。レンジ相場での損失リスクが激減。  |
| **Phase 4** | **Purged K-Fold CV 導入**      | 評価スコアが現実的な値に収束。Trend 検出率が 0.1% -> **10.0%** に改善。 |

## 6. Triple Barrier Method (TBM) の実装 ✅

Marcos Lopez de Prado の「Advances in Financial Machine Learning」で提唱された **Triple Barrier Method** を実装し、より洗練されたラベル生成手法を実現しました。

### 6.1 従来手法の課題

従来の固定ホライズン方式（「N 本先のリターンで分類」）には以下の問題がありました:

- **時間軸の柔軟性ゼロ**: トレンドの発生速度に関係なく、常に N 本先で評価するため、早い段階で利益が出ても損失方向に反転すれば「DOWN」とラベル付けされる矛盾が生じる。
- **リスク管理の欠如**: 損切りラインを無視したラベリングであり、実際のトレードで許容できる最大損失を学習できない。
- **ノイズの混入**: 垂直バリア(時間切れ)のみで判定するため、微小な変動がラベルに反映され、学習効率が低下。

### 6.2 TBM の仕組み

TBM は、各観測点(t0)から **3 つのバリア**を設定し、**最初に触れたバリア**でラベルを確定します:

1. **Upper Barrier (利確ライン, PT)**: `close[t] > close[t0] * (1 + volatility * pt_multiplier)` を満たす最初の時刻。
2. **Lower Barrier (損切りライン, SL)**: `close[t] < close[t0] * (1 - volatility * sl_multiplier)` を満たす最初の時刻。
3. **Vertical Barrier (時間切れ)**: `t0 + horizon_n` の時刻。PT/SL のどちらにも触れなかった場合に評価。

#### ラベル決定ロジック

```python
# TripleBarrier.get_bins() より
if ret > target * pt * 0.999:
    label = 1.0  # UP (利確成功)
elif ret < -target * sl * 0.999:
    label = -1.0  # DOWN (損切り)
else:
    label = 0.0  # RANGE (時間切れ、微小変動)
```

- **Volatility-Based Thresholds**: バリア幅は固定値ではなく、過去 24 時間のローリングボラティリティ(`returns.rolling(24).std()`)に基づいて動的に調整されます。
- **min_ret フィルタ**: ボラティリティが極小(`< 0.0001`)のイベントは除外され、ノイズラベルの混入を防ぎます。

### 6.3 実装の特徴

| 項目                   | 内容                                                                                                       |
| :--------------------- | :--------------------------------------------------------------------------------------------------------- |
| **ファイル**           | `backend/app/utils/label_generation/triple_barrier.py`                                                     |
| **統合先**             | `backend/app/services/ml/label_cache.py` (`ThresholdMethod.TRIPLE_BARRIER`)                                |
| **動的バリア幅**       | 過去 24 時間のボラティリティを使用 (`volatility = returns.rolling(24).std()`)                              |
| **PT/SL 比率**         | `pt_multiplier` と `sl_multiplier` を Optuna で最適化可能（例: `threshold=1.5` → 1.5σ の変動で判定）       |
| **テスト**             | `backend/tests/utils/label_generation/test_triple_barrier.py` で利確/損切り/時間切れの全パターンを検証済み |
| **Purged K-Fold 対応** | `LabelCache.get_t1()` メソッドで各サンプルのラベル確定時刻(t1)を取得し、情報リークを完全に防止             |

### 6.4 期待される効果

- ✅ **リアルな勝率の学習**: 実際のトレードで使用する「利確 2%、損切り 1%」などのルールをモデルが直接学習できる。
- ✅ **早期利確の捕捉**: トレンドが早い段階で発生した場合、固定ホライズンを待たずにラベルが確定するため、シグナルの遅延が減少。
- ✅ **レンジ相場の明確化**: PT/SL のどちらにも触れない「往復ビンタ」パターンを `RANGE (0)` として明示的に学習。

### 6.5 使用例 (Optuna 最適化)

```python
# backend/scripts/ml_optimization/run_ml_pipeline.py より
trial.suggest_categorical("threshold_method", ["QUANTILE", "KBINS_DISCRETIZER", "TRIPLE_BARRIER"])
if threshold_method == "TRIPLE_BARRIER":
    threshold = trial.suggest_float("threshold", 0.5, 3.0)  # 0.5σ ~ 3.0σ
```

最適化結果(例):

```json
{
  "threshold_method": "TRIPLE_BARRIER",
  "threshold": 1.8,
  "horizon_n": 6
}
```

→ **1.8σ の変動** を利確/損切りラインとし、**6 時間** 以内にバリアに触れない場合は RANGE と判定。

## 7. 今後のロードマップ (Next Steps)

モデルの「守り（レンジ回避）」「信頼性（検証）」「ラベル品質（TBM）」が確立されました。次は「攻め（トレード機会の増加）」に焦点を当てます。

1. **メタラベリング (Meta-Labeling)**:

   - 1 次モデルが「エントリー」と判断したシグナルに対し、2 次モデルが「勝てる確率」を判定する構成にする。
   - 期待効果: サイズの大きなベットを行うべき局面の選定。

2. **特徴量の追加調査**:
   - On-chain データや Sentiment データの導入検討。

## 8. ファイル構造 (Key Files)

- `backend/scripts/ml_optimization/run_ml_pipeline.py`: パイプラインのエントリーポイント。Optuna 最適化、OOF 生成、スタッキング実行を統括。
- `backend/app/services/ml/feature_engineering/`: 特徴量計算ロジック。
  - `funding_rate_features.py`: FR 加工ロジック。
  - `advanced_features.py`: MTF, RANGE 指標など。
- `backend/app/services/ml/models/`: モデル定義。
  - `gru_model.py`, `lstm_model.py`: PyTorch 実装の DL モデル。
- `backend/app/services/ml/stacking_service.py`: Ridge Regression によるメタ学習器。
- `backend/app/utils/purged_cv.py`: Purged K-Fold の実装。
- `backend/app/utils/label_generation/triple_barrier.py`: Triple Barrier Method の実装。
