# ML Fakeout Detection Implementation Plan

## 1. 概要

現在の「トレンド/レンジ予測（方向予測）」モデルから、**「ダマシ検知（Fakeout Detection / Meta-Labeling）」モデル**への転換を計画します。
このアプローチでは、ML モデルは市場の方向を予測するのではなく、**「特定のシグナル（ブレイクアウト等）が成功するか、ダマシで終わるか」**を判定することに特化します。

## 2. 現状分析

- **特徴量 (`market_data_features.py`)**:
  - `OI_RSI`, `OI_Price_Divergence`, `FR_OI_Sentiment` など、ダマシ検知に極めて有効な OI/FR 関連の特徴量は既に実装されています。
- **ラベリング (`triple_barrier.py`, `label_cache.py`)**:
  - `TripleBarrier` クラスは実装済みで、`binary_label=True` によりメタラベリング（成功=1, 失敗=0）に対応可能です。
  - **課題**: 現在の `LabelCache.get_labels` は `t_events=close_prices.index` としており、**全ての足をイベント（エントリー候補）として扱っています**。メタラベリングでは「ベースとなるシグナルが出た足」のみを対象にする必要があります。

## 3. 実装フェーズ

### Phase 1: ベースシグナル（イベント）生成ロジックの実装

ML モデルが判定するための「種」となるシグナルを定義します。

- **タスク**: `SignalGenerator` クラスの実装
- **配置場所**: `backend/app/services/ml/label_generation/signal_generator.py` (新規作成)
- **ロジック案**:
  - **BB Breakout**: ボリンジャーバンド ±2σ のブレイク
  - **Donchian Breakout**: 過去 N 期間の高値/安値更新
  - **Volume Spike**: 平均出来高の X 倍以上の急増
- **出力**: シグナルが発生したタイムスタンプのリスト (`t_events`)

### Phase 2: ラベリングパイプラインの改修

`LabelGenerationService` を修正し、`SignalGenerator` で生成されたイベントに基づいてラベルをフィルタリングします。
**注記**: `LabelCache` 自体は修正せず（全期間のラベルをキャッシュする機能を維持）、サービス層でフィルタリングを行うことでキャッシュ効率と設計の綺麗さを保ちます。

- **修正対象**: `backend/app/services/ml/label_generation/label_generation_service.py`
- **変更点**:
  - `prepare_labels` メソッド内で `SignalGenerator` を呼び出す。
  - `LabelCache` から全期間のラベルを取得した後、`t_events` に含まれるインデックスのみを抽出（フィルタリング）する。

### Phase 3: 特徴量エンジニアリングの最適化

ダマシ検知に特化した特徴量セットを構成します。

- **重点特徴量**:
  - **Volume & OI**: ブレイク時の出来高と OI の変化率（本物は両方増える、ダマシは OI が減る/変わらない）。
  - **Price Action**: ブレイク前の「タメ（Squeeze）」の有無（BB Width の縮小など）。
  - **Volatility**: ATR や標準偏差の急拡大。
- **アクション**: `feature_engineering_service.py` にダマシ検知用の特徴量プリセット（`FAKEOUT_DETECTION_ALLOWLIST`）を定義し、設定で切り替えられるようにする。

### Phase 4: 学習・評価パイプラインの更新

モデルの評価指標と学習プロセスをメタラベリング用に調整します。

- **評価指標**:
  - Accuracy よりも **Precision（適合率）** を重視。「エントリーした時にどれだけ勝てるか」が重要。
  - **F1-Score** もバランスを見るために使用。
- **バックテスト**:
  - 「ベースシグナルのみの損益」 vs 「ML でフィルタリングした後の損益」を比較するスクリプトを作成。

## 4. 具体的なコード変更案

### `SignalGenerator` の実装イメージ

```python
class SignalGenerator:
    def get_bb_breakout_events(self, df: pd.DataFrame, window=20, dev=2.0) -> pd.DatetimeIndex:
        # BB計算
        # 上抜け/下抜けを検知
        # タイムスタンプを返す
        pass
```

### `LabelGenerationService` の修正イメージ

```python
    def prepare_labels(self, ...):
        # 1. 全期間のラベルを取得（キャッシュ利用）
        labels_all = label_cache.get_labels(...)

        # 2. シグナル（イベント）を生成
        signal_gen = SignalGenerator()
        events = signal_gen.get_bb_breakout_events(ohlcv_df)

        # 3. イベント発生時のみにフィルタリング
        # labels_all のインデックスと events の共通部分のみを残す
        valid_events = labels_all.index.intersection(events)
        labels_filtered = labels_all.loc[valid_events]

        return features.loc[valid_events], labels_filtered
```

## 5. 今後のステップ

1.  `backend/app/services/ml/label_generation/signal_generator.py` を作成し、BB ブレイクアウトロジックを実装する。
2.  `LabelGenerationService` を修正し、フィルタリングロジックを組み込む。
3.  既存の OI 特徴量を使って学習を回し、Precision が向上するか確認する。
