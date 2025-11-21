# 低計算リソース環境下における暗号資産トレーディングMLモデルのための高度な特徴量エンジニアリングとラベリング戦略

本ドキュメントは、暗号資産を対象とした金融時系列予測において、**計算資源が限られたソロ開発者**でも実務的に運用可能な機械学習ワークフローを構築するための指針を整理したものである。  
特に以下を中心に論じる。

- 金融時系列特有の問題（非IID性、非定常性、ボラティリティクラスタリング 等）
- トリプルバリアメソッド（Triple Barrier Method; TBM）およびメタラベリングによるラベリング戦略
- HMM・分数階差分を含む特徴量エンジニアリング
- LightGBM等を用いた低コストな特徴量選択
- 時系列に適したクロスバリデーションとリーケージ防止
- Optunaを用いた効率的なハイパーパラメータ最適化
- それらを統合した「低コスト・高効率」パイプライン

---

## I. 序論：金融時系列予測における特異性と「低コンピューティング能力」の制約

### I.A. 標準的MLの前提の崩壊：IID仮定の不適合

画像認識や自然言語処理など、多くの標準的な機械学習タスクは「独立同分布（IID）」仮定に依存している。  
しかし、暗号資産を含む金融時系列データはこの仮定から大きく逸脱する。

代表的な特性は以下の通りである。

1. **自己相関（Autocorrelation）**  
   過去のリターンが将来のリターンに影響を与える、あるいはその逆の関係が存在する。
2. **ボラティリティ・クラスタリング（Volatility Clustering）**  
   高ボラティリティ期間と低ボラティリティ期間がクラスターとして現れやすい。
3. **非定常性（Non-stationarity）**  
   平均・分散などの統計的性質が時間とともに変化する。

このような特性により、「独立なサンプル」を前提とした一般的なクロスバリデーション（CV）は不適切となり、特に**バックテストおよび汎化性能評価に深刻な歪みや情報リーケージ**を引き起こす。

### I.B. 低コンピューティング能力という制約の本質

個人開発者にとっての「低コンピューティング能力」は、単に学習時間の長さの問題ではない。  
これは次のような、本質的な設計制約として理解すべきである。

- 利用可能なアルゴリズム・検証手法の選択範囲を制限する
- 高コストな手法（例：Combinatorial Purged CV (CPCV)）は、理論上優れていても現実的に採用できない
- 最も重大なリスクは**「バックテスト過学習」に有限リソースを投じてしまうこと**

とりわけ、時系列構造を無視した標準的K-Fold CVは、「未来データの情報を訓練データに混入させる（データリーケージ）」ことで、**非現実的に高い性能値を示す「幻想的モデル」**を生み出しやすい。この誤ったモデルに対してハイパーパラメータ最適化を実行すると、限られた計算資源を無駄に消費する。

本レポートの目的は、この制約条件下でも実行可能な形で

- リーケージを厳格に防止しつつ
- リスク管理を内包したラベリングを行い
- 有効な特徴量集合を構築し
- 計算コストに見合うHPOと検証を行う

という**「生存可能なワークフロー」**を提示することである。

---

## II. 高度なラベリング戦略：予測対象（ターゲット）の定義

モデル性能は、「何を予測させるか（ラベル定義）」によって本質的に規定される。本節では、代表的手法の問題点と、TBM・メタラベリングによる解決策を整理する。

### II.A. 固定時間ホライズン法（Fixed-Time Horizon）の罠

多くの入門記事で紹介されるラベリング手法は、`pandas.DataFrame.shift(-n)` を用いて n 期間後の価格変化に基づきラベル付けする **固定時間ホライズン法**である。

典型例：

- n期間後の価格が閾値 \(+X\%\) 以上上昇 → ラベル \(+1\)（買い）
- n期間後の価格が \(-X\%\) 以下下落 → ラベル \(-1\)（売り）
- それ以外 → ラベル \(0\)（中立）

この手法は実装が容易である一方、暗号資産市場における実務利用には以下の致命的欠陥がある。

1. **ボラティリティへの非適応性**  
   - 平常時の \(1\%\) と暴落局面の \(1\%\) は意味が異なる。
   - 固定閾値（例：\(\pm1\%\)）は市場ボラティリティを無視する。

2. **価格経路の無視**  
   - n期間後に最終的に +1% であっても、その途中で -50% を記録していれば実運用では強制ロスカットされている可能性が高い。
   - 終値のみを参照するため、**清算リスクやドローダウンを考慮しない非現実的なラベル**となる。

3. **非効率なサンプリング**  
   - 時間バー（例：1時間足）で一律にサンプリングするが、情報量は時間当たりで均一でない。
   - 高流動時間帯と閑散時間帯を同一に扱うことは、効率的ではない。

### II.B. トリプルバリアメソッド（TBM）によるリスク管理一体型ラベリング

上記の欠陥を克服するため、Marcos Lopez de Prado による **トリプルバリアメソッド（Triple Barrier Method; TBM）** の導入が推奨される。

TBMでは、各エントリー時点に対して以下の3種のバリアを設定する。

1. **上部バリア（Profit-Take）**：利確ライン
2. **下部バリア（Stop-Loss）**：損切りライン
3. **垂直バリア（Time-Limit）**：最大保有期間（時間切れ）

ラベル付けは「最初に到達したバリア」に基づき決定される。

- 上部バリア到達 → ラベル \(+1\)
- 下部バリア到達 → ラベル \(-1\)
- どちらも到達せず垂直バリア到達 → その時点のリターンに基づき \(+1/-1/0\) を判定

TBMの核心は、**上部・下部バリアを「直近ボラティリティ」に応じて動的設定する点**にある。

例：

- ATRの複数倍などを用い、相場の荒れ具合に応じてバリア幅を自動調整
- 静穏相場ではタイトなバリア、荒い相場では広いバリア

これにより、ラベル生成の段階でリスク管理とボラティリティ適応が組み込まれ、固定時間ホライズン法の多くの問題を解消できる。

### II.C. メタラベリング（Meta-Labeling）による計算リソース最適化

TBMは方向性のラベリングに有効だが、**低リソース環境**ではさらなる効率化が必要となる。そこで導入されるのが **メタラベリング**である。

メタラベリングの構成:

1. **プライマリモデル（主モデル）の構築**  
   - 計算コストが非常に低いルールベースや単純な指標を用いる（例：ゴールデンクロス、CUSUMフィルター等）。
   - 目的は「方向性シグナルを大量に生成すること」であり、高再現率（Recall）重視で偽陽性を許容する。

2. **メタラベルの生成**  
   - プライマリモデルの各シグナルに対し、TBMを適用。
   - バリア内で有益だったシグナル → ラベル \(1\)  
     有益でなかったシグナル → ラベル \(0\)
   - これがメタモデルの教師ラベルとなる。

3. **セカンダリモデル（メタモデル）の学習**  
   - LightGBMやXGBoostなど、計算コストの高いモデルはこの二値分類問題のみに集中投入する。
   - タスクは「主モデルが出したシグナルのうち、どれが本物かを判定すること」。

この枠組みにより、

- 難しい「市場方向予測」ではなく
- 「偽陽性フィルタリング」という比較的単純だが価値の高い問題

に計算リソースを集中できる。

### II.D. ラベリング戦略の比較（テーブル1）

**テーブル1：ラベリング戦略の比較**

| 戦略                     | 概要                                           | リスク管理                   | ボラティリティ適応          | 計算コスト | 主な課題・コメント                                        |
|--------------------------|----------------------------------------------|------------------------------|-----------------------------|------------|---------------------------------------------------------|
| 固定時間ホライズン法    | n期間後の価格変動が固定閾値を超えるかで判定 | 組み込まれていない           | 不可（固定閾値）            | 低         | 経路無視、市場状況無視で非現実的。過学習リスク高。      |
| トリプルバリアメソッド（TBM） | 利確・損切り・時間切れバリアでラベリング   | 組み込み済み（損切りバリア） | 可（動的バリア設定）        | 中         | イベント開始条件（エントリー条件）の設計が必要。        |
| メタラベリング           | プライマリシグナルの成否を二値分類          | TBM経由で組み込み済み        | TBM経由で適応可能           | 中〜高     | 偽陽性のフィルタリングに特化し、高精度化に有効。        |

---

## III. 予測シグナルの構築（特徴量エンジニアリング）

適切なラベル定義の後は、それを予測可能にするシグナル（特徴量）の設計が重要となる。

### III.A. 基盤特徴量：テクニカル指標と価格派生量

計算コストが低く有用な基本特徴量は次の通り。

- **TA-Lib ベース指標**（RSI, MACD, ボリンジャーバンド, ADX, ATR 等）
- **pandas 派生特徴量**
  - `pct_change(periods=n)`：n期間リターン
  - `diff(periods=n)`：n期間差分
  - `rolling(window=n).mean(), .std()`：移動平均・移動標準偏差

これらは**低コストかつ情報量の高い基盤特徴量群**として有用である。

### III.B. 状態特徴量：市場レジーム検出（hmmlearn の活用）

`hmmlearn` を用いた隠れマルコフモデル（HMM）は、市場状態（レジーム）の潜在構造を特徴量として抽出する手段となる。

実装の要点:

1. `GaussianHMM` を用い、`n_components` にレジーム数（例：2〜3）を指定。
2. 観測データとして日次リターンなどを用いモデルを学習。
3. `predict` により、各時点の隠れ状態列（例：`[0,0,1,1,0,...]`）を取得。
4. 得られた状態インデックスをカテゴリ特徴量として追加。

この「レジーム特徴量」は、他の指標に市場文脈を与える。ツリーモデル（LightGBM等）は、

- 「高ボラティリティ状態におけるRSIの解釈」
- 「低ボラティリティ状態でのトレンドフォロー有効性」

などを自動的に学習可能であり、単純なテクニカル指標単独より高い表現力を持つ。

### III.C. 定常化：非定常性への対策と「記憶」の保持

金融時系列の非定常性は、モデル安定性を著しく損なう。  
単純な1次差分（`pct_change(1)` 等）は定常化に有効だが、**長期トレンドやメモリ効果を破壊してしまう**。

推奨されるのは、**分数階差分（Fractional Differencing）**である。

- パラメータ \(d \in [0,1]\) を調整し
- ADF検定等で定常性を満たしつつ
- 元系列との相関（記憶）を最大限維持する

分数階差分はナイーブ実装では \(O(n^2)\) だが、FFT等を用いた \(O(n \log n)\) の高速アルゴリズムが存在し、低リソース環境でも現実的に適用可能である。

**推奨実務フロー**

- 価格・出来高など主要系列に分数階差分を適用
- その上でTA-Lib指標を構成
- 結果として、**定常性と情報保持を両立した特徴量集合**を得る

### III.D. 外部特徴量：センチメント情報の導入

暗号資産市場はセンチメント依存性が高いため、外部指標の活用も有効である。

- 例：`Fear & Greed Index`（alternative.me 提供）
- 専用ライブラリやAPIから取得し、OHLCVデータにマージして特徴量化

これにより、市場心理の極端な状態（恐怖・強欲）が価格形成に与える影響をモデルに反映できる。

---

## IV. 低リソース環境下での特徴量選択と重要度評価

多数の特徴量を生成した後、そのまま学習させると

- 訓練時間の増大
- 次元の呪いによる過学習

を招くため、適切な特徴量選択が必須となる。

### IV.A. 特徴量選択手法の分類とトレードオフ

計算コスト観点から、代表的手法は以下に分類される。

1. **Wrapper法（例：RFE）**
   - モデルを繰り返し訓練しながら特徴量サブセットを評価
   - 特徴量間相互作用を考慮できるが、**計算コスト極大**で低リソース環境では不適

2. **Filter法**
   - 学習前に統計量（分散、相関、F値、カイ二乗 等）のみで特徴量を評価
   - 非常に高速だが、**特徴量間相互作用を無視**

3. **Embedded法**
   - LightGBMやXGBoost等のモデルが提供する特徴量重要度を利用
   - 一度の学習で相互作用も含め評価可能で、コスト効率が高い

### IV.B. LightGBMの特徴量重要度を用いた低コスト選択

`lightgbm>=4.0.0` を前提とした推奨ワークフロー：

1. **Filterステップ**  
   - `VarianceThreshold` 等でゼロ分散特徴量を除去。

2. **Embeddedステップ（LightGBM）**
   - 全特徴量を用いて一度モデルを学習（パラメータはデフォルト近傍で可）。
   - `model.feature_importance(importance_type="gain")` を取得。
   - 上位 N 個（例：50）の特徴量のみを残す。

3. **HPO対象の縮小**
   - 最も計算コストの高いHPOは、この選別済み特徴量集合に対してのみ実施。

`importance_type` の意味:

- `split`：分岐に使用された回数（高カーディナリティに偏りやすい）
- `gain`：損失改善量の総和（モデル貢献度を直接反映し、より信頼性が高い）

### IV.C. 特徴量選択手法の比較（テーブル2）

**テーブル2：特徴量選択手法のトレードオフ分析**

| 手法                    | 選択ロジック                 | 特徴量間相互作用 | 計算コスト              | 過学習リスク | ソロ開発者への推奨度                         |
|-------------------------|------------------------------|------------------|-------------------------|--------------|----------------------------------------------|
| Filter法（SelectKBest） | 単変量統計量に基づく評価     | 考慮しない       | 非常に低い              | 低           | 初期フィルタとして有用                       |
| Wrapper法（RFE）        | 再帰的なモデル再学習と評価   | 考慮する         | 非常に高い              | 高           | 非推奨（計算資源を大きく消費）               |
| Embedded法（LGBM Gain） | 学習過程から重要度を取得     | ツリー構築過程で考慮 | モデル1回分（実質低コスト） | 中           | 強く推奨（性能とコストのバランスが最適）     |

---

## V. 堅牢なバックテストとリーケージ防止

適切なラベルと特徴量があっても、評価手法を誤れば全てが無意味となる。

### V.A. 標準的クロスバリデーションの失敗とリーケージ

時系列データに対して標準的K-Fold CVを適用すると、

- データシャッフルにより時間順序が崩壊
- 「未来の情報」で「過去」を予測する構造になりうる

結果として**致命的な情報リーケージ**が発生し、過大評価されたスコアに基づく誤った意思決定を招く。

### V.B. TimeSeriesSplit と gap パラメータの重要性

`sklearn.model_selection.TimeSeriesSplit` はウォークフォワード（Walk-Forward）型のCVを提供し、時系列順序を保持できる。

しかし、デフォルト設定（`gap=0`）のまま **TBMラベル** と組み合わせると、以下の問題が発生する。

- TBMは最大保有期間 \(N\) を参照してラベルを付与するため、時点 \(t\) のラベルには \(t+1 \sim t+N\) の将来情報が含まれる。
- `gap=0` の場合、訓練期間末尾のラベルが、直後のテスト期間の情報を既に利用している。
- 結果として、**訓練セットがテスト期間の情報で汚染されるリーケージ**が発生。

**解決策（実装は削除済み）:**

- `gap` を TBMの最大垂直バリア期間 \(N\) 以上に設定する。

```python
from sklearn.model_selection import TimeSeriesSplit

# 過去の設計ではts_cv = TimeSeriesSplit(n_splits=5, gap=N) のように gap 設定が検討されたが、
# 現在のTrdingerプロジェクトのML実装ではこの gap パラメータは使用されていません。
# より簡素な時系列分割によってリーケージ対策を行っています。
```

これにより、訓練期間末尾とテスト開始の間に「情報隔離帯（Embargo）」が生じ、TBM由来の先読みリークを防止できる。

### V.C. Purged K-Fold の概念（ゴールドスタンダード）

Lopez de Prado による **Purged K-Fold CV** は、

- K-Foldのような複数分割によるロバスト性
- テスト期間と重複する訓練データの「パージ」によるリーケージ防止

を両立する手法であり、理論的には望ましい。ただし、

- 実装が複雑
- 計算コストもTimeSeriesSplitより高い

ため、低リソース環境ではまず

- `TimeSeriesSplit` を正しく設定し、現状のTrdingerプロジェクトではより簡素な時系列分割によってリーケージ対策を行っている。
- `n_splits` を適度に増やしてロバスト性を確保

するアプローチが現実解となる。

### V.D. 時系列CV手法の比較（テーブル3）

**テーブル3：時系列クロスバリデーション手法の比較（TBMラベル使用時）**

| 手法                        | リーケージリスク（TBM使用時）                | 頑健性（ロバスト性） | 計算コスト      | ソロ開発者への推奨度                                               |
|-----------------------------|----------------------------------------------|----------------------|-----------------|--------------------------------------------------------------------|
| Standard K-Fold             | 致命的（未来データで訓練）                   | 低                   | 低              | 絶対に使用禁止                                                     |
| TimeSeriesSplit（gap=0）    | 致命的（TBMラベル由来の先読みリーク）       | 非常に低い（単一経路） | 低              | 危険（過学習した幻想的結果を生むが、現在の実装ではこの問題は発生しない） |
| TimeSeriesSplit（gap > N） | 安全（垂直バリア長以上の隔離により防止）     | 低〜中               | 低              | 推奨（検討されたが、現在の実装では使用されていない）             |
| Purged K-Fold               | 安全（パージ処理により防止）                | 中〜高               | 中〜高          | 上級者向け（実装・計算コストともに重い）                         |

---

## VI. 効率的なハイパーパラメータ最適化（HPO）ワークフロー

低リソース環境においては、**不要な試行を早期に打ち切る仕組み**がHPOの鍵となる。Optunaはこの要件に適合する。

### VI.A. Optuna とプルーニング（枝刈り）の重要性

HPOの主要コストは、「性能の低いハイパーパラメータセット」を最後まで学習させてしまうことにある。

Optunaは以下を通じてこれを削減する。

- 中間評価値に基づく **プルーニング（Pruning）**
- LightGBM / XGBoost との統合による自動的な枝刈り

これにより、限られた時間で探索可能な試行数を実質的に増大させる。

### VI.B. LightGBMPruningCallback を用いた実装例

以下は、OptunaとLightGBMを連携させる実装フローの例である。

```python
import optuna
from optuna.integration import LightGBMPruningCallback
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score  # 必要に応じて f1_score, log_loss 等に変更
import numpy as np

# X_train, y_train, X_val, y_val は事前に構築しておく。
# より堅牢にする場合、TimeSeriesSplit を用いたループ内で評価する。

def objective(trial):
    # 1. プルーニング用コールバック
    pruning_callback = LightGBMPruningCallback(
        trial,
        "binary_logloss",
        valid_name="valid_1",
    )

    # 2. ハイパーパラメータ空間の定義
    param_grid = {
        "objective": "binary",
        "metric": "binary_logloss",
        "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 20, 300),
        # 必要に応じて max_depth, min_data_in_leaf 等を追加
    }

    # 3. モデル学習
    model = lgb.LGBMClassifier(**param_grid)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_names=["train", "valid_1"],
        eval_metric="binary_logloss",
        early_stopping_rounds=100,
        callbacks=[pruning_callback],
    )

    # 4. 評価指標（Optuna が最適化対象とする値）を返却
    score = model.best_score_["valid_1"]["binary_logloss"]
    return score

# 5. study の作成と実行
study = optuna.create_study(
    direction="minimize",
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
)

study.optimize(objective, n_trials=100)

print("Best trial:", study.best_trial.params)
```

このフローにより、

- 中間結果が悪い試行は早期終了
- 有望な試行のみ最後まで実行

という形で探索効率を最大化できる。

---

## VII. 結論：ソロ開発者のための低コスト・高効率MLワークフロー

本ドキュメントで整理した概念を、**暗号資産トレーディングモデルを構築するソロ開発者向けの統合パイプライン**として再構成する。

### ステップ1：データ収集とラベリング

- OHLCVデータを収集し、`Fear & Greed Index` 等の外部特徴量をマージ。
- `pandas.shift()` を用いた固定時間ホライズンラベルは使用しない。
- **トリプルバリアメソッド（TBM）**を実装し、バリアはボラティリティ（例：ATR）に基づき動的設定。
- 計算コストの低いCUSUMフィルターやシンプルなTAシグナルをプライマリモデルとし、**メタラベリング**を導入。  
  ターゲット \(y\) は \([1]=\mathrm{True\ Positive}, [0]=\mathrm{False\ Positive}\) の二値問題とする。

### ステップ2：特徴量エンジニアリング

- TA-LibでRSI, MACD, ATR等の基礎指標を生成。
- `hmmlearn` の GaussianHMM により市場レジーム特徴量を追加。
- 価格・出来高系列に分数階差分（FFT等で高速化）を適用し、定常性と記憶保持を両立。

### ステップ3：特徴量選択（低コスト手法に限定）

- `VarianceThreshold` 等でゼロ分散特徴量を除去。
- Wrapper法（RFE等）のような高コスト手法は使用しない。
- 全特徴量でLightGBMを一度訓練し、`importance_type="gain"` に基づき上位N個（例：50個）を採用。これを最終特徴量集合 \(X\) とする。

### ステップ4：ハイパーパラメータ最適化と検証

- Optunaの `objective` 関数を定義し、LightGBMと連携。
- クロスバリデーションには `TimeSeriesSplit` を使用。
- **現在のTrdingerプロジェクトのML実装では、TBMのような複雑なラベリング手法は採用されておらず、それに伴う `gap` パラメータも使用されていません。** 時系列順序を保持した簡素な分割でリーケージ対策を行っています。
- `LightGBMPruningCallback` を導入し、見込みの薄い試行を早期枝刈り。
- `study.optimize()` により制約時間内で効率的に探索。

### ステップ5：最終モデルの学習とデプロイ

- `study.best_params` を取得し、選別済み特徴量 \(X\) と確定ラベル \(y\) の全期間データで最終モデルを再学習。
- 得られたモデルを実運用（デプロイ）用モデルとして採用。

この一連のフローにより、

- リーケージ防止
- リスク管理一体型ラベリング
- レジーム認識・分数階差分等を用いた高品質特徴量
- LightGBM重要度とOptunaプルーニングによる計算効率化

を同時に実現し、**低リソース環境でも再現性と堅牢性を備えた暗号資産トレーディングMLモデル開発**が可能となる。

---

## 参考文献

以下は原文に基づく参考文献一覧であり、新規の出典追加は行っていない。

- 金融時系列予測におけるクロスバリデーション手法 - GMO ..., <https://recruit.group.gmo/engineer/jisedai/blog/cv-in-financial-time-series/>
- Time Series Data Labeling​ - Budgie Bytes & Bucks, <https://yellowplannet.com/time-series-data-labeling/>
- Label it your way…. Pandas is the secret sauce of financial… | by Denny Joseph, CFA | Medium, <https://medium.com/@quant_views/label-it-your-way-9a139eafa651>
- The Triple Barrier Method: Labeling Financial Time Series for ML in Elixir | by Yair Oz, <https://medium.com/@yairoz/the-triple-barrier-method-labeling-financial-time-series-for-ml-in-elixir-e539301b90d6>
- You shouldn't need a PhD to understand Financial Machine ... , <https://ai.plainenglish.io/advances-in-financial-machine-learning-for-dummies-part-0-c08e169335f>
- python Archives - Hudson & Thames, <https://hudsonthames.org/tag/python/>
- Machine Learning Trading Bot for Interactive Brokers in Python - YouTube, <https://www.youtube.com/watch?v=cGPtRrOOKuw>
- Labeling Stock Prices for ML with Triple Barrier Methods - GuruFinance Insights, <https://ayratmurtazin.beehiiv.com/p/labeling-stock-prices-for-ml-with-triple-barrier-methods>
- Enhanced Genetic-Algorithm-Driven Triple Barrier Labeling Method and Machine Learning Approach for Pair Trading Strategy in Cryptocurrency Markets - MDPI, <https://www.mdpi.com/2227-7390/12/5/780>
- Stock Price Prediction Using Triple Barrier Labeling and Raw OHLCV Data: Evidence from Korean Markets - arXiv, <https://arxiv.org/html/2504.02249v2>
- Alternative Bars in Alpaca: Part III - (Machine Learning Trading Strategy), <https://alpaca.markets/learn/alternative-bars-03>
- Optimal Trend Labeling in Financial Time Series - ResearchGate, <https://www.researchgate.net/publication/372976059_Optimal_Trend_Labeling_in_Financial_Time_Series>
- TA lib with Python and Pandas - YouTube, <https://www.youtube.com/watch?v=MQyyATi3_Vs>
- Documentation — Technical Analysis Library in Python 0.1.4 ..., <https://technical-analysis-library-in-python.readthedocs.io/en/stable/ta.html>
- Introduction to using of TA-Lib - Kaggle, <https://www.kaggle.com/code/sndorburian/introduction-to-using-of-ta-lib>
- pandas.DataFrame.pct_change — pandas 2.3.3 documentation, <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.pct_change.html>
- pandas.DataFrame.pct_change — dev docs, <https://pandas.pydata.org/docs/dev/reference/api/pandas.DataFrame.pct_change.html>
- Step-by-Step Python Guide for Regime-Specific Trading Using HMM ..., <https://blog.quantinsti.com/regime-adaptive-trading-python/>
- Market Regime Detection using Hidden Markov Models in QSTrader ..., <https://www.quantstart.com/articles/market-regime-detection-using-hidden-markov-models-in-qstrader/>
- Regime-Switching Factor Investing with Hidden Markov Models - MDPI, <https://www.mdpi.com/1911-8074/13/12/311>
- Market regime detection using Statistical and ML based approaches | Devportal, <https://developers.lseg.com/en/article-catalog/article/market-regime-detection>
- Fractional differencing as viewed from the frequency domain | by NTTP - Medium, <https://medium.com/@nttp/fractional-differencing-as-viewed-from-the-frequency-domain-8ff33741408f>
- PyELW: Exact Local Whittle Estimation for Long Memory Time Series in Python - Jason Blevins, <https://jblevins.org/research/pyelw.pdf>
- Frequency response of fractional differencing and integration, <https://diffent.com/mcrtrain/freqrespfracdiffV2.pdf>
- Machine Learning Trading Essentials (Part 2): Fractionally differentiated features, Filtering, and Labelling - Hudson & Thames, <https://hudsonthames.org/machine-learning-trading-essentials-part-2-fractionally-differentiated-features-filtering-and-labelling/>
- Fractional Differencing Implementation (FD Part 3) - Kid Quant, <http://kidquant.blogspot.com/2019/03/fractional-differencing-implementation.html>
- feargreed_crypto - GitHub Pages, <https://dk81.github.io/dkmathstats_site/feargreed_crypto.html>
- Fear & greed index: useful statistics to integrate it in a trading strategy - Barbotine - Medium, <https://barbotine.medium.com/fear-greed-index-and-how-to-ue-it-with-live-market-data-d4a108105695>
- kukapay/crypto-feargreed-mcp - GitHub, <https://github.com/kukapay/crypto-feargreed-mcp>
- Cryptocurrency Fear & Greed Index Trading Strategy with Python - YouTube, <https://www.youtube.com/watch?v=vCRSU9Cqzxg>
- fear-and-greed-crypto · PyPI, <https://pypi.org/project/fear-and-greed-crypto/>
- Diabetes Prediction Using Feature Selection Algorithms and Boosting-Based Machine Learning Classifiers - PMC, <https://pmc.ncbi.nlm.nih.gov/articles/PMC12563305/>
- Recursive Feature Elimination (RFE): Working, Advantages & Examples - Analytics Vidhya, <https://www.analyticsvidhya.com/blog/2023/05/recursive-feature-elimination/>
- Recursive Feature Elimination (RFE) for Feature Selection in Python - MachineLearningMastery.com, <https://machinelearningmastery.com/rfe-feature-selection-in-python/>
- Recursive Feature Elimination (RFE) Made Simple: How To Tutorial - Spot Intelligence, <https://spotintelligence.com/2024/11/18/recursive-feature-elimination-rfe/>
- Ensemble Machine Learning Models Utilizing a Hybrid Recursive Feature Elimination (RFE) Technique for Detecting GPS Spoofing Attacks Against Unmanned Aerial Vehicles - MDPI, <https://www.mdpi.com/1424-8220/25/8/2388>
- 1.13. Feature selection — scikit-learn 1.7.2 documentation, <https://scikit-learn.org/stable/modules/feature_selection.html>
- VarianceThreshold — scikit-learn 1.7.2 documentation, <https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html>
- Filter Methods | Codecademy, <https://www.codecademy.com/article/fe-filter-methods>
- SelectKBest — scikit-learn 1.7.2 documentation, <https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html>
- LightGBM — ELI5 0.15.0 documentation, <https://eli5.readthedocs.io/en/latest/libraries/lightgbm.html>
- feature_importances split vs gain: a demo - Kaggle, <https://www.kaggle.com/code/marychin/feature-importances-split-vs-gain-a-demo>
- LightGBM Feature Importance and Visualization - GeeksforGeeks, <https://www.geeksforgeeks.org/machine-learning/lightgbm-feature-importance-and-visualization/>
- Insight of Feature Importance: Split and Gain - Kaggle, <https://www.kaggle.com/code/masatakasuzuki/insight-of-feature-importance-split-and-gain>
- 3.1. Cross-validation: evaluating estimator performance - scikit-learn, <https://scikit-learn.org/stable/modules/cross_validation.html>
- Data Leakage Basics, with Examples in Scikit-Learn | by Moshe Sipper, Ph.D. | AI Mind, <https://pub.aimind.so/data-leakage-basics-with-examples-in-scikit-learn-9c946a6f75b2>
- Time-related feature engineering — scikit-learn 1.7.2 documentation, <https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html>
- Understanding TimeSeriesSplit Cross-Validation for Time Series Data | by KoshurAI, <https://koshurai.medium.com/understanding-timeseriessplit-cross-validation-for-time-series-data-4c232cc4f844>
- Avoiding Data Leakage in Time Series Analysis with TimeSeriesSplit - CodeCut, <https://codecut.ai/cross-validation-with-time-series/>
- TimeSeriesSplit — scikit-learn 1.7.2 documentation, <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html>
- Advances in Financial Machine Learning | Wiley, <https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086>
- How to use 'purging' in predicting stock price tomorrow based on information today? - Quant StackExchange, <https://quant.stackexchange.com/questions/61079/how-to-use-purging-in-predicting-stock-price-tomorrow-based-on-information-tod>
- Advances in Financial Machine Learning Summary - Marcos, Shortform PDF, <https://www.shortform.com/pdf/advances-in-financial-machine-learning-pdf-marcos-lopez-de-prado>
- Lopez de Prado, Advances in Financial Machine Learning, 1e, <https://app.perusall.com/catalog/book/9781119482109>
- TSCV: A Python package for Time Series Cross-Validation : r/algotrading - Reddit, <https://www.reddit.com/r/algotrading/comments/boybys/tscv_a_python-package-for-time-series/>
- Python: How to retrieve the best model from Optuna LightGBM study? - Stack Overflow, <https://stackoverflow.com/questions/62144904/python-how-to-retrieve-the-best-model-from-optuna-lightgbm-study>
- Optuna tutorial for hyperparameter optimization - Kaggle, <https://www.kaggle.com/code/corochann/optuna-tutorial-for-hyperparameter-optimization>
- optuna.integration.LightGBMPruningCallback — Optuna 2.0.0 docs, <https://optuna.readthedocs.io/en/v2.0.0/reference/generated/optuna.integration.LightGBMPruningCallback.html>
- optuna.integration.LightGBMPruningCallback - Optuna 3.4.1 docs, <https://optuna.readthedocs.io/en/v3.4.1/reference/generated/optuna.integration.LightGBMPruningCallback.html>
- optuna.integration.lightgbm ソースコード, <https://optuna.readthedocs.io/en/v2.0.0/_modules/optuna/integration/lightgbm.html>
- Accuracy measure in Optuna/LightGBM - Stack Overflow, <https://stackoverflow.com/questions/73894656/accuracy-measure-in-optuna-lightgbm>