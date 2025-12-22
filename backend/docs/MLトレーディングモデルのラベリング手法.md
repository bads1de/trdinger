# **金融機械学習における先進的ラベリング手法とターゲットエンジニアリングに関する包括的研究報告書**

## **1\. 序論：金融機械学習における「正解」の再定義**

金融市場における予測モデリング、いわゆる金融機械学習（Financial Machine Learning）において、アルゴリズムの選定やハイパーパラメータの最適化以上に決定的な影響を与える工程が「ラベリング（Labeling）」である。教師あり学習における「正解データ（Target Variable）」をどのように定義するかという問いは、画像認識や自然言語処理といった他の AI ドメインとは異なり、金融工学においては自明ではない。画像における「猫」のラベルは不変かつ客観的であるが、金融市場における「買いシグナル」は、投資期間、リスク許容度、取引コスト、そして市場環境（レジーム）によってその定義が流動的であるからである。

本報告書は、現代の定量的取引（Quantitative Trading）において採用されている高度なラベリング手法について、既存の実装済み手法（Triple Barrier Method および Trend Scanning）を除外した上で、網羅的かつ詳細に分析を行うものである。具体的には、伝統的な固定期間法（Fixed Time Horizon）の統計的な限界とその克服、高頻度取引（HFT）やヘッジファンドの実務で用いられる内在時間（Intrinsic Time）に基づくイベント駆動型ラベリング、Kaggle などのデータサイエンスコンペティションで洗練されたターゲットエンジニアリング（残差化、中立化、補助ターゲット）、そしてマクロ経済や市場の微細構造変化を捉えるレジーム検出に基づく動的ラベリングについて詳述する。

これらの手法は、金融時系列データ特有の「低い S/N 比（信号対雑音比）」、「非定常性（Non-Stationarity）」、「因果関係の希薄さ」といった課題に対処するために進化してきたものであり、それぞれの数理的背景と実装上の含意を深く理解することが、堅牢なトレーディング戦略構築の鍵となる。

## ---

**2\. 伝統的手法の批判的検討と高度化：ボラティリティ適応型アプローチ**

金融機械学習の黎明期より使用されてきた最も基本的なアプローチは、固定期間法（Fixed Time Horizon Method）である。しかし、現代のクオンツ実務において、単純な固定期間法がそのまま用いられることは稀である。ここでは、その構造的な欠陥を明らかにし、実務レベルに耐えうる改良手法について論じる。

### **2.1 固定期間法（Fixed Time Horizon）の統計的欠陥**

固定期間法は、時点 $t$ における価格 $P\_t$ と、物理的に固定された時間幅 $h$（例：1 日後、10 分後）経過後の価格 $P\_{t+h}$ の変化率（リターン）に基づいてラベル $y\_t$ を生成する 1。

$$r\_{t, t+h} \= \\frac{P\_{t+h} \- P\_t}{P\_t}$$  
一般的には、このリターン $r\_{t, t+h}$ に対して定数の閾値 $\\tau$ を設け、以下のような三値分類（Buy, Sell, Hold）を行う。

$$y\_t \= \\begin{cases} 1 & \\text{if } r\_{t, t+h} \> \\tau \\\\ \-1 & \\text{if } r\_{t, t+h} \< \-\\tau \\\\ 0 & \\text{otherwise} \\end{cases}$$  
このアプローチには、以下の三つの重大な統計的欠陥が存在する。

1. **分散の不均一性（Heteroscedasticity）の無視:** 金融時系列はボラティリティ・クラスタリング（Volatility Clustering）と呼ばれる特性を持ち、分散が時間によって大きく変動する。定数 $\\tau$ を用いると、高ボラティリティ局面では閾値突破が容易となり、ラベル「1」や「-1」が大量発生する一方、低ボラティリティ局面ではラベル「0」が大半を占めることになる。これは、モデルが「シグナルの質」ではなく「市場のボラティリティ」を学習してしまう原因となる 1。
2. **物理時間の非定常性:** 金融市場における情報の流れは、物理的なクロックタイム（時計の時間）とは一致しない。例えば、雇用統計発表直後の 1 分間と、ランチタイムの 1 分間では、価格形成に与える情報の密度が桁違いに異なる。固定期間法は、この情報密度の違いを無視して等間隔にサンプリングするため、情報の希薄な区間のノイズを過剰に評価するリスクがある 4。
3. **経路依存性（Path Dependence）の欠如:** $t$ から $t+h$ の間に価格がどのように推移したかを無視する。例えば、一度ストップロス水準まで下落してから $t+h$ 時点で上昇して終わった場合、実運用ではロスカットされているにもかかわらず、教師データ上は「成功（Buy）」としてラベル付けされる。これはバックテスト結果と実運用成績の乖離（Overestimation）を生む主要因となる（この点は Triple Barrier Method で解決されるが、固定期間法の限界として認識しておく必要がある）3。

### **2.2 動的閾値（Dynamic Thresholds）によるボラティリティ調整**

固定期間法の最大の弱点であるボラティリティへの脆弱性を克服するために導入されたのが、動的閾値（Dynamic Thresholds）である。この手法では、閾値 $\\tau$ を定数ではなく、その時点のボラティリティの関数として定義する 2。

#### **数理的定式化**

まず、時点 $t$ におけるボラティリティ $\\sigma\_t$ を推定する。一般的には、指数加重移動平均（EWMA）や GARCH モデルを用いて、局所的な標準偏差を算出する。

$$\\sigma\_t^2 \= \\lambda \\sigma\_{t-1}^2 \+ (1 \- \\lambda) r\_{t-1}^2$$  
ここで、$\\lambda$ は減衰係数である。この $\\sigma\_t$ を用いて、動的閾値 $\\tau\_t$ を設定する。

$$\\tau\_t \= k \\cdot \\sigma\_t$$  
ここで $k$ は、シグナルの感度を調整するパラメータ（例：0.5 や 1.0）である。ラベリング関数は以下のように修正される。

$$y\_t \= \\begin{cases} 1 & \\text{if } r\_{t, t+h} \> k \\cdot \\sigma\_t \\\\ \-1 & \\text{if } r\_{t, t+h} \< \-k \\cdot \\sigma\_t \\\\ 0 & \\text{otherwise} \\end{cases}$$

#### **統計的利点と実務的含意**

この改良により、モデルが学習するターゲットは「絶対リターン」から「リスク調整後リターン（単位リスクあたりのリターン）」へと質的に変化する。これはシャープレシオ（Sharpe Ratio）の概念をラベリングに組み込むことと同義である。  
高ボラティリティ局面では閾値が拡大するため、単なるノイズによる変動がシグナルとして誤検知されるのを防ぎ、逆に低ボラティリティ局面では閾値が縮小するため、微小だが有意なトレンドを捕捉することが可能になる。これにより、クラスラベルの分布が時間を通じて安定化し（定常性の向上）、機械学習モデルの収束性が改善されることが報告されている 8。

| 特性                     | 固定閾値法 (Static Threshold)   | 動的閾値法 (Dynamic Threshold)         |
| :----------------------- | :------------------------------ | :------------------------------------- |
| **ボラティリティ感応度** | 無視 (ボラティリティ変化に脆弱) | 適応 (ボラティリティに応じて調整)      |
| **高 Vol 時の挙動**      | シグナル多発 (過剰検知)         | 閾値拡大により抑制 (ノイズ除去)        |
| **低 Vol 時の挙動**      | シグナル消失 (見逃し)           | 閾値縮小により検知 (機会損失防止)      |
| **学習対象**             | 方向と大きさ                    | リスク調整後の方向と優位性             |
| **主な用途**             | 基礎的な分析                    | 実践的な ML 戦略、リスク管理重視の戦略 |

### **2.3 ターゲットとしてのシャープレシオと情報レシオ**

さらに一歩進めて、分類問題（0/1 ラベル）ではなく、回帰問題としてボラティリティ調整済みリターンそのものを予測対象とするアプローチもある。

$$y\_t \= \\frac{r\_{t, t+h}}{\\sigma\_t}$$  
このターゲット変数は、局所的なシャープレシオと解釈できる。Jane Street などのマーケットメーカー系コンペティションや高頻度取引では、単なるリターンの最大化ではなく、リスクあたりのリターン（効用関数）の最大化が目的関数となることが多いため、この形式のラベリングは非常に整合性が高い 8。

## ---

**3\. 内在時間（Intrinsic Time）とイベント駆動型ラベリング**

物理的な時間（Chronological Time）の呪縛から逃れ、市場の活動量や価格変動そのものを時間の単位とする「内在時間（Intrinsic Time）」の概念は、ブノワ・マンデルブロ（Benoit Mandelbrot）の研究に端を発し、現代の HFT やアルゴリズム取引において不可欠な視点となっている。ここでは、物理時間バー（Time Bars）に依存しない、イベント駆動型の高度なラベリング手法について解説する。

### **3.1 方向転換（Directional Change: DC）イベントによるラベリング**

Directional Change（DC）は、Dacorogna らによって体系化された概念であり、価格が一定の割合（閾値 $\\theta$）だけ反転した瞬間を「イベント」として定義する 4。物理的な時間間隔でデータをサンプリングするのではなく、市場が「有意な動き」を見せた時のみサンプリングを行うことで、情報の密度を均質化する。

#### **DC イベントの定義とメカニズム**

DC のフレームワークでは、価格変動を「トレンド（Directional Change）」と「オーバーシュート（Overshoot）」の二つの局面に分解する。

1. **Directional Change (DC) Event:**
   - **Downturn Event:** 直近の高値（Last High）から、事前に定義した閾値 $\\theta$（例：0.5%）以上価格が下落した時点で発生。
   - **Upturn Event:** 直近の安値（Last Low）から、閾値 $\\theta$ 以上価格が上昇した時点で発生。
2. **Overshoot (OS) Event:**
   - DC イベントが確定した後、次の逆方向の DC イベントが発生するまでの間の超過的な価格変動。トレンドの持続力を表す。

#### **ラベリングへの応用戦略**

DC を用いたラベリングは、市場の「レジーム転換点」や「トレンドの強さ」を捉えるのに極めて有効である。

- **トレンド持続予測:** DC イベントが確定した時点 $t\_{DC}$ において、その後のオーバーシュート（OS）の大きさや期間を予測する。これは、「トレンドフォロー戦略において、いつ利食いすべきか」という問いに対する直接的な答えとなる。
  - ターゲット変数 $y\_{t}$ \= OS の大きさ（Magnitude）あるいは OS の期間（Duration）。
  - 研究によれば、DC イベントの発生頻度や OS の大きさにはスケーリング則（Scaling Laws）が存在し、これを ML モデルに学習させることで高い予測精度が得られるとされる 11。
- **物理時間ノイズの除去:** レンジ相場で価格が $\\theta$ 未満の変動を繰り返している間は新しいイベントが発生しないため、モデルは無意味なノイズを学習せずに済む。これにより、トレンド構造の本質的な特徴のみを抽出可能となる。

### **3.2 ZigZag インジケーターを用いた「オラクル（Oracle）」ラベリング**

ZigZag インジケーターは、過去のチャート上の主要な高値（Peak）と安値（Trough）を直線で結ぶテクニカル指標である。リアルタイムのトレード判断には「リペイント（未来の価格によって過去の線が書き換わる）」の性質があるため使用できないが、\*\*教師あり学習における「正解ラベル作成（Ground Truth Generation）」\*\*としては極めて強力なツールとなる 12。

#### **オラクル・ラベリングの哲学**

機械学習モデル（特に Deep Learning）の学習において、「もし未来が完全に見えていたとしたら、どこで売買するのが最適だったか？」という理想的な行動を教師データとして与える手法を「オラクル（Oracle）ラベリング」と呼ぶ。ZigZag はこの理想的な転換点を特定するために利用される。

#### **実装アルゴリズム**

Python における実装（pandas や scipy.signal、または専用ライブラリを使用）では、以下のパラメータが重要となる。

- **Deviation（偏差）:** 転換とみなすための最小変動率（例：5%）。この値未満の逆行はノイズとして無視され、既存のトレンドの一部とみなされる。
- **Depth（深度）:** ピーク/ボトムを判定するために参照する最小期間。

**ラベリングプロセス:**

1. 全期間のヒストリカルデータに対し、ZigZag アルゴリズムを適用し、確定した Peak と Trough のリストを作成する。
2. 各時点 $t$ について、その時点が「Trough から Peak に向かう上昇脚（Up Leg）」にあるのか、「Peak から Trough に向かう下降脚（Down Leg）」にあるのかを判定する。
3. **分類タスク:** ラベル $y\_t \\in \\{1, \-1, 0\\}$ を付与する。
   - $y\_t \= 1$: 上昇脚の中にあり、かつ次の Peak までの期待収益が取引コストを上回る場合。
   - $y\_t \= \-1$: 下降脚の中にあり、かつ次の Trough までの期待収益が十分な場合。
   - $y\_t \= 0$: 転換点直前でリスクが高い、または期待収益が低い場合。

#### **機械学習モデルへの適用（推論時の注意）**

学習時（Training）には未来の情報を利用して作成された ZigZag ラベルを使用するが、推論時（Inference）には当然未来は見えない。したがって、モデルのタスクは\*\*「現在の市場特徴量 $X\_t$ から、現在自分が ZigZag のどの脚（Leg）に位置しているかを確率的に推定すること」\*\*になる 15。

これは、画像認識におけるセグメンテーション（このピクセルは猫の一部か否か）に似たタスクであり、単純な次点予測よりも文脈（Context）を考慮した学習が可能になる。特に、トレンドフォロー戦略において「現在はトレンドの中腹なのか、終焉間近なのか」を判断するメタ情報として非常に有用である。

## ---

**4\. メタラベリング（Meta-Labeling）：フィルタリングによる精度向上**

「メタラベリング（Meta-Labeling）」は、Marcos Lopez de Prado によって提唱され、多くのクオンツファンドや実務家の間でデファクトスタンダードとなりつつある高度なラベリングフレームワークである 2。これは、単一のモデルですべてを決定するのではなく、「トレード機会の発見（Signal Generation）」と「実行可否の判断（Signal Filtering）」を分離するという戦略的思想に基づいている。

### **4.1 構造：Primary Model と Secondary Model の分離**

メタラベリングは、役割の異なる二つのモデルを階層的に組み合わせることで機能する。

#### **第 1 段階：Primary Model（方向の決定）**

- **目的:** 高いリコール（High Recall）を実現すること。すなわち、収益機会を可能な限り見逃さないこと。
- **手法:** 任意のモデルやルールが使用可能。
  - 伝統的なテクニカル指標（RSI のダイバージェンス、移動平均線のクロス）。
  - ファンダメンタルズ分析に基づくバリュエーションモデル。
  - 経済理論に基づく仮説。
  - 低精度だが感度の高い機械学習モデル。
- **出力:** サイド（Side）の決定。$y\_{primary} \\in \\{-1, 1\\}$（売り または 買い）。

#### **第 2 段階：Secondary Model / Meta-Model（サイズの決定・フィルタリング）**

- **目的:** 高いプレシジョン（High Precision）を実現すること。すなわち、Primary Model が出したシグナルのうち、ダマシ（False Positive）を排除し、本当に利益になるものだけを選別すること。
- **入力:** 元の市場特徴量 $X$ に加えて、Primary Model の出力 $y\_{primary}$（およびその確信度）を入力として受け取る。
- **ターゲット:** Primary Model のシグナルに従った場合の成否。
  - Primary Model が「買い」と判断し、実際に利益が出た $\\rightarrow$ ラベル 1
  - Primary Model が「買い」と判断し、損失が出た $\\rightarrow$ ラベル 0
  - Primary Model がシグナルを出さなかった $\\rightarrow$ 学習対象外
- **出力:** トレードを実行すべき確率（Confidence）。$p\_{meta} \\in $。

### **4.2 クオンタメンタル（Quantamental）戦略への応用**

メタラベリングの最大の強みは、「ホワイトボックス（理論）」と「ブラックボックス（ML）」の融合にある。  
多くのファンドでは、完全にブラックボックスな End-to-End の ML モデルよりも、経済的直観に基づくベース戦略（Primary）の上に、ML によるフィルタリング（Secondary）を重ねるアプローチ（Quantamental）が好まれる 16。

- **例:**
  - **Primary:** 「PER が歴史的低水準にある銘柄をロングする」というバリュー投資のルール。
  - **Secondary (Meta):** 「そのバリュー投資が成功するのは、ボラティリティが低く、かつ流動性が高い時だけである」というような複雑な条件（レジーム）を ML（Random Forest や GBDT）に学習させる。

### **4.3 数理的利点：オーバーフィッティングの抑制**

単一のモデルで「方向（Side）」と「確信度（Size）」を同時に学習させようとすると、モデルは複雑になりすぎ、オーバーフィッティングのリスクが増大する。メタラベリングによってタスクを分解することで、各モデルの複雑性を抑えることができる。  
また、Secondary Model は二値分類（儲かるか否か）に特化するため、F1 スコアの最大化に直接的に寄与する。特に、Primary Model が多くの偽陽性（False Positives）を出す傾向がある場合、Meta-Model によるフィルタリング効果は劇的である 7。

### **4.4 ポジションサイジング（Bet Sizing）への接続**

Meta-Model の出力確率は、ケリー基準（Kelly Criterion）などの資金管理モデルに直接入力することができる。  
確信度 $p$ が高い場合はポジションサイズを大きくし、低い場合（0.5 に近い場合など）はポジションを持たないか、縮小する。これにより、単なる売買シグナル生成機ではなく、リスク管理を含めたポートフォリオ構築システムとしての機能を果たすことになる 17。

## ---

**5\. ターゲットエンジニアリング：Kaggle/Quant の最先端手法**

金融コンペティション（Kaggle 等）のトップソリューションや、先進的なクオンツファンドでは、ターゲット変数そのものに高度な統計的処理を施す「ターゲットエンジニアリング」が標準化している。これは、市場全体の動き（ベータ）を除去し、純粋なアルファ（超過収益）のみを抽出するための技術である。

### **5.1 残差化リターン（Residualized Returns）**

G-Research Crypto Forecasting コンペティション等で採用されたこの手法は、資産のリターンから市場共通の要因（ファクター）を回帰分析により除去し、その「残差」を予測対象とする 19。

#### **数理的定義**

資産 $i$ のリターン $R\_{i,t}$ に対し、市場インデックス（またはセクターインデックス）のリターン $R\_{m,t}$ を用いて以下の回帰を行う。

$$R\_{i,t} \= \\beta\_i R\_{m,t} \+ \\alpha\_{i,t} \+ \\epsilon\_{i,t}$$  
ここで、予測対象とするターゲット $y\_{i,t}$ は、回帰の残差部分である。

$$y\_{i,t} \= R\_{i,t} \- \\beta\_i R\_{m,t}$$

#### **目的と効果**

- **ベータの除去:** 仮想通貨や株式市場では、全銘柄がビットコインや S\&P500 に連動して動く相関（Correlation）が極めて高い。生のリターンを学習させると、モデルは支配的な要因である「市場全体の動き」ばかりを学習してしまい、個別銘柄の固有の動き（Idiosyncratic movement）を捉えられなくなる。残差化により、市場全体が暴落している中でも「相対的に強い」銘柄を識別できるようになる。
- **マーケットニュートラル戦略:** このターゲットを用いて学習したモデルは、ロングポートフォリオとショートポートフォリオを構築した際に、市場全体のリスク（マーケットベータ）が自然と相殺され、純粋なアルファのみを収益源とする戦略（マーケットニュートラル）に適している。

### **5.2 ターゲットの中立化（Target Neutralization / Orthogonalization）**

Ubiquant Market Prediction や Numerai で頻出する、より汎用的な手法である。これは、ターゲット変数を特定の特徴量（リスクファクターなど）に対して直交化（無相関化）する処理である 21。

#### **線形直交化プロセス**

ターゲットベクトル $\\mathbf{y}$ と、中立化したい特徴量行列 $\\mathbf{F}$（例：モメンタム、ボラティリティ、セクターダミーなどの既知のリスクファクター）があるとする。ターゲットを $\\mathbf{F}$ に射影し、その成分を除去する。

$$\\mathbf{y}\_{neutral} \= \\mathbf{y} \- \\mathbf{F}(\\mathbf{F}^T \\mathbf{F})^{-1} \\mathbf{F}^T \\mathbf{y}$$  
Python コードでの実装イメージ（Ridge 回帰を用いる場合など）:

Python

model \= Ridge(alpha=1.0)  
model.fit(features_to_neutralize, target)  
pred \= model.predict(features_to_neutralize)  
neutralized_target \= target \- proportion \* pred

ここで proportion（係数 $\\gamma$）は、どの程度中立化するかを制御するパラメータである（完全中立化なら 1.0）。

#### **戦略的意義：Feature Neutralization**

ターゲット側ではなく、モデルの予測値（Prediction）に対してこの処理を行う場合もある（Feature Neutralization）。  
この処理の目的は、「誰でも知っている単純なファクター（例：単なるモメンタム）」への依存を排除し、モデルに未発見の非線形なパターンや深い相互作用（Interaction）を学習させることにある。また、特定のリスクファクターへのエクスポージャーを強制的にゼロにすることで、予期せぬファクターの急変（ローテーション）によるドローダウンを防ぐリスク管理的な意味合いも強い。

### **5.3 補助ターゲット（Auxiliary Targets）とマルチタスク学習**

Jane Street Market Prediction などの大規模データセットを用いたコンペティションでは、本来予測したいターゲット（例：将来の収益効用）以外に、複数の関連するターゲット（Auxiliary Targets/Responders）を同時に学習させるマルチタスク学習（Multi-Task Learning）が有効であることが示されている 24。

#### **補助ターゲットの具体例**

1. **マルチホライズン予測:** 本命が「1 日後のリターン」だとしても、「5 分後」「1 時間後」「5 日後」のリターンを同時に予測させる。これにより、モデルは短期的なノイズと長期的なトレンドの整合性を学習できる。
2. **ボラティリティ予測:** リターンと同時に、将来のボラティリティを予測させる。
3. **ノイズ付加ターゲット:** ターゲットにガウシアンノイズを加えたものを別のターゲットとして学習させ、正則化効果（Regularization）を狙う。

#### **ニューラルネットワークでの実装**

共有層（Shared Layers）で市場の抽象的な表現（Latent Representation）を学習させ、出力層付近で各タスク（メインターゲット、補助ターゲット群）に分岐させる（Head を分ける）。これにより、データ不足やノイズによる過学習を防ぎ、汎化性能の高いロバストな特徴抽出が可能となる。

### **5.4 幾何ブラウン運動（GBM）パラメータの推定**

Jane Street の上位解法の一つとして、価格が幾何ブラウン運動（GBM）に従う確率過程であると仮定し、リターンそのものではなく、その確率微分方程式（SDE）のパラメータを推定させるというアプローチがある 27。

$$dS\_t \= \\mu S\_t dt \+ \\sigma S\_t dW\_t$$  
ここで、ニューラルネットワークに予測させるのは、ドリフト項 $\\mu$（トレンド成分）と拡散項 $\\sigma$（ボラティリティ成分）である。  
このアプローチは、単に価格が上がるか下がるかという離散的な予測ではなく、将来の価格分布の形状（期待値と分散）を推定することに等しい。これにより、期待リターンが正であってもボラティリティが高すぎる場合はトレードを見送る、といった高度な意思決定を数理的に裏付けられた形で行うことができる。

## ---

**6\. レジーム検出に基づく動的ラベリング（Regime-Aware Labeling）**

金融市場は常に同じ物理法則で動いているわけではない。強気相場（Bull）、弱気相場（Bear）、横ばい（Sideways）、パニック（Crisis）といった異なる「レジーム（Regime）」が存在し、局面によって有効な戦略は全く異なる。レジームを検出し、それに応じてラベリングやモデル自体を切り替える手法は、適応型戦略の中核をなす。

### **6.1 隠れマルコフモデル（HMM）による教師なし状態推定**

HMM は、観測されるデータ（リターン、ボラティリティ、相関など）の背後に、直接は観測できない離散的な「隠れた状態（Hidden States）」が存在すると仮定する確率モデルである 28。

#### **実装とラベリング**

Gaussian Mixture HMM などを用いて、過去のデータから教師なし学習で状態を推定する。通常、2〜4 つの状態が識別される。

- **State 0:** 低ボラティリティ・安定上昇（Trend Following が有効）
- **State 1:** 高ボラティリティ・急落（Mean Reversion や Short 戦略が有効）
- **State 2:** 方向感のない乱高下（Cash Position 推奨）

**活用法:**

1. **条件付きラベリング:** 現在が「State 0」である確率が高い期間のデータのみを用いてトレンドフォローモデルを学習させる。
2. **メタ特徴量:** HMM が算出する「各状態への所属確率（State Probability）」を特徴量ベクトルの一部として ML モデルに入力する。これにより、モデルは「現在は高ボラティリティ局面だから、テクニカル指標の解釈を変える」といった非線形な判断が可能になる。

### **6.2 Wasserstein K-Means による分布形状クラスタリング**

従来の K-Means（ユークリッド距離に基づく）は、平均や分散の違いは捉えられるが、相関構造やテールリスク（分布の歪み）の変化を捉えるのが苦手である。これに対し、最適輸送理論（Optimal Transport）に基づく「Wasserstein 距離（Earth Mover's Distance）」を用いたクラスタリングが、より高度なレジーム検出手法として注目されている 31。

この手法では、リターンの分布（ヒストグラム）そのものを一つのデータポイントとして扱い、分布間の「形状の類似度」に基づいてクラスタリングを行う。これにより、平均リターンが変わらなくても、テールリスクが拡大している（クラッシュの前兆など）微妙なレジーム変化を検出できる可能性がある。これは、テイルヘッジ戦略や危機回避のためのラベリングとして極めて有効である。

## ---

**7\. 実現ボラティリティ（Realized Volatility）予測**

Optiver Realized Volatility Prediction コンペティションのように、ターゲットを「方向（リターン）」ではなく「変動の大きさ（ボラティリティ）」に設定する場合がある 33。

### **7.1 実現ボラティリティ（RV）の定義**

一般に、高頻度データ（ティックや 1 分足）のリターンの対数収益率 $r\_t$ を用いて、特定期間の RV は以下のように計算される。

$$RV \= \\sqrt{\\sum\_{t} r\_t^2}$$

### **7.2 戦略的優位性**

方向（リターン）の予測は、効率的市場仮説の影響を強く受け、予測精度（R-squared）は極めて低い（ほぼゼロに近い）。一方、ボラティリティは「ボラティリティ・クラスタリング」や「平均回帰性」といった強い統計的性質を持ち、はるかに高い精度で予測が可能である。

**ラベリングとしての利用:**

- **ボラティリティ・トレーディング:** オプションのストラドル/ストラングル戦略や、VIX 先物などのボラティリティ商品をトレードするための直接的なターゲットとする。
- **メタラベリングの入力:** 方向予測モデルの「信頼区間」を推定するために、RV 予測モデルを併用する。RV 予測が高い場合は、方向予測モデルのシグナルに対する閾値を引き上げる（動的閾値）などの制御を行う。

## ---

**8\. 結論：統合的アプローチの推奨**

本報告書で検討したラベリング手法は、それぞれ異なる市場の課題に対処するために設計されている。現代の高度な ML トレーディングシステムにおいては、これらを単独で用いるのではなく、以下のように統合（Ensemble）して用いることが推奨される。

1. **ベースラインの確保:** まずは**動的閾値を用いた固定期間法**あるいは TBM でベースラインモデルを構築する。
2. **シグナルの純化:** \*\*残差化（Residualization）**や**中立化（Neutralization）\*\*を適用し、ベータ依存を排除した真のアルファ学習を目指す。
3. **構造の理解:** **ZigZag**や**DC イベント**を用いたモデルを補助的に導入し、長期的なトレンド構造を把握させる。
4. **精度の向上:** **メタラベリング**フレームワークを採用し、エントリー（Primary）とフィルタリング（Secondary）を分離して F1 スコアを最適化する。
5. **環境適応:** **HMM**や**Wasserstein クラスタリング**によるレジーム判定をメタ情報として組み込み、市場環境の変化に応じてモデルの挙動を動的に調整する。

これらの手法を組み合わせることで、ノイズの多い金融データの中から有意なシグナルを抽出し、過学習を抑制しつつ、実運用においても堅牢なパフォーマンスを発揮するモデル構築が可能となるだろう。

---

参考文献およびデータソースの識別子:  
本レポートの記述は、以下のリサーチスニペットに基づいている。

1

#### **引用文献**

1. A Dynamic Labeling Approach for Financial Assets Forecasting \- Neuravest, 12 月 23, 2025 にアクセス、 [https://www.neuravest.net/a-dynamic-labeling-approach-for-financial-assets-forecasting-2/](https://www.neuravest.net/a-dynamic-labeling-approach-for-financial-assets-forecasting-2/)
2. Labeling Financial Data \- RiskLab AI, 12 月 23, 2025 にアクセス、 [https://www.risklab.ai/research/financial-data-science/labeling](https://www.risklab.ai/research/financial-data-science/labeling)
3. Financial Machine Learning Part 1: Labels | by Maks Ivanov | TDS Archive | Medium, 12 月 23, 2025 にアクセス、 [https://medium.com/data-science/financial-machine-learning-part-1-labels-7eeed050f32e](https://medium.com/data-science/financial-machine-learning-part-1-labels-7eeed050f32e)
4. Algorithmic trading with directional changes \- Essex Research Repository, 12 月 23, 2025 にアクセス、 [https://repository.essex.ac.uk/33750/1/s10462-022-10307-0.pdf](https://repository.essex.ac.uk/33750/1/s10462-022-10307-0.pdf)
5. (PDF) A Directional-Change Event Approach for Studying Financial Time Series, 12 月 23, 2025 にアクセス、 [https://www.researchgate.net/publication/228316274_A_Directional-Change_Event_Approach_for_Studying_Financial_Time_Series](https://www.researchgate.net/publication/228316274_A_Directional-Change_Event_Approach_for_Studying_Financial_Time_Series)
6. The Triple Barrier Method: Labeling Financial Time Series for ML in Elixir | by Yair Oz, 12 月 23, 2025 にアクセス、 [https://medium.com/@yairoz/the-triple-barrier-method-labeling-financial-time-series-for-ml-in-elixir-e539301b90d6](https://medium.com/@yairoz/the-triple-barrier-method-labeling-financial-time-series-for-ml-in-elixir-e539301b90d6)
7. Data Labelling \- Mlfin.py, 12 月 23, 2025 にアクセス、 [https://mlfinpy.readthedocs.io/en/latest/Labelling.html](https://mlfinpy.readthedocs.io/en/latest/Labelling.html)
8. Volatility-Adjusted Means \- QuantConnect Quant League Open, 12 月 23, 2025 にアクセス、 [https://www.quantconnect.com/league/18110/2024-q4/volatility-adjusted-means/](https://www.quantconnect.com/league/18110/2024-q4/volatility-adjusted-means/)
9. Volatility And Measures Of Risk-Adjusted Return With Python \- QuantInsti Blog, 12 月 23, 2025 にアクセス、 [https://blog.quantinsti.com/volatility-and-measures-of-risk-adjusted-return-based-on-volatility/](https://blog.quantinsti.com/volatility-and-measures-of-risk-adjusted-return-based-on-volatility/)
10. The Quality of Returns: Crypto Risk-Adjusted Performance \- XBTO, 12 月 23, 2025 にアクセス、 [https://www.xbto.com/resources/the-quality-of-returns-crypto-risk-adjusted-performance](https://www.xbto.com/resources/the-quality-of-returns-crypto-risk-adjusted-performance)
11. Directional-change intrinsic time \- Wikipedia, 12 月 23, 2025 にアクセス、 [https://en.wikipedia.org/wiki/Directional-change_intrinsic_time](https://en.wikipedia.org/wiki/Directional-change_intrinsic_time)
12. Master the Zig Zag Indicator: Definition, Usage, and Formula for Trend Analysis, 12 月 23, 2025 にアクセス、 [https://www.investopedia.com/terms/z/zig_zag_indicator.asp](https://www.investopedia.com/terms/z/zig_zag_indicator.asp)
13. Zig Zag Indicator: Filtering Noise to Highlight Significant Price Swings \- LuxAlgo, 12 月 23, 2025 にアクセス、 [https://www.luxalgo.com/blog/zig-zag-indicator-filtering-noise-to-highlight-significant-price-swings/](https://www.luxalgo.com/blog/zig-zag-indicator-filtering-noise-to-highlight-significant-price-swings/)
14. Zig Zag \- Indicators \- QuantConnect.com, 12 月 23, 2025 にアクセス、 [https://www.quantconnect.com/docs/v2/writing-algorithms/indicators/supported-indicators/zig-zag](https://www.quantconnect.com/docs/v2/writing-algorithms/indicators/supported-indicators/zig-zag)
15. Event-Driven LSTM For Forex Price Prediction \- arXiv, 12 月 23, 2025 にアクセス、 [https://arxiv.org/pdf/2102.01499/1000](https://arxiv.org/pdf/2102.01499/1000)
16. Why Meta-Labeling Is Not a Silver Bullet by Francesco Baldisserri \- QuantConnect.com, 12 月 23, 2025 にアクセス、 [https://www.quantconnect.com/forum/discussion/14706/why-meta-labeling-is-not-a-silver-bullet/](https://www.quantconnect.com/forum/discussion/14706/why-meta-labeling-is-not-a-silver-bullet/)
17. Meta-Labeling \- Wikipedia, 12 月 23, 2025 にアクセス、 [https://en.wikipedia.org/wiki/Meta-Labeling](https://en.wikipedia.org/wiki/Meta-Labeling)
18. Meta Labeling (A Toy Example) \- Hudson & Thames, 12 月 23, 2025 にアクセス、 [https://hudsonthames.org/meta-labeling-a-toy-example/](https://hudsonthames.org/meta-labeling-a-toy-example/)
19. G-Research Crypto Forecasting \- Kaggle, 12 月 23, 2025 にアクセス、 [https://www.kaggle.com/competitions/g-research-crypto-forecasting](https://www.kaggle.com/competitions/g-research-crypto-forecasting)
20. G-Research Crypto Forecasting \- Kaggle, 12 月 23, 2025 にアクセス、 [https://www.kaggle.com/competitions/g-research-crypto-forecasting/discussion/286778](https://www.kaggle.com/competitions/g-research-crypto-forecasting/discussion/286778)
21. \[CVXPY\] Feature Neutralization \- Kaggle, 12 月 23, 2025 にアクセス、 [https://www.kaggle.com/code/marketneutral/cvxpy-feature-neutralization/input](https://www.kaggle.com/code/marketneutral/cvxpy-feature-neutralization/input)
22. \[CVXPY\] Feature Neutralization \- Kaggle, 12 月 23, 2025 にアクセス、 [https://www.kaggle.com/code/marketneutral/cvxpy-feature-neutralization](https://www.kaggle.com/code/marketneutral/cvxpy-feature-neutralization)
23. Feature Neutralization \- Google Colab, 12 月 23, 2025 にアクセス、 [https://colab.research.google.com/github/numerai/example-scripts/blob/master/feature_neutralization.ipynb](https://colab.research.google.com/github/numerai/example-scripts/blob/master/feature_neutralization.ipynb)
24. \[Private LB 8th\] solution \- Kaggle, 12 月 23, 2025 にアクセス、 [https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/writeups/evgeniia-grigoreva-private-lb-8th-solution](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/writeups/evgeniia-grigoreva-private-lb-8th-solution)
25. 39th Place \- Solution Overview & Code | Kaggle, 12 月 23, 2025 にアクセス、 [https://www.kaggle.com/competitions/jane-street-market-prediction/writeups/dmitry-yudin-39th-place-solution-overview-code](https://www.kaggle.com/competitions/jane-street-market-prediction/writeups/dmitry-yudin-39th-place-solution-overview-code)
26. Jane Street Real-Time Market Data Forecasting | Kaggle, 12 月 23, 2025 にアクセス、 [https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/discussion/556548](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/discussion/556548)
27. 10th Place Solution: Geometric Brownian Motion and Mixture Density Networks \- Kaggle, 12 月 23, 2025 にアクセス、 [https://www.kaggle.com/competitions/jane-street-market-prediction/writeups/float-10th-place-solution-geometric-brownian-motio](https://www.kaggle.com/competitions/jane-street-market-prediction/writeups/float-10th-place-solution-geometric-brownian-motio)
28. Market Regime Detection using Hidden Markov Models in QSTrader | QuantStart, 12 月 23, 2025 にアクセス、 [https://www.quantstart.com/articles/market-regime-detection-using-hidden-markov-models-in-qstrader/](https://www.quantstart.com/articles/market-regime-detection-using-hidden-markov-models-in-qstrader/)
29. Market Regime using Hidden Markov Model \- My Framer Site \- Numatix, 12 月 23, 2025 にアクセス、 [https://numatix.in/blogs/market-regimes-via-hidden-markov-models](https://numatix.in/blogs/market-regimes-via-hidden-markov-models)
30. Market Regime Detection using Hidden Markov Models \- PyQuantLab, 12 月 23, 2025 にアクセス、 [https://www.pyquantlab.com/articles/Market%20Regime%20Detection%20using%20Hidden%20Markov%20Models.html](https://www.pyquantlab.com/articles/Market%20Regime%20Detection%20using%20Hidden%20Markov%20Models.html)
31. Market Regime Detection: From Hidden Markov Models to Wasserstein Clustering | by Arshad Ansari | Hikmah Techstack \- Medium, 12 月 23, 2025 にアクセス、 [https://medium.com/hikmah-techstack/market-regime-detection-from-hidden-markov-models-to-wasserstein-clustering-6ba0a09559dc](https://medium.com/hikmah-techstack/market-regime-detection-from-hidden-markov-models-to-wasserstein-clustering-6ba0a09559dc)
32. The Next Generation of Trading: From Hidden Markov Models to Wasserstein Clustering (The Regime Detection Revolution) | by Nayab Bhutta | Nov, 2025 | InsiderFinance Wire, 12 月 23, 2025 にアクセス、 [https://wire.insiderfinance.io/the-next-generation-of-trading-from-hidden-markov-models-to-wasserstein-clustering-the-regime-3697d79e99d7](https://wire.insiderfinance.io/the-next-generation-of-trading-from-hidden-markov-models-to-wasserstein-clustering-the-regime-3697d79e99d7)
33. Optiver Realized Volatility Prediction | Kaggle, 12 月 23, 2025 にアクセス、 [https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/data](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/data)
34. Optiver Realized Volatility Introduction \- Kaggle, 12 月 23, 2025 にアクセス、 [https://www.kaggle.com/code/lucasmorin/optiver-realized-volatility-introduction](https://www.kaggle.com/code/lucasmorin/optiver-realized-volatility-introduction)
35. Meta Labeling for Algorithmic Trading: How to Amplify a Real Edge : r/algotrading \- Reddit, 12 月 23, 2025 にアクセス、 [https://www.reddit.com/r/algotrading/comments/1lnm48w/meta_labeling_for_algorithmic_trading_how_to/](https://www.reddit.com/r/algotrading/comments/1lnm48w/meta_labeling_for_algorithmic_trading_how_to/)
36. Adaptive Event-Driven Labeling: Multi-Scale Causal Framework with Meta-Learning for Financial Time Series \- MDPI, 12 月 23, 2025 にアクセス、 [https://www.mdpi.com/2076-3417/15/24/13204](https://www.mdpi.com/2076-3417/15/24/13204)
37. Economic regimes identification using machine learning technics, 12 月 23, 2025 にアクセス、 [https://dspace.unia.es/bitstreams/2be1ab7d-4259-404d-86df-fbea73e2e316/download](https://dspace.unia.es/bitstreams/2be1ab7d-4259-404d-86df-fbea73e2e316/download)
38. A forest of opinions: A multi-model ensemble-HMM voting framework for market regime shift detection and trading \- AIMS Press, 12 月 23, 2025 にアクセス、 [https://www.aimspress.com/article/id/69045d2fba35de34708adb5d](https://www.aimspress.com/article/id/69045d2fba35de34708adb5d)
