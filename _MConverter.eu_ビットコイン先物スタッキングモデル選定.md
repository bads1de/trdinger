ビットコイン無期限先物市場における定量的取

引戦略：機械学習モデルのスタッキングとアー

キテクチャ選定に関する包括的研究\*\*\*\*

**1.**序論：暗号資産市場におけるアルゴリズム取引の進化と現

在地\*\*\*\*

**1.1 **市場の成熟と**α**（アルファ）の枯渇\*\*\*\*

かつて「デジタル・ワイルドウエスト」と形容された暗号資産市場は、機関投資家の参入、デリバティ

ブ市場の爆発的な拡大、そして高頻度取引（HFT）業者の技術的洗練により、劇的な変貌を遂げて

いる。特にビットコイン（BTC）の無期限先物（Perpetual Futures）市場は、現物価格との乖離を調整

する資金調達率（Funding Rate）という独自のメカニズムや、24 時間 365 日停止しない連続性、そし

て最大 100 倍を超えるレバレッジが可能にする非線形な価格変動特性を有しており、世界で最も流

動性が高く、かつ複雑な金融商品の一つとなっている。

このような環境下において、単純なテクニカル指標のクロスオーバーや、単一の回帰モデルに依存

した戦略が持続的な超過収益（アルファ）を生み出すことは極めて困難になっている。効率的市場仮

説（EMH）が示唆するように、既知の情報は瞬時に価格に織り込まれる傾向が強まっており、市場参

加者が容易にアクセスできる情報や単純な線形関係性から得られるエッジは、競争によって急速に

消失しているからである。しかし、適応的市場仮説（AMH）の視点に立てば、市場の効率性は一定で

はなく、環境の変化に伴って変動する。この市場の「非効率性のポケット」を見つけ出し、利益に変え

るためには、従来の統計的手法を超えた、より高度で適応的なアプローチが不可欠となっている。

**1.2**機械学習とアンサンブル戦略の台頭\*\*\*\*

この課題に対する現代的な解として浮上しているのが、機械学習（ML）、特に複数のモデルを組み

合わせる「スタッキング（Stacking）」技術である。Kaggle などのデータサイエンスコンペティション、特

に「G-Research Crypto Forecasting」や「Ubiquant Market Prediction」、「Optiver Realized Volatility Prediction」といった金融時系列予測のコンテストにおいて、上位に入賞するソリューション

（Winning Solutions）のほぼ全てが、単一のモデルではなく、数百から数千のモデルを巧みに組み

合わせたアンサンブル戦略を採用していることは偶然ではない 1。

スタッキングが金融市場において強力である理論的根拠は、モデルの多様性（Diversity）によるバイ

アス・バリアンスのトレードオフの最適化にある。勾配ブースティング決定木（GBDT）は構造化データ

の非線形な閾値処理に優れ、リカレントニューラルネットワーク（RNN）や Transformer は時系列の文

脈理解に長け、そして 1 次元畳み込みニューラルネットワーク（1D-CNN）は局所的な波形パターンの

抽出に適している。これら性質の異なる「弱い学習器（Weak Learners）」あるいは「専門特化した学

習器」を統合することで、個々のモデルが持つ弱点（特定の市場局面での過学習や適応不全）を補

完し、未知の市場環境に対してもロバストな予測能力を獲得することが可能となる。

本報告書は、ビットコイン無期限先物取引という特定のドメインに焦点を当て、最新の学術研究、コ

ンペティションの解法、そして実務家の知見を網羅的に分析し、現在考えうる最も有力なスタッキング

アーキテクチャとモデル選定の指針を提示することを目的とする。

**2.**ビットコイン無期限先物市場の微細構造と予測困難性\*\*\*\*

**2.1**無期限先物特有のメカニズム：**Funding Rate**

ビットコイン無期限先物をモデル化する上で、従来の先物取引と決定的に異なるのが「満期（Expiry

）」の欠如と、それに代わる価格調整メカニズムとしての「資金調達率（Funding Rate）」の存在であ

る。通常の先物であれば、満期日に先物価格と現物価格は強制的に収束するが、無期限先物では

この収束メカニズムが存在しない。その代わり、先物価格が現物価格（インデックス価格）よりも高く

乖離している場合（プレミアム状態）、ロングポジション保有者はショートポジション保有者に対して金

利（Funding Fee）を支払う義務が生じる。逆に安く乖離している場合（ディスカウント状態）は、ショー

トがロングに支払う 4。

このメカニズムは、機械学習モデルにとって二つの重要な意味を持つ。

第一に、Funding Rate そのものが市場のセンチメントとポジションの偏りを表す強力な特徴量となる

ことである。Funding Rate が極端に高い正の値を示している場合、市場は過度に強気であり、ロング

ポジションが積み上がっていることを示唆する。これは、逆張り（Mean Reversion）戦略のシグナルと

なり得るし、あるいは強いトレンドの継続を示唆する場合もある。この解釈は市場のレジーム（ボラ

ティリティの大小やトレンドの強弱）によって異なるため、非線形なモデルによる学習が有効である。

第二に、Funding Rate によるコスト（または収益）自体が、トレーダーの行動を変容させ、価格形成に

フィードバックループをもたらす点である。アービトラージャー（裁定取引者）は、Funding Rate を受け

取る方向（Carry Trade）にポジションを取ることで、価格乖離を縮小させる圧力をかける。この需給

のダイナミクスをモデルに組み込むことは、短期および中期の価格予測において不可欠な要素とな

る 8。

**2.2**清算連鎖（**Liquidation Cascades**）とテールの肥大化\*\*\*\*

暗号資産市場、特に無期限先物市場の特徴として、高いレバレッジ比率が挙げられる。多くの取引

所では最大 100 倍程度のレバレッジが提供されており、わずか 1%の価格変動で証拠金が全損し、強

制決済（Liquidation）が執行される可能性がある。

大量のロングポジションが一度に強制決済されると、成行売り注文が市場に殺到し、価格をさらに下

落させる。これが次のロングポジションの清算を誘発し、連鎖的な価格崩壊（カスケード）を引き起こ

す。この現象は価格変動の分布を正規分布から大きく逸脱させ、テール（裾野）が極端に厚い分布（

Fat Tail）を生み出す。

従来の線形モデルや、正規分布を前提とした統計モデル（ARIMA など）は、このような突発的かつ非

連続的な価格変動を「異常値」として処理してしまいがちである。しかし、ML トレーディングにおいて

は、この清算連鎖こそが最大の収益機会（またはリスク）であり、これを捉えるために LSTM や

Transformer、そして最新の Mamba といった、長期記憶と急激な変化への適応能力を持つモデルが

スタッキングの構成要素として重要視される 9。

**3.**スタッキング戦略の理論的枠組み\*\*\*\*

**3.1**アンサンブル学習の階層構造\*\*\*\*

効果的なスタッキングシステムを構築するためには、無秩序にモデルを混ぜ合わせるのではなく、明

確な役割分担を持った階層構造（Layered Architecture）を設計する必要がある。一般的には以下

の 3 層構造が採用されることが多い。

● **Level-0**（入力層）: 生の市場データおよび特徴量エンジニアリングによって生成された派生

データ。

● **Level-1**（ベースモデル層）: Level-0 のデータを入力とし、それぞれ異なるアルゴリズム、異なる

ハイパーパラメータ、あるいは異なる特徴量サブセットを用いて学習された多様なモデル群。こ

こでの目的は、高い予測精度を持つこと以上に、\*\*「予測誤差の相関が低い」\*\*モデルを揃える

ことである。

● **Level-2**（メタモデル層）: Level-1 の各モデルが出力した予測値（および必要に応じて元の特徴

量）を入力とし、最終的な統合予測を行うモデル。メタモデルは、各ベースモデルの信頼度を市

場環境に応じて動的に重み付けする役割を担う。

**3.2**多様性（**Diversity**）の確保\*\*\*\*

スタッキングが機能するための絶対条件は、ベースモデル間の多様性である。全てのモデルが似た

ような間違いをするならば、それらを平均化しても精度は向上しない。Kaggle の「Ubiquant Market Prediction」などの事例 2 からも分かるように、決定木ベースのモデル（GBDT）とニューラルネットワー

ク（NN）は、特徴量の捉え方が根本的に異なるため、これらを組み合わせることで最大のシナジー効

果が得られる。

● **GBDT**の視点: 特徴量空間を直交する超平面で分割し、矩形の領域ごとに定数値を割り当て

る。データの大小関係や閾値（例：「RSI > 80」）に基づくルールを学習するのに極めて優れてい

るが、線形トレンドの外挿は苦手とする。

● **NN**の視点: 特徴量の線形結合と非線形活性化関数を通じて、滑らかな関数近似を行う。データ

の連続性や、より複雑な位相構造を捉えることができるが、スケーリングに敏感で、外れ値の影

響を受けやすい。

次章以降では、ビットコイン無期限先物において特に有効性が確認されている Level-1 モデルの詳細

な選定とアーキテクチャについて論じる。

**4. Level-1**ベースモデルの選定：決定木ブースティング\*\*\*\*

**\(Tree-based Models\)**

金融時系列データは、画像や音声データと比較して S/N 比（信号対雑音比）が極めて低いという特徴

がある。このような「表形式（Tabular）」かつノイズの多いデータに対して、勾配ブースティング決定木

（Gradient Boosting Decision Trees: GBDT）は依然として最強のベースラインであり、スタッキング

の中核を成す存在である。

**4.1 LightGBM**：高速性と精度のバランス\*\*\*\*

G-Research のコンペティションにおいて、2 位の解法は複雑なアンサンブルを行わず、高度にチュー

ニングされた LightGBM 単体で構成されていた 11。この事実は、特徴量エンジニアリングさえ適切であ

れば、LightGBM が深層学習モデルに匹敵、あるいは凌駕するポテンシャルを持つことを示唆してい

る。

● アーキテクチャの特徴: LightGBM は「Leaf-wise（葉単位）」の木成長アルゴリズムを採用してい

る。これは、損失関数の減少に最も寄与する葉を選択して分割を行う手法であり、従来の「

Level-wise（深さ単位）」の手法と比較して、収束が速く、より複雑なパターンを学習できる。

● **BTC**トレードへの適用: ビットコイン市場の非定常性に対応するためには、過学習の抑制が鍵と

なる。reg_alpha（L1 正則化）や reg_lambda（L2 正則化）を強めに設定し、min_data_in_leaf（葉

に含まれる最小データ数）を大きくすることで、ノイズへの過敏な反応を防ぐ。また、カテゴリカル

変数（例：時間帯、曜日）の扱いに長けており、特定の時間帯（米国市場オープン時など）におけ

る特異なボラティリティパターンを捉えるのに有効である。

**4.2 CatBoost**：時系列データへの適性\*\*\*\*

CatBoost は、その名の通りカテゴリカル変数の扱いに優れているだけでなく、\*\*「Ordered Boosting

」\*\*という独自の手法を採用しており、これが時系列予測において大きな強みとなる 12。

● リーク（**Data Leakage**）の防止: 通常のブースティングでは、ターゲット統計量（Target Encoding）を計算する際に、自分自身のデータを含めてしまうことによるリークが発生しやす

い。CatBoost の Ordered Boosting は、データの順序を考慮し、過去のデータのみから統計量

を計算する仕組みが組み込まれているため、時系列データの学習においてよりロバストなモデ

ル構築が可能となる。

● 対称木（**Symmetric Trees**）: CatBoost が生成する決定木は対称構造を持つ。これにより推論

時の計算が非常に高速化されるため、ミリ秒単位のレイテンシが要求される HFT（高頻度取引）

に近い領域での実運用において有利に働く。

**4.3 XGBoost**：安定性と実績\*\*\*\*

XGBoost は、Kaggle の歴史の中で最も多くの勝利をもたらしたアルゴリズムの一つであり、その信頼

性は揺るぎない。特に、GPU による高速化が容易であるため、大規模なハイパーパラメータ探索（

Optuna などを用いたチューニング）を行う際の効率が良い。また、2 次の勾配（Hessian）まで考慮し

た損失関数の近似を行っており、最適化の精度が高い。スタッキングにおいては、LightGBM や

CatBoost とは異なる乱数シードや異なる特徴量サブセットで学習させた XGBoost を混ぜることで、ア

ンサンブルの多様性を担保する役割を担うことが多い。

**5. Level-1**ベースモデルの選定：系列モデリングと深層学習\*\*\*\*

**\(Sequence Models & Deep Learning\)**

GBDT は「点」としてのデータ（ある瞬間の特徴量ベクトル）からの推論には強いが、時系列データの

「文脈」や「順序」が持つ情報を直接的に学習することは構造上不可能である（ラグ特徴量として明示

的に与える必要がある）。これに対し、ディープラーニングモデルは、過去の価格変動のシーケンス

そのものを入力とし、そこに潜む時間的な依存関係を学習することができる。

**5.1 LSTM / GRU**：リカレントニューラルネットワークの再評価\*\*\*\*

長短期記憶（LSTM）および Gated Recurrent Unit（GRU）は、時系列予測のスタンダードとして長く君

臨してきた。最近の研究 9 やコンペティションの上位解法 13 においても、これらは依然として重要な地

位を占めている。

● アーキテクチャの選択: ビットコイン価格予測においては、単純な LSTM よりも**GRU**が選好される

傾向がある。GRU はゲート構造がシンプルでパラメータ数が少ないため、データ量が限られて

いる場合やノイズが多い場合でも過学習しにくく、計算コストも低い。

● 双方向（**Bi-directional**）モデル: 過去から未来への流れだけでなく、未来から過去への流れ

（学習時のみ可能）も考慮する Bi-LSTM/Bi-GRU は、データの文脈理解を深めるために有効で

ある。ただし、実運用時の推論では未来のデータは使えないため、学習時のアーキテクチャ設

計には注意が必要である（多くの場合は過去方向のみの片方向モデル、あるいは学習時のみ

双方向の特徴抽出を行う構成がとられる）。

● **GBDT**との補完性: G-Research の 3 位解法 15 では、LightGBM と GRU ベースのモデルが併用さ

れている。GRU は、数分〜数時間前の価格変動パターン（「急騰後の調整」や「徐々にボラティ

リティが低下する収束」など）を内部状態として記憶し、GBDT が見逃す長期的なモメンタムを捉

えることができる。

**5.2 Transformer**と**Squeezeformer**：**Attention**機構の活用\*\*\*\*

自然言語処理（NLP）分野で革命を起こした Transformer は、金融時系列予測にも応用されている。

Self-Attention 機構により、遠く離れた過去のイベント（例：数時間前の重要発言や大口約定）と現在

の価格変動との関連性を直接的に学習できる点が最大の強みである。

● **Squeezeformer**の採用: G-Research コンペティションの優勝解法の一部として、

**Squeezeformer**の使用が報告されている 1。標準的な Transformer や Conformer は、毎時刻の

計算コストが高く、冗長な情報（価格がほとんど動かない時間帯など）も等しく処理してしまう。

Squeezeformer は、時間方向の冗長性を「Squeeze（圧縮）」して計算量を削減しつつ、

1D-CNN と Attention を組み合わせることで、高頻度データの局所的な特徴と大局的なトレンド

の両方を効率的に抽出する。これは、ノイズが多く、かつ重要なシグナルが散発的に現れる暗

号資産市場の特性に極めて適している。

**5.3 Mamba \(State Space Models\)**：次世代の時系列モデリング\*\*\*\*

2024 年以降、Transformer に代わる新たなアーキテクチャとして注目を集めているのが**Mamba**を含

む状態空間モデル（SSM）である。

● 線形計算量の利点: Transformer の Attention 計算量はシーケンス長 $L$ の二乗 $O\(L^2\)$ で

増加するため、長い過去データを入力することが困難であった。一方、Mamba は $O\(L\)$ の線

形計算量で済むため、例えば過去数日分の 1 分足データ（数千ステップ）を丸ごと入力として扱う

ことが現実的となる。

● **CryptoMamba**: ビットコイン価格予測に Mamba を適用した研究「CryptoMamba」10 によれば、

Mamba は従来の LSTM や Transformer と比較して、RMSE（二乗平均平方根誤差）や MAPE（平

均絶対パーセント誤差）において顕著に優れた成績を収めている。特に、Mamba の「選択的走

査（Selective Scan）」メカニズムは、無関係なノイズを無視し、重要な情報だけを長期記憶とし

て保持する能力に長けており、ボラティリティが急変する局面での適応力が高いと評価されてい

る。スタッキングの構成要素として、Mamba は「長期記憶」を担当するスペシャリストとして非常

に有望である。

**6. Level-1**ベースモデルの選定：特徴量抽出とノイズ除去\*\*\*\*

**\(Feature Extraction & Denoising\)**

生の市場データや単純なテクニカル指標をそのまま予測モデルに入力するのではなく、一度ニュー

ラルネットワークを通して「表現学習（Representation Learning）」を行うことで、データの S/N 比を向

上させるアプローチが近年主流となっている。

**6.1 Encoder \+ MLP \(Denoising Autoencoder Strategy\)**

G-Research コンペティションの 1 位解法や、Jane Street コンペティションで有名になったこのアーキテ

クチャは、金融 ML における「デファクトスタンダード」の一つとなりつつある 17。

● メカニズム:

1. **Denoising Autoencoder \(DAE\)**: 入力特徴量にガウシアンノイズやドロップアウトを加え

て破損させ、それを元の綺麗な特徴量に復元するようにニューラルネットワーク（

Encoder-Decoder）を学習させる。この過程で、ネットワークはデータに含まれる本質的な

構造（多様体）を学習し、ノイズを除去したロバストな特徴表現（潜在変数）を獲得する。

2. **MLP**による予測: DAE の Encoder 部分から得られる潜在変数と、元の特徴量を結合（

Concatenate）し、それを入力として多層パーセプトロン（MLP）でリターン予測を行う。

● **ResNet**の活用: MLP 部分は、深い層を持たせても学習が進むように、ResNet（残差結合）構造

を採用することが一般的である。

● 効果: 暗号資産市場のデータは極めてノイズが多いが、この Denoising プロセスを挟むことで、

モデルが偶発的なノイズに過学習することを防ぎ、汎化性能を劇的に向上させることができる。

**6.2 DeepLOB**：オーダーブックの画像化\*\*\*\*

高頻度データ（LOB: Limit Order Book）を利用する場合、板情報を「画像」として扱うアプローチが有

効である 18。

● **CNN-LSTM**アーキテクチャ:

○ 入力: 過去 $T$ 時点の板情報（価格レベルごとの注文量）を、$P \\times T$ の行列（画像）

として扱う。

○ **CNN**層: 1 次元または 2 次元の畳み込み層を用いて、板の形状（注文の壁、スプレッドの開

き、流動性の偏り）などの局所的な特徴を抽出する。

○ **LSTM**層: CNN で抽出された特徴ベクトルの時系列変化を LSTM で学習し、次の瞬間の価

格変動（または中値の方向）を予測する。

● マイクロストラクチャの捕捉: このモデルは、人間が板情報を見て「買い板が厚いから上がりそう

だ」と判断するような直感を、数理的にモデル化したものと言える。スタッキングにおいて、

GBDT や通常の時系列モデルが見落とす「需給の微細構造」からのシグナルを提供する役割を

果たす。

**6.3 TabNet**：決定木と**NN**の融合\*\*\*\*

Google Cloud AI の研究チームによって開発された TabNet は、ニューラルネットワークでありながら、

決定木のような「特徴量選択」のメカニズム（Sequential Attention）を内包している。Ubiquant コンペ

ティションの 1 位解法では、LightGBM と TabNet のアンサンブルが採用された 2。TabNet は、どの特徴

量が重要かを学習過程で明示的に選択するため、解釈可能性が高く、かつ不要な特徴量（ノイズ）

の影響を受けにくい。

**7.**ビットコイン無期限先物に特化した特徴量エンジニアリング\*\*\*\*

モデルの性能は入力データの質に依存する。ここでは、ビットコイン無期限先物市場で特に効力を持

つ「アルファ生成特徴量」を定義する。

**7.1 Funding Rate**と**Basis**関連指標\*\*\*\*

前述の通り、Funding Rate は市場の歪みを表す最重要指標である。

● **Basis**（ベーシス）: 先物価格と現物価格の差。Basis = Futures_Price - Spot_Price。ベーシス

が拡大すると、アービトラージ圧力により平均回帰が発生しやすい。

● **Funding Rate**の移動平均と乖離: 過去$N$期間の平均 FR からの乖離は、短期的な過熱感（

Overbought/Oversold）を示唆する。

● **Predicted Funding Rate**: 取引所が提供する「次回の Funding Rate 予測値」。これが確定す

る直前（8 時間ごとの徴収タイミング）には、支払いを回避しようとするポジション調整が発生し、

価格変動のパターンが生まれる。

**7.2 **マイクロストラクチャ指標** \(Order Book Dynamics\)**

板情報（L2 データ）から計算される指標は、超短期の予測において支配的な力を持つ 20。

● Order Book Imbalance \(OBI\):

$$OBI\_t = \\frac\{V\_t^\{bid\} - V\_t^\{ask\}\}\{V\_t^\{bid\} \+ V\_t^\{ask\}\}$$

ここで $V^\{bid\}, V^\{ask\}$ はベストビッド/アスクの数量。買い板が厚ければプラス、売り板が厚

ければマイナスとなる。

● Order Flow Imbalance \(OFI\): 特定の期間における、板の「更新」に基づく実質的な需給フ

ロー。

$$
OFI\_t = \\sum\_\{i\} \(q\_\{t,i\}^\{bid\} \\mathbb\{I\}\(P\_\{t,i\}^\{bid\} \\ge P\_\{t-1,i\}^\{bid\}\) - q\_\{t,i\}^\{ask\}

\\mathbb\{I\}\(P\_\{t,i\}^\{ask\} \\le P\_\{t-1,i\}^\{ask\}\)\)
$$

これは、単なる板の厚さではなく、「新しい注文がどちらの方向に積み上がっているか」あるいは

「キャンセルされたか」という動的な意図を捉える。

**7.3**オンチェーンデータとセンチメント\*\* \*\*

● **Whale Alert / Large Transactions**: 大口の送金情報は、相場の変動要因となり得るが、即

時性に欠ける場合があるため、長めのタイムホライズンを持つモデル（LSTM/Mamba）への入

力として適している。

● **Sentiment Analysis**: Twitter \(X\) やニュースの見出しから、BERT ベースのモデル（

CryptoBERT など）を用いてセンチメントスコアを算出する。特に「Fear & Greed Index」のような

市場心理指標は、Funding Rate と組み合わせることで、逆張り戦略の精度を高める 22。

**8. Level-2**メタ学習器の設計と統合戦略\*\* \*\*

多様な Level-1 モデルが出揃ったところで、それらを統合する Level-2（メタ）モデルの設計について論

じる。

**8.1**線形モデルによる堅実な統合\*\* \*\*

Kaggle の多くのコンペティションにおいて、メタモデルには複雑な非線形モデルではなく、**Ridge**回帰

（**L2**正則化線形回帰）や単純な加重平均が採用されることが多い 24。

● 理由: Level-1 のモデル群はすでに高度な非線形性を学習済みであるため、Level-2 でさらに複

雑なモデルを使うと、Level-1 モデルの過学習（外れ値への適合）を増幅させてしまうリスクがあ

る。線形モデルは、各ベースモデルの予測値の「信頼度」を重み係数として学習するだけであ

り、スタッキング全体のロバスト性を保つのに適している。

● 制約付き最適化: 重みが負にならないように制約をかけた**Non-negative Least Squares** **\(NNLS\)** も有効である。これにより、直感に反する係数（あるモデルの予測が高いほど最終予

測を下げるなど）を排除できる。

**8.2**動的重み付け（**Dynamic Weighting**）と体制切り替え\*\* \*\*

市場環境に応じて、信頼するモデルを切り替えるアプローチである。

● **Volatility-Adjusted Weighting**: 市場のボラティリティが高い局面では、急変に強い

CryptoMamba や Squeezeformer の重みを増やし、ボラティリティが低いレンジ相場では、平均

回帰を得意とする LightGBM の重みを増やすといった調整を行う。これを実現するために、メタ

モデルにはベースモデルの予測値だけでなく、「現在のボラティリティ」や「出来高」といった\*\*メ

タ特徴量（Meta-features）\*\*を入力する。

**8.3**メタラベリング（**Meta-Labeling**）：最先端のフィルタリング\*\* \*\*

スタッキングの出力（最終予測値）をそのままトレードに使うのではなく、もう一段階のフィルタリング

を噛ませる手法であり、ファイナンシャル ML の権威である Marcos Lopez de Prado 氏によって提唱さ

れている 25。

1. **Primary Model**: 従来の予測モデル（ここまでのスタッキングモデル全体）。「価格が上がるか

下がるか（方向）」、あるいは「エントリーシグナル」を出力する。感度（Recal ）を高めに設定し、

多くのチャンスを捉えさせる。

2. **Meta Model \(Secondary Model\)**:

○ タスク: Primary Model が出したシグナルに対して、\*\*「そのトレードを実行した場合、利益

が出るか（1）損するか（0）」\*\*を予測する 2 値分類モデル。

○ 入力: Primary Model の確信度（Probability）、その瞬間のボラティリティ、スプレッド、板の

厚さ、直近の勝率など。

○ 判定: Meta Model が高い確率で「利益が出る（1）」と判定した場合のみ、実際にエントリー

する。

3. 利点: 方向予測（Primary）と、ベットサイズ/実行可否の判断（Secondary）を分離することで、モ

デルの役割を専門化できる。特に、相場がランダムウォークに近い状態（予測困難な状態）であ

ることを Meta Model が検知し、トレードを見送る（パスする）ことができるため、ドローダウンを劇

的に抑制し、シャープレシオを向上させる効果がある。

**9.**検証手法と実装上の重要事項\*\* \*\*

**9.1 Purged K-Fold Cross Validation**

金融時系列データにおいて、ランダムなシャッフルによる K-Fold 交差検証は厳禁である。未来の情

報が過去の学習データに漏れる「Leakage」が発生するためである。また、単なる時系列分割（Time

Series Split）でも、トレードの保有期間が重なる部分で相関が生じる。

これを防ぐために、Purged K-Fold（パージ付き交差検証）を用いる必要がある 25。

● **Purging**: テストデータの期間と重複する（または近接する）トレーニングデータを削除する。例

えば、予測ホライズンが 1 時間のモデルであれば、テスト期間の前後 1 時間分のデータを学習か

ら除外する。

● **Embargo**: テスト期間の直後のデータも、情報の遅延浸透（Serial Correlation）を考慮して、さ

らに長期間（例えば数パーセント分）除外する。

**9.2**ターゲット変数の設計：**Triple Barrier Method**

教師あり学習のラベル（正解データ）を作成する際、単純な「$n$分後のリターン」を使うと、その間に

含み損が許容範囲を超えていたり、あるいはもっと早く利益確定すべきタイミングがあったりする事

実を無視してしまう。

Triple Barrier Method29 では、以下の 3 つのバリアを設定し、最初に接触したバリアによってラベル

を決定する。

1. 上値バリア（**Profit Taking**）: 目標利益（例：\+2%）。接触すればクラス「1」。

2. 下値バリア（**Stop Loss**）: 許容損失（例：-1%）。接触すればクラス「-1」。

3. 時間バリア（Time Horizon）: 一定時間（例：1 時間）経過。接触すれば、その時点のリターンの符

号に応じてクラス分け、あるいは「0（中立）」とする。

このラベリング手法を用いることで、トレーディングの実態（利確と損切り）に即した現実的なモ

デルを学習させることができる。

**9.3**実装とインフラ\*\* \*\*

● 推論の高速化: スタッキングモデルは巨大になりがちである。Python（PyTorch/Scikit-learn）で

学習させたモデルを、実運用時には**ONNX**形式に変換し、C\+\+や Rust で実装された推論エンジ

ン（ONNX Runtime）で実行することで、レイテンシを最小化する手法が HFT 領域では一般的で

ある 30。

● データパイプライン: Binance 等の WebSocket から LOB データをリアルタイムで受信し、OFI など

の特徴量を計算、モデルに入力してシグナルを出し、発注するまでの一連の処理を、数ミリ

秒〜数十ミリ秒以内に完了させる堅牢なパイプラインが必要となる。

**10.**結論：究極のスタッキング構成\*\* \*\*

ビットコイン無期限先物市場を攻略するための、現時点で考えうる最も有力なスタッキング構成（The

"Holy Grail" Architecture）は以下の通りである。

階層\*\* \*\*

構成モデル**/**手法\*\* \*\*

役割と狙い\*\* \*\*

**Level-1**

**M1: LightGBM / CatBoost**

テーブルデータ（テクニカル、

Funding Rate）からの非線形

ルール学習。高速かつ高精

度。

**M2: Squeezeformer \+**

時系列シーケンスからの長

**GRU**

期依存性・トレンド学習。ノイ

ズ圧縮。

**M3: CryptoMamba \(SSM\)**

超長期コンテキストの保持

と、ボラティリティ急変時の適

応力。

**M4: Encoder \+ MLP**

Denoising Autoencoder によ

**\(ResNet\)**

る特徴量抽出と、ノイズ耐性

の強化。

**M5: DeepLOB**

板情報（画像）からのマイクロ

**\(CNN-LSTM\)**

ストラクチャ・需給不均衡の

検知。

**Level-2**

**Ridge Regression \(NNLS\)**

各ベースモデルの予測値の

線形統合。過学習を防ぎつ

つ最適ウェイトを算出。

**Filter**

**Meta-Labeling**

Level-2 の予測に対する「勝

**\(RF/LGBM\)**

率・収益性」の判定。不要なト

レードのフィルタリング。

このアーキテクチャは、市場を「表形式データ」「時系列シーケンス」「画像（板情報）」「ノイズ」という

多角的な視点から捉え、それぞれのスペシャリストであるモデル群を統合し、さらにメタラベリングに

よって厳選された機会のみに資金を投じるものである。この包括的なアプローチこそが、効率化が進

む暗号資産市場において、持続可能なエッジを確立するための最適解と言えるだろう。

参考文献**\(Selected Citations\)**

● 1 Kaggle Competition Winning Solutions \(G-Research, Ubiquant, etc.\)

● 2 Ubiquant Market Prediction 1st Place Solution \(Ensemble Strategies\)

● 9 Bitcoin Price Prediction & Ensemble Learning \(Academic Papers\)

● 3 Optiver Realized Volatility Prediction \(Volatility Modeling\)

● 10 CryptoMamba: State Space Models for Bitcoin

● 17 Encoder \+ MLP / Denoising Autoencoder Architecture

● 1 Squeezeformer & GRU Hybrid Models

● 25 Meta-Labeling & Financial Machine Learning \(Lopez de Prado\)

● 18 DeepLOB & Microstructure Modeling

引用文献\*\* \*\*

1. Winning solutions of kaggle competitions, 11 月 23, 2025 にアクセス、

<https://www.kaggle.com/code/sudalairajkumar/winning-solutions-of-kaggle-com>

petitions

2. \[1st Place Solution\] - Our Betting Strategy | Kaggle, 11 月 23, 2025 にアクセス、

<https://www.kaggle.com/competitions/ubiquant-market-prediction/writeups/k-i-y>

-1st-place-solution-our-betting-strategy

3. 1st Place Solution - Nearest Neighbors - Kaggle, 11 月 23, 2025 にアクセス、

<https://www.kaggle.com/c/optiver-realized-volatility-prediction/discussion/27497>

0

4. Best Way to Trade Bitcoin Futures \(2025\): Strategies, Risks & Steps - Mudrex Learn, 11 月 23, 2025 にアクセス、

<https://mudrex.com/learn/best-way-to-trade-bitcoin-futures/>

5. Understanding Funding Rates in Perpetual Futures and Their Impact - Coinbase, 11 月 23, 2025 にアクセス、

<https://www.coinbase.com/learn/perpetual-futures/understanding-funding-rates->

in-perpetual-futures

6. Perpetual Futures Pricing\* - Wharton's Finance Department, 11 月 23, 2025 にアクセ

ス、 <https://finance.wharton.upenn.edu/~jermann/AHJ-main-10.pdf>

7. Where smart money does more - Blog - Understanding the Funding Rate in Perpetual Futures - One Trading, 11 月 23, 2025 にアクセス、

<https://www.onetrading.com/blog/understanding-the-funding-rate-in-perpetual->

futures

8. Fundamentals of Perpetual FuturesWe are grateful to Lin Wil iam Cong, Urban Jermann, Shimon Kogan, Tim Roughgarden, Adrien Verdelhan, as wel as

conference participants at the 2024 Utah Winter Finance Conference and seminar participants at a16z Crypto, Hebrew University, Reichman University, and the Virtual Derivatives Workshop for their insightful feedback and helpful comments. Songrun He - arXiv, 11 月 23, 2025 にアクセス、

<https://arxiv.org/html/2212.06888v5>

9. Ensemble Based Approach for Bitcoin Price Prediction - IEEE Xplore, 11 月 23, 2025

にアクセス、 <https://ieeexplore.ieee.org/iel8/11085141/11085148/11086125.pdf>

10. CryptoMamba: Leveraging State Space Models for Accurate Bitcoin Price Prediction - arXiv, 11 月 23, 2025 にアクセス、 <https://arxiv.org/abs/2501.01010>

11. 2nd place solution | Kaggle, 11 月 23, 2025 にアクセス、

<https://www.kaggle.com/competitions/g-research-crypto-forecasting/writeups/n>

athaniel-maddux-2nd-place-solution

12. Trading with Machine Learning: Ensemble Methods - Kaggle, 11 月 23, 2025 にアク

セス、

<https://www.kaggle.com/code/lusfernandotorres/trading-with-machine-learning->

ensemble-methods

13. Kaggle Winning Solutions: AI Trends & Insights, 11 月 23, 2025 にアクセス、

<https://www.kaggle.com/code/tahaalselwi> /kaggle-winning-solutions-ai-trends-in

sights

14. A Novel Hybrid Approach Using an Attention-Based Transformer \+ GRU Model for Predicting Cryptocurrency Prices - MDPI, 11 月 23, 2025 にアクセス、

<https://www.mdpi.com/2227-7390/13/9/1484>

15. 3rd place solution | Kaggle, 11 月 23, 2025 にアクセス、

<https://www.kaggle.com/competitions/g-research-crypto-forecasting/writeups/g>

aba-3rd-place-solution

16. MShahabSepehri/CryptoMamba: The implementation of . . - GitHub, 11 月 23, 2025

にアクセス、 <https://github.com/MShahabSepehri/CryptoMamba>

17. G-Research Crypto Forecasting | Kaggle, 11 月 23, 2025 にアクセス、

<https://www.kaggle.com/competitions/g-research-crypto-forecasting/discussion/>

286676

18. LeonardoBerti00/Deep-Learning-Models-for-financial-time . . - GitHub, 11 月 23, 2025 にアクセス、

<https://github.com/LeonardoBerti00/Deep-Learning-Models-for-financial-time-s>

erie-forecasting-with-LOB-Data

19. Deep Learning Models Meet Financial Data Modalities - arXiv, 11 月 23, 2025 にアク

セス、 <https://arxiv.org/html/2504.13521v2>

20. Learning to Predict Short-Term Volatility with Order Flow Image Representation -

arXiv, 11 月 23, 2025 にアクセス、 <https://arxiv.org/html/2304.02472v2>

21. Cryptocurrency markets microstructure, with a machine learning application to the Binance bitcoin market - UNITesi, 11 月 23, 2025 にアクセス、

<https://unitesi.unive.it/retrieve/eed2f223-f3d3-459e-b4a6-25f233437bde/893488->

1286715.pdf

22. Multi-level deep Q-networks for Bitcoin trading strategies - PMC - NIH, 11 月 23, 2025 にアクセス、 <https://pmc.ncbi.nlm.nih.gov/articles/PMC10774387/>

23. Fusion of Sentiment and Market Signals for Bitcoin Forecasting: A SentiStack

Network Based on a Stacking LSTM Architecture - MDPI, 11 月 23, 2025 にアクセス、

<https://www.mdpi.com/2504-2289/9/6/161>

24. A Stacking Ensemble Deep Learning Model for Bitcoin Price . . - MDPI, 11 月 23, 2025 にアクセス、 <https://www.mdpi.com/2227-7390/10/8/1307>

25. Meta labeling in Cryptocurrencies Market. | by Quang Khải Nguyễn Hưng |

Medium, 11 月 23, 2025 にアクセス、

<https://medium.com/@liangnguyen612/meta-labeling-in-cryptocurrencies-market>

-95f761410fac

26. Meta Labeling Explained | Boost Your Trading Strategy with ML Filters - YouTube, 11 月 23, 2025 にアクセス、 <https://www.youtube.com/watch?v=Sm03GTT6OOw>

27. Why Meta-Labeling Is Not a Silver Bul et by Francesco Baldisserri -

QuantConnect.com, 11 月 23, 2025 にアクセス、

<https://www.quantconnect.com/forum/discussion/14706/why-meta-labeling-is-n>

ot-a-silver-bul et/

28. Meta Labeling for Algorithmic Trading: How to Amplify a Real Edge : r/algotrading

- Reddit, 11 月 23, 2025 にアクセス、

<https://www.reddit.com/r/algotrading/comments/1lnm48w/meta\_labeling\_for\_algo>

rithmic_trading_how_to/

29. Data Label ing - Mlfin.py, 11 月 23, 2025 にアクセス、

<https://mlfinpy.readthedocs.io/en/latest/Label> ing.html

30. Python in High-Frequency Trading: Low-Latency Techniques - PyQuant News, 11

月 23, 2025 にアクセス、

<https://www.pyquantnews.com/free-python-resources/python-in-high-frequenc>

y-trading-low-latency-techniques

31. user-jiyichen/Ubiquant-Market-Prediction-Competition - GitHub, 11 月 23, 2025 に

アクセス、

<https://github.com/user-jiyichen/Ubiquant-Market-Prediction-Competition>

32. Kaggle Solutions, 11 月 23, 2025 にアクセス、 <https://farid.one/kaggle-solutions/>

33. CryptoMamba: Leveraging State Space Models for Accurate Bitcoin Price Prediction - arXiv, 11 月 23, 2025 にアクセス、 <https://arxiv.org/html/2501.01010v1>

34. HKML S5E2 - Optiver Realized Volatility Prediction competition by Caleb Yung, Kaggle Expert - YouTube, 11 月 23, 2025 にアクセス、

<https://www.youtube.com/watch?v=0WLkxkc36Hk>

35. Architecture of the Encoder-MLP | Download Scientific Diagram - ResearchGate, 11 月 23, 2025 にアクセス、

<https://www.researchgate.net/figure/Architecture-of-the-Encoder-MLP\_fig2\_355>

928233

36. Meta-Labeling - Wikipedia, 11 月 23, 2025 にアクセス、

<https://en.wikipedia.org/wiki/Meta-Labeling>

# Document Outline

- ビットコイン無期限先物市場における定量的取引戦略：機械学習モデルのスタッキングとアーキテクチャ選定に関する包括的研究

  - 1. 序論：暗号資産市場におけるアルゴリズム取引の進化と現在地

    - 1.1 市場の成熟と α（アルファ）の枯渇
    - 1.2 機械学習とアンサンブル戦略の台頭

  - 2. ビットコイン無期限先物市場の微細構造と予測困難性

    - 2.1 無期限先物特有のメカニズム：Funding Rate
    - 2.2 清算連鎖（Liquidation Cascades）とテールの肥大化

  - 3. スタッキング戦略の理論的枠組み

    - 3.1 アンサンブル学習の階層構造
    - 3.2 多様性（Diversity）の確保

  - 4. Level-1 ベースモデルの選定：決定木ブースティング \(Tree-based Models\)

    - 4.1 LightGBM：高速性と精度のバランス
    - 4.2 CatBoost：時系列データへの適性
    - 4.3 XGBoost：安定性と実績

  - 5. Level-1 ベースモデルの選定：系列モデリングと深層学習 \(Sequence Models & Deep Learning\)

    - 5.1 LSTM / GRU：リカレントニューラルネットワークの再評価
    - 5.2 Transformer と Squeezeformer：Attention 機構の活用
    - 5.3 Mamba \(State Space Models\)：次世代の時系列モデリング

  - 6. Level-1 ベースモデルの選定：特徴量抽出とノイズ除去 \(Feature Extraction & Denoising\)

    - 6.1 Encoder \+ MLP \(Denoising Autoencoder Strategy\)
    - 6.2 DeepLOB：オーダーブックの画像化
    - 6.3 TabNet：決定木と NN の融合

  - 7. ビットコイン無期限先物に特化した特徴量エンジニアリング

    - 7.1 Funding Rate と Basis 関連指標
    - 7.2 マイクロストラクチャ指標 \(Order Book Dynamics\)
    - 7.3 オンチェーンデータとセンチメント

  - 8. Level-2 メタ学習器の設計と統合戦略

    - 8.1 線形モデルによる堅実な統合
    - 8.2 動的重み付け（Dynamic Weighting）と体制切り替え
    - 8.3 メタラベリング（Meta-Labeling）：最先端のフィルタリング

  - 9. 検証手法と実装上の重要事項

    - 9.1 Purged K-Fold Cross Validation
    - 9.2 ターゲット変数の設計：Triple Barrier Method
    - 9.3 実装とインフラ

  - 10. 結論：究極のスタッキング構成
    - 参考文献 \(Selected Citations\)
      - 引用文献
