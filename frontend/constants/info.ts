export const ML_INFO_MESSAGES = {
  maxOhlcvRows:
    "特徴量計算に使うOHLCVデータの最大行数。多いほど長期分析できますがPC負荷も増えます。例: 1時間足で2400行なら約100日分。",
  maxFeatureRows:
    "モデル学習に使う特徴量データの最大行数。多いほど多くのパターンを学べますが、学習時間とメモリ消費が増えます。",
  featureCalculationTimeout:
    "特徴量計算のタイムアウト時間(秒)。複雑な指標を多数使う場合は長めに設定しないと、計算が途中で中断されます。",
  modelTrainingTimeout:
    "モデル学習のタイムアウト時間(秒)。データ量が多い、または複雑なモデルを試す場合は長めに設定しないと、学習が中断されます。",
  learningRate:
    "モデルが学習する際の「歩幅」。小さいと慎重ですが時間がかかり、大きいと速いですが最適解を見逃す事も。0.01～0.1が一般的。",
  numLeaves:
    "モデルの複雑さを決めるパラメータ。大きいほど賢くなりますが、未知のデータに弱くなる「過学習」のリスクが増えます。最初は小さめが推奨。",
  featureFraction:
    "学習時に毎回ランダムで使う特徴量(指標)の割合。意図的に一部を隠すことで、未知の状況に対応しやすくなります(過学習抑制)。",
  baggingFraction:
    "学習時に毎回ランダムで使うデータの割合。毎回違うデータを使うことでモデルの汎用性を高め、過学習を防ぎます。",
  trainTestSplit:
    "データを「学習用」と「実力テスト用」に分ける比率。0.8なら80%で学習し、残りの20%で性能をテストします。",
  predictionHorizon:
    "モデルが何時間先の未来を予測するかを指定します。短期なら1や4、長期なら24などを設定します。",
  thresholdUp:
    "「上昇」と判断するための基準値。大きいほど確信度の高い上昇のみを狙う慎重な判断になります。",
  thresholdDown:
    "「下落」と判断するための基準値。小さい(マイナス方向に大きい)ほど確信度の高い下落のみを狙います。",
  minTrainingSamples:
    "モデル学習を開始するために最低限必要なデータ数。信頼性のあるモデルにはある程度のデータ量が必要です。",
  defaultUpProb:
    "モデルが予測不能な場合に適用される、上昇のデフォルト確率（保険のような値）。",
  defaultDownProb:
    "モデルが予測不能な場合に適用される、下落のデフォルト確率（保険のような値）。",
  defaultRangeProb:
    "モデルが予測不能な場合に適用される、レンジのデフォルト確率（保険のような値）。",
  nCalls:
    "最適なパラメータ設定を見つけるための総試行回数。多いほど良い設定が見つかる可能性は高まりますが、時間もかかります。",
  nInitialPoints:
    "最適化の初期段階で、ランダムに試行する回数。全体像を掴み、狭い範囲に固執するのを防ぎます。",
  acquisitionFunctionEI:
    "獲得関数EI(期待改善量):「探索(未知)」と「活用(既知)」のバランスが良い一般的な設定。",
  acquisitionFunctionPI:
    "獲得関数PI(改善確率): 今見つかっている良い設定をさらに深掘りする「活用」を重視します。",
  acquisitionFunctionUCB:
    "獲得関数UCB(信頼上限): 未知の可能性を試す「探索」を重視します。",
  randomState:
    "実験の再現性を確保するための固定値。同じ番号なら常に同じ結果になり、条件の比較が正確にできます。",
};

export const GA_INFO_MESSAGES = {
  population_size:
    "各世代の候補者(戦略)の数。多いほど多様な戦略を試せますが、計算に時間がかかります。",
  generations:
    "戦略を進化させる世代交代の回数。多いほど戦略は洗練されますが、計算時間も長くなります。",
  mutation_rate:
    "戦略のパラメータがランダムに変化する確率。イノベーションの源ですが、高すぎると良い戦略が壊れるリスクも。",
  crossover_rate:
    "優れた2つの親戦略を組み合わせて新しい子戦略を作る確率。高いほど「良いとこ取り」を積極的に狙います。",
};
