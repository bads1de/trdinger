export type ModelDescription = {
  title: string;
  description: string;
};

export const MODEL_DESCRIPTIONS: Record<string, ModelDescription> = {
  lightgbm: {
    title: "LightGBM",
    description:
      "たくさんの“決定木”を組み合わせて素早く賢く予測する人気の手法です。計算がとても速く、大きなデータでもサクサク動きます。カテゴリ(文字列)の特徴量も扱いやすく、まず迷ったらこれを試す選択肢になります。",
  },
  xgboost: {
    title: "XGBoost",
    description:
      "定番の高性能モデル。多くの場面で“強いベースライン”になりやすく、細かな設定で伸びしろもあります。迷ったらLightGBMかXGBoostのどちらかを試すのが一般的です。",
  },
  catboost: {
    title: "CatBoost",
    description:
      "文字やカテゴリのような特徴量を“前処理少なめ”で上手に扱えるモデル。データ準備の負担を減らしつつ、精度も出しやすいのが魅力です。",
  },
  tabnet: {
    title: "TabNet",
    description:
      "表形式データ向けのディープラーニング。自動で“どの特徴を見るか”に注目しながら学習します。うまくハマると強力ですが、他の木ベース手法より学習や設定がやや難しめです。",
  },
  randomforest: {
    title: "Random Forest",
    description:
      "たくさんの決定木の“多数決”で予測する安定志向のモデル。過学習(覚えすぎ)に比較的強く、まず無難に使いたいときに役立ちます。",
  },
  extratrees: {
    title: "Extra Trees",
    description:
      "ランダム性をより強く加えた決定木の集まり。学習が速く、ノイズが多いデータでも安定しやすい傾向があります。",
  },
  gradientboosting: {
    title: "Gradient Boosting",
    description:
      "誤りを少しずつ修正しながら木を積み重ねる考え方のモデル。学習率などの設定次第で性能が大きく変わるため、丁寧なチューニングが活きます。",
  },
  adaboost: {
    title: "AdaBoost",
    description:
      "間違えたデータを重点的に学習していく“テコ入れ”型のモデル。外れ値(極端な値)が多いと影響を受けやすい点に注意が必要です。",
  },
  ridge: {
    title: "Ridge (線形モデル)",
    description:
      "シンプルで速い“直線的な”発想のモデル。複雑なパターンは苦手ですが、軽量でベースラインや学習の土台づくりに向いています。",
  },
  naivebayes: {
    title: "Naive Bayes",
    description:
      "とても軽量・高速な確率モデル。特徴同士は独立と仮定します。テキスト分類などで手早く目安を出したいときに便利です。",
  },
  knn: {
    title: "k-Nearest Neighbors",
    description:
      "“似ているデータ”を探して決めるシンプルな方法。学習は速い一方で、予測時に探す処理が重くなることがあります。小規模データや直感的な検証に向いています。",
  },
};

export function getModelDescription(key: string): ModelDescription | undefined {
  if (!key) return undefined;
  const normalized = String(key).toLowerCase().trim();
  return MODEL_DESCRIPTIONS[normalized];
}