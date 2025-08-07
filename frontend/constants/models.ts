// 新しいアルゴリズムを含む利用可能なモデル
export const AVAILABLE_MODEL_NAMES = [
  "lightgbm",
  "xgboost",
  "catboost",
  "tabnet",
  "randomforest",
  "extratrees",
  "gradientboosting",
  "adaboost",
  "ridge",
  "naivebayes",
  "knn",
];

export const AVAILABLE_MODELS = [
  {
    value: "lightgbm",
    label: "LightGBM",
    description: "高速で高精度な勾配ブースティング",
  },
  {
    value: "xgboost",
    label: "XGBoost",
    description: "強力な勾配ブースティング",
  },
  {
    value: "catboost",
    label: "CatBoost",
    description: "カテゴリ特徴量に強い勾配ブースティング",
  },
  {
    value: "tabnet",
    label: "TabNet",
    description: "表形式データ用ニューラルネットワーク",
  },
  {
    value: "randomforest",
    label: "Random Forest",
    description: "アンサンブル決定木",
  },
  {
    value: "extratrees",
    label: "Extra Trees",
    description: "ランダム性強化決定木",
  },
  {
    value: "gradientboosting",
    label: "Gradient Boosting",
    description: "勾配ブースティング",
  },
  {
    value: "adaboost",
    label: "AdaBoost",
    description: "適応的ブースティング",
  },
  {
    value: "ridge",
    label: "Ridge Classifier",
    description: "L2正則化線形分類器",
  },
  {
    value: "naivebayes",
    label: "Naive Bayes",
    description: "ベイズ分類器",
  },
  {
    value: "knn",
    label: "K-Nearest Neighbors",
    description: "K近傍法",
  },
];

export const META_MODELS = [
  { value: "randomforest", label: "Random Forest" },
  { value: "lightgbm", label: "LightGBM" },
  { value: "ridge", label: "Ridge Classifier" },
  { value: "naivebayes", label: "Naive Bayes" },
];