// Essential 5 Models
export const AVAILABLE_MODEL_NAMES = [
  "lightgbm",
  "xgboost",
  "catboost",
  "gru",
  "lstm",
];

export const AVAILABLE_MODELS = [
  {
    value: "lightgbm",
    label: "LightGBM",
    description: "Lightning Gradient Boosting - 高速かつ高精度な実務モデル",
  },
  {
    value: "xgboost",
    label: "XGBoost", 
    description: "eXtreme Gradient Boosting - 安定性と精度に優れたブースティング",
  },
  {
    value: "catboost",
    label: "CatBoost",
    description: "Categorical Boosting - 時系列やカテゴリカル変数に強い",
  },
  {
    value: "gru",
    label: "GRU",
    description: "Gated Recurrent Unit - 短期的な時系列パターンを学習するRNN",
  },
  {
    value: "lstm",
    label: "LSTM",
    description: "Long Short-Term Memory - 長期的な依存関係を捉えるRNN",
  },
];

export const META_MODELS = [
  { value: "logistic_regression", label: "Logistic Regression (NNLS)" },
  { value: "lightgbm", label: "LightGBM" },
  { value: "xgboost", label: "XGBoost" },
];