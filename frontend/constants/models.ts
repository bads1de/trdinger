// Essential 2 Models のみ
export const AVAILABLE_MODEL_NAMES = [
  "lightgbm",
  "xgboost", 
  "tabnet",
];

export const AVAILABLE_MODELS = [
  {
    value: "lightgbm",
    label: "LightGBM",
    description: "Lightning Gradient Boosting - 世界最高級実務モデル",
  },
  {
    value: "xgboost",
    label: "XGBoost", 
    description: "精度で最高レベルの勾配ブースティング",
  },
  {
    value: "tabnet",
    label: "TabNet",
    description: "深層学習アプローチの表形式データ用",
  },
];

export const META_MODELS = [
  { value: "lightgbm", label: "LightGBM" },
  { value: "xgboost", label: "XGBoost" },
  { value: "tabnet", label: "TabNet" },
];