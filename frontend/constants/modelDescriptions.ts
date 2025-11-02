export type ModelDescription = {
  title: string;
  description: string;
};

// Essential 4 Models のみのモデル説明
export const MODEL_DESCRIPTIONS: Record<string, ModelDescription> = {
  lightgbm: {
    title: "LightGBM",
    description:
      "Lightning Gradient Boosting - 世界最高級実務モデル。高速で高精度、大規模データ対応、最優秀の実務実績。暗号通貨取引データに最適。",
  },
  xgboost: {
    title: "XGBoost", 
    description:
      "精度で最高レベルの勾配ブースティング。Kaggle実績豊富、最高レベルの精度と堅牢性、特徴量重要度完全サポート。重要プロジェクトに最適。",
  },
  catboost: {
    title: "CatBoost",
    description:
      "カテゴリ特徴量対応の勾配ブースティング。カテゴリエンコーディング自動、過学習防止優秀、前処理時間最小、混合データ型に最適。",
  },
  tabnet: {
    title: "TabNet",
    description:
      "深層学習アプローチの表形式データ用。自動特徴選択、注意力機構、複雑パターン対応、Interpretability良好。高次元データに最適。",
  },
};

export function getModelDescription(key: string): ModelDescription | undefined {
  if (!key) return undefined;
  const normalized = String(key).toLowerCase().trim();
  return MODEL_DESCRIPTIONS[normalized];
}
