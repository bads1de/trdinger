/**
 * 機械学習評価指標の説明文定数
 * 
 * 各評価指標について、素人にもわかりやすい説明、値の範囲、
 * 解釈方法を提供します。
 */

export interface MetricInfo {
  name: string;
  description: string;
  range: string;
  interpretation: string;
  example: string;
  whenImportant: string;
}

export const ML_METRICS_INFO: Record<string, MetricInfo> = {
  accuracy: {
    name: "精度 (Accuracy)",
    description: "全体の予測のうち、正しく予測できた割合を示します。最も基本的な評価指標です。",
    range: "0% ～ 100%",
    interpretation: "高いほど良い。80%以上なら優秀、70%以上なら良好、60%以下は改善が必要です。",
    example: "100回予測して80回正解なら80%の精度です。",
    whenImportant: "データが均衡している場合（各クラスの数が同じくらい）に最も有効です。"
  },

  precision: {
    name: "適合率 (Precision)",
    description: "「上昇」と予測したもののうち、実際に上昇した割合です。偽陽性（間違った上昇予測）を避けたい場合に重要です。",
    range: "0% ～ 100%",
    interpretation: "高いほど良い。予測の信頼性を表します。80%以上なら信頼できる予測です。",
    example: "「上昇」と10回予測して8回実際に上昇なら80%の適合率です。",
    whenImportant: "間違った買いシグナルを避けたい場合や、確実性を重視する戦略で重要です。"
  },

  recall: {
    name: "再現率 (Recall)",
    description: "実際に上昇した場面のうち、正しく「上昇」と予測できた割合です。見逃しを避けたい場合に重要です。",
    range: "0% ～ 100%",
    interpretation: "高いほど良い。チャンスを逃さない能力を表します。80%以上なら見逃しが少ないです。",
    example: "実際に10回上昇して8回予測できたなら80%の再現率です。",
    whenImportant: "利益機会を逃したくない場合や、網羅性を重視する戦略で重要です。"
  },

  f1_score: {
    name: "F1スコア",
    description: "適合率と再現率のバランスを取った指標です。両方を同時に考慮したい場合に使用します。",
    range: "0% ～ 100%",
    interpretation: "高いほど良い。適合率と再現率の調和平均です。75%以上なら優秀なバランスです。",
    example: "適合率80%、再現率70%なら、F1スコアは約75%になります。",
    whenImportant: "予測の正確性と網羅性の両方を重視したい場合に最適です。"
  },

  auc_roc: {
    name: "AUC-ROC",
    description: "様々な閾値での分類性能を総合的に評価した指標です。確率的な予測の品質を測ります。",
    range: "0% ～ 100%",
    interpretation: "50%はランダム、70%以上は良好、80%以上は優秀、90%以上は非常に優秀です。",
    example: "80%なら、ランダムに選んだ2つのケースを80%の確率で正しく順序付けできます。",
    whenImportant: "予測確率の信頼性や、閾値に依存しない性能評価をしたい場合に重要です。"
  },

  balanced_accuracy: {
    name: "バランス精度",
    description: "各クラスの予測精度を均等に評価した指標です。データが不均衡な場合に通常の精度より信頼できます。",
    range: "0% ～ 100%",
    interpretation: "高いほど良い。不均衡データでは通常の精度より重要です。70%以上なら良好です。",
    example: "上昇が少ないデータでも、各クラスを公平に評価した精度を示します。",
    whenImportant: "上昇・下降・横ばいの発生頻度が大きく異なる場合に特に重要です。"
  },

  matthews_corrcoef: {
    name: "マシューズ相関係数 (MCC)",
    description: "予測と実際の結果の相関を-1から1で表す指標です。不均衡データでも信頼できる総合的な性能指標です。",
    range: "-100% ～ 100%",
    interpretation: "0%はランダム、50%以上は良好、70%以上は優秀です。負の値は予測が逆効果を意味します。",
    example: "80%なら予測と実際の結果に強い正の相関があることを示します。",
    whenImportant: "データが不均衡で、総合的な予測性能を1つの指標で評価したい場合に最適です。"
  },

  specificity: {
    name: "特異度 (Specificity)",
    description: "実際に下降した場面のうち、正しく「下降」と予測できた割合です。偽陽性を避ける能力を示します。",
    range: "0% ～ 100%",
    interpretation: "高いほど良い。間違った上昇予測を避ける能力を表します。80%以上なら優秀です。",
    example: "実際に10回下降して8回正しく予測したなら80%の特異度です。",
    whenImportant: "間違った買いシグナルによる損失を避けたい場合に重要です。"
  },

  auc_pr: {
    name: "PR-AUC",
    description: "適合率と再現率の関係を総合的に評価した指標です。不均衡データでAUC-ROCより信頼できます。",
    range: "0% ～ 100%",
    interpretation: "高いほど良い。不均衡データでは特に重要です。60%以上なら良好、80%以上なら優秀です。",
    example: "上昇が稀な場合でも、上昇予測の品質を適切に評価できます。",
    whenImportant: "上昇機会が少ない市場で、上昇予測の精度を重視する場合に重要です。"
  },

  cohen_kappa: {
    name: "コーエンのカッパ",
    description: "偶然の一致を除いた予測の一致度を測る指標です。ランダム予測との差を明確に示します。",
    range: "-100% ～ 100%",
    interpretation: "0%はランダム、40%以上は中程度、60%以上は良好、80%以上は優秀です。",
    example: "60%なら、ランダム予測を大きく上回る一致度があることを示します。",
    whenImportant: "予測がランダムよりもどれだけ優れているかを明確に知りたい場合に有用です。"
  },

  log_loss: {
    name: "対数損失 (Log Loss)",
    description: "予測確率の品質を測る指標です。確信を持った間違い予測に大きなペナルティを与えます。",
    range: "0 ～ ∞（低いほど良い）",
    interpretation: "低いほど良い。0.5以下なら良好、0.3以下なら優秀です。高い値は過信した間違い予測を示します。",
    example: "90%の確信で間違えると大きなペナルティ、60%の確信なら小さなペナルティです。",
    whenImportant: "予測確率の信頼性や、過信による大きな損失を避けたい場合に重要です。"
  },

  brier_score: {
    name: "ブライアスコア",
    description: "確率予測の精度を測る指標です。予測確率と実際の結果の差の二乗平均です。",
    range: "0 ～ 1（低いほど良い）",
    interpretation: "低いほど良い。0.2以下なら良好、0.1以下なら優秀です。0.25はランダム予測のレベルです。",
    example: "70%の確率で上昇予測して実際に上昇なら、スコアは(0.7-1)²=0.09です。",
    whenImportant: "確率予測の正確性を重視し、リスク管理に確率を使用する場合に重要です。"
  },

  sensitivity: {
    name: "感度 (Sensitivity)",
    description: "再現率と同じ指標です。医学分野でよく使われる用語で、陽性を正しく検出する能力を示します。",
    range: "0% ～ 100%",
    interpretation: "高いほど良い。上昇機会を見逃さない能力を表します。80%以上なら優秀です。",
    example: "実際の上昇を見逃さずに検出する能力を示します。",
    whenImportant: "利益機会を逃したくない場合や、網羅的な検出が必要な場合に重要です。"
  },

  npv: {
    name: "陰性的中率 (NPV)",
    description: "「下降」と予測したもののうち、実際に下降した割合です。下降予測の信頼性を示します。",
    range: "0% ～ 100%",
    interpretation: "高いほど良い。下降予測の正確性を表します。80%以上なら信頼できる下降予測です。",
    example: "「下降」と10回予測して8回実際に下降なら80%のNPVです。",
    whenImportant: "売りシグナルの信頼性や、下降予測に基づく戦略で重要です。"
  },

  ppv: {
    name: "陽性的中率 (PPV)",
    description: "適合率と同じ指標です。「上昇」予測の正確性を示し、偽陽性を避ける能力を表します。",
    range: "0% ～ 100%",
    interpretation: "高いほど良い。上昇予測の信頼性を表します。80%以上なら信頼できる上昇予測です。",
    example: "「上昇」予測の的中率を示し、間違った買いシグナルの頻度を表します。",
    whenImportant: "買いシグナルの信頼性や、確実性を重視する投資戦略で重要です。"
  }
};

/**
 * 評価指標のカテゴリ分類
 */
export const METRIC_CATEGORIES = {
  basic: {
    name: "基本指標",
    metrics: ["accuracy", "precision", "recall", "f1_score"],
    description: "最も基本的で理解しやすい評価指標"
  },
  advanced: {
    name: "高度な指標",
    metrics: ["auc_roc", "auc_pr", "balanced_accuracy", "matthews_corrcoef"],
    description: "より詳細で信頼性の高い評価指標"
  },
  specialized: {
    name: "専門指標",
    metrics: ["specificity", "sensitivity", "npv", "ppv", "cohen_kappa"],
    description: "特定の用途に特化した評価指標"
  },
  probabilistic: {
    name: "確率指標",
    metrics: ["log_loss", "brier_score"],
    description: "予測確率の品質を評価する指標"
  }
};

/**
 * 指標の重要度レベル
 */
export const METRIC_IMPORTANCE = {
  high: ["accuracy", "f1_score", "auc_roc", "balanced_accuracy"],
  medium: ["precision", "recall", "matthews_corrcoef", "auc_pr"],
  low: ["specificity", "sensitivity", "npv", "ppv", "cohen_kappa", "log_loss", "brier_score"]
};
