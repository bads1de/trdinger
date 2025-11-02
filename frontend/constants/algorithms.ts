/**
 * Essential 2 Models アルゴリズム定数定義
 * 
 * バックエンドのalgorithm_registry.pyと同期してEssential 2 Modelsのみ定義します。
 * フロントエンドのアルゴリズム選択を簡潔で明確に保ちます。
 */

// アルゴリズムタイプの定義
export enum AlgorithmType {
  BOOSTING = 'boosting',
  NEURAL_NETWORK = 'neural_network',
}

// アルゴリズム機能の定義
export enum AlgorithmCapability {
  CLASSIFICATION = 'classification',
  REGRESSION = 'regression',
  PROBABILITY_PREDICTION = 'probability_prediction',
  FEATURE_IMPORTANCE = 'feature_importance',
  MULTICLASS = 'multiclass',
}

// アルゴリズム情報の型定義
export interface Algorithm {
  name: string;
  display_name: string;
  description: string;
  type: AlgorithmType;
  capabilities: AlgorithmCapability[];
  pros: string[];
  cons: string[];
  best_for: string[];
  has_probability_prediction: boolean;
  has_feature_importance: boolean;
  note?: string;
}

// アルゴリズムタイプの日本語マッピング
export const ALGORITHM_TYPE_LABELS: Record<AlgorithmType, string> = {
  [AlgorithmType.BOOSTING]: 'ブースティング系',
  [AlgorithmType.NEURAL_NETWORK]: 'ニューラルネットワーク',
};

// アルゴリズム機能の日本語マッピング
export const CAPABILITY_LABELS: Record<AlgorithmCapability, string> = {
  [AlgorithmCapability.CLASSIFICATION]: '分類',
  [AlgorithmCapability.REGRESSION]: '回帰',
  [AlgorithmCapability.PROBABILITY_PREDICTION]: '確率予測',
  [AlgorithmCapability.FEATURE_IMPORTANCE]: '特徴量重要度',
  [AlgorithmCapability.MULTICLASS]: '多クラス分類',
};

// Essential 4 Algorithms のみ
export const ALGORITHMS: Record<string, Algorithm> = {
  // LightGBM - 世界最高級実務モデル
  lightgbm: {
    name: 'lightgbm',
    display_name: 'LightGBM',
    description: 'Lightning Gradient Boosting - 世界最高級実務モデル',
    type: AlgorithmType.BOOSTING,
    capabilities: [
      AlgorithmCapability.CLASSIFICATION,
      AlgorithmCapability.REGRESSION,
      AlgorithmCapability.PROBABILITY_PREDICTION,
      AlgorithmCapability.FEATURE_IMPORTANCE,
      AlgorithmCapability.MULTICLASS,
    ],
    pros: [
      '世界最高級実務実績',
      '最速処理',
      'メモリ効率優秀',
      '大規模データ対応',
      '高精度予測'
    ],
    cons: [
      'カテゴリ特徴量に直接対応しない',
      'ハイパーパラメータ調整が重要'
    ],
    best_for: [
      '大規模データ',
      '高精度が必要',
      'リアルタイム予測',
      'メモリ制約環境',
      '高速学習が必要'
    ],
    has_probability_prediction: true,
    has_feature_importance: true,
  },

  // XGBoost - 最高精度
  xgboost: {
    name: 'xgboost',
    display_name: 'XGBoost',
    description: '精度で最高レベルの勾配ブースティング',
    type: AlgorithmType.BOOSTING,
    capabilities: [
      AlgorithmCapability.CLASSIFICATION,
      AlgorithmCapability.REGRESSION,
      AlgorithmCapability.PROBABILITY_PREDICTION,
      AlgorithmCapability.FEATURE_IMPORTANCE,
      AlgorithmCapability.MULTICLASS,
    ],
    pros: [
      '最高レベル精度',
      'Kaggle実績豊富',
      '特徴量重要度完全サポート',
      '堅牢性最高',
      ' регуларизация 内蔵'
    ],
    cons: [
      '学習時間長い',
      'メモリ使用量大',
      'パラメータ調整複雑'
    ],
    best_for: [
      '最高精度が必要',
      '重要プロジェクト',
      '競争環境',
      '特徴量エンジニアリング済み',
      '予測精度が最重要'
    ],
    has_probability_prediction: true,
    has_feature_importance: true,
  },

  // TabNet - 深層学習アプローチ
  tabnet: {
    name: 'tabnet',
    display_name: 'TabNet',
    description: '深層学習アプローチの表形式データ用',
    type: AlgorithmType.NEURAL_NETWORK,
    capabilities: [
      AlgorithmCapability.CLASSIFICATION,
      AlgorithmCapability.REGRESSION,
      AlgorithmCapability.PROBABILITY_PREDICTION,
      AlgorithmCapability.MULTICLASS,
    ],
    pros: [
      '自動特徴選択',
      'Interpretability良好',
      '複雑パターン対応',
      ' внимания mechanism',
      '段階的学習'
    ],
    cons: [
      '計算リソース必要',
      'ハイパーパラメータ複雑',
      '学習時間長い',
      'データサイズ Larger必要'
    ],
    best_for: [
      '深層学習アプローチ',
      '特徴量自動選択',
      '複雑データパターン',
      '高次元データ',
      'Interpretability重視'
    ],
    has_probability_prediction: true,
    has_feature_importance: false, // TabNet uses attention mechanisms
  },
};

// アルゴリズム配列（表示用）
export const ALGORITHM_LIST: Algorithm[] = Object.values(ALGORITHMS);

// アルゴリズム名のリスト
export const ALGORITHM_NAMES: string[] = Object.keys(ALGORITHMS);

// タイプ別アルゴリズム
export const ALGORITHMS_BY_TYPE: Record<string, Algorithm[]> = ALGORITHM_LIST.reduce(
  (acc, algorithm) => {
    const typeLabel = ALGORITHM_TYPE_LABELS[algorithm.type];
    if (!acc[typeLabel]) {
      acc[typeLabel] = [];
    }
    acc[typeLabel].push(algorithm);
    return acc;
  },
  {} as Record<string, Algorithm[]>
);

// 機能別アルゴリズム
export const ALGORITHMS_BY_CAPABILITY: Record<string, Algorithm[]> = ALGORITHM_LIST.reduce(
  (acc, algorithm) => {
    algorithm.capabilities.forEach(capability => {
      const capLabel = CAPABILITY_LABELS[capability];
      if (!acc[capLabel]) {
        acc[capLabel] = [];
      }
      acc[capLabel].push(algorithm);
    });
    return acc;
  },
  {} as Record<string, Algorithm[]>
);

// 確率予測対応アルゴリズム
export const PROBABILITY_ALGORITHMS: Algorithm[] = ALGORITHM_LIST.filter(
  algorithm => algorithm.has_probability_prediction
);

// 特徴量重要度対応アルゴリズム
export const FEATURE_IMPORTANCE_ALGORITHMS: Algorithm[] = ALGORITHM_LIST.filter(
  algorithm => algorithm.has_feature_importance
);

// アルゴリズム統計情報
export const ALGORITHM_STATISTICS = {
  total: ALGORITHM_LIST.length,
  byType: Object.entries(ALGORITHMS_BY_TYPE).map(([type, algorithms]) => ({
    type,
    count: algorithms.length,
    algorithms: algorithms.map(a => a.name),
  })),
  byCapability: Object.entries(ALGORITHMS_BY_CAPABILITY).map(([capability, algorithms]) => ({
    capability,
    count: algorithms.length,
    algorithms: algorithms.map(a => a.name),
  })),
  probabilityCount: PROBABILITY_ALGORITHMS.length,
  featureImportanceCount: FEATURE_IMPORTANCE_ALGORITHMS.length,
};
