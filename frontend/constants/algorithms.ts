/**
 * 機械学習アルゴリズム定数定義
 * 
 * バックエンドのalgorithm_registry.pyと同期している必要があります。
 * アルゴリズムの基本情報を定数として定義し、API呼び出しを削減します。
 */

// アルゴリズムタイプの定義
export enum AlgorithmType {
  TREE_BASED = 'tree_based',
  LINEAR = 'linear',
  ENSEMBLE = 'ensemble',
  BOOSTING = 'boosting',
  PROBABILISTIC = 'probabilistic',
  INSTANCE_BASED = 'instance_based',
  NEURAL_NETWORK = 'neural_network',
}

// アルゴリズム機能の定義
export enum AlgorithmCapability {
  CLASSIFICATION = 'classification',
  REGRESSION = 'regression',
  PROBABILITY_PREDICTION = 'probability_prediction',
  FEATURE_IMPORTANCE = 'feature_importance',
  INCREMENTAL_LEARNING = 'incremental_learning',
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
  [AlgorithmType.TREE_BASED]: 'ツリー系',
  [AlgorithmType.LINEAR]: '線形系',
  [AlgorithmType.ENSEMBLE]: 'アンサンブル系',
  [AlgorithmType.BOOSTING]: 'ブースティング系',
  [AlgorithmType.PROBABILISTIC]: '確率的',
  [AlgorithmType.INSTANCE_BASED]: 'インスタンスベース',
  [AlgorithmType.NEURAL_NETWORK]: 'ニューラルネットワーク',
};

// アルゴリズム機能の日本語マッピング
export const CAPABILITY_LABELS: Record<AlgorithmCapability, string> = {
  [AlgorithmCapability.CLASSIFICATION]: '分類',
  [AlgorithmCapability.REGRESSION]: '回帰',
  [AlgorithmCapability.PROBABILITY_PREDICTION]: '確率予測',
  [AlgorithmCapability.FEATURE_IMPORTANCE]: '特徴量重要度',
  [AlgorithmCapability.INCREMENTAL_LEARNING]: '増分学習',
  [AlgorithmCapability.MULTICLASS]: '多クラス分類',
};

// 利用可能なアルゴリズム定義（バックエンドのalgorithm_registry.pyと同期）
export const ALGORITHMS: Record<string, Algorithm> = {
  // ツリー系
  randomforest: {
    name: 'randomforest',
    display_name: 'ランダムフォレスト',
    description: 'ランダムフォレスト - 複数の決定木のアンサンブル',
    type: AlgorithmType.TREE_BASED,
    capabilities: [
      AlgorithmCapability.CLASSIFICATION,
      AlgorithmCapability.PROBABILITY_PREDICTION,
      AlgorithmCapability.FEATURE_IMPORTANCE,
      AlgorithmCapability.MULTICLASS,
    ],
    pros: ['高い精度', '特徴量重要度', 'オーバーフィッティング耐性'],
    cons: ['解釈性が低い', 'メモリ使用量大'],
    best_for: ['中規模データ', 'ノイズ耐性が必要', '特徴量重要度が必要'],
    has_probability_prediction: true,
    has_feature_importance: true,
  },

  extratrees: {
    name: 'extratrees',
    display_name: 'エクストラツリー',
    description: 'エクストラツリー - より高いランダム性を持つ決定木アンサンブル',
    type: AlgorithmType.TREE_BASED,
    capabilities: [
      AlgorithmCapability.CLASSIFICATION,
      AlgorithmCapability.PROBABILITY_PREDICTION,
      AlgorithmCapability.FEATURE_IMPORTANCE,
      AlgorithmCapability.MULTICLASS,
    ],
    pros: ['高速学習', 'オーバーフィッティング耐性', '高い汎化性能'],
    cons: ['解釈性が低い', 'ハイパーパラメータ調整が重要'],
    best_for: ['大規模データ', '高速学習が必要', 'ノイズの多いデータ'],
    has_probability_prediction: true,
    has_feature_importance: true,
  },

  // ブースティング系
  gradientboosting: {
    name: 'gradientboosting',
    display_name: 'グラディエントブースティング',
    description: 'グラディエントブースティング - 逐次的に弱学習器を改善',
    type: AlgorithmType.BOOSTING,
    capabilities: [
      AlgorithmCapability.CLASSIFICATION,
      AlgorithmCapability.PROBABILITY_PREDICTION,
      AlgorithmCapability.FEATURE_IMPORTANCE,
      AlgorithmCapability.MULTICLASS,
    ],
    pros: ['高い精度', '特徴量重要度', '柔軟性'],
    cons: ['オーバーフィッティングしやすい', '学習時間長'],
    best_for: ['高精度が必要', '構造化データ', '特徴量エンジニアリング済み'],
    has_probability_prediction: true,
    has_feature_importance: true,
  },

  adaboost: {
    name: 'adaboost',
    display_name: 'アダブースト',
    description: 'アダブースト - 適応的ブースティング',
    type: AlgorithmType.BOOSTING,
    capabilities: [
      AlgorithmCapability.CLASSIFICATION,
      AlgorithmCapability.PROBABILITY_PREDICTION,
      AlgorithmCapability.FEATURE_IMPORTANCE,
      AlgorithmCapability.MULTICLASS,
    ],
    pros: ['シンプル', '解釈しやすい', '少ないハイパーパラメータ'],
    cons: ['ノイズに敏感', '外れ値に弱い'],
    best_for: ['クリーンなデータ', 'シンプルなモデルが必要', '二値分類'],
    has_probability_prediction: true,
    has_feature_importance: true,
  },

  // 線形系
  ridge: {
    name: 'ridge',
    display_name: 'リッジ分類器',
    description: 'リッジ分類器 - L2正則化線形分類器',
    type: AlgorithmType.LINEAR,
    capabilities: [
      AlgorithmCapability.CLASSIFICATION,
      AlgorithmCapability.FEATURE_IMPORTANCE,
      AlgorithmCapability.MULTICLASS,
    ],
    pros: ['高速', '解釈しやすい', '正則化効果'],
    cons: ['確率予測なし', '非線形関係を捉えられない'],
    best_for: ['線形関係', '高次元データ', '高速予測が必要'],
    has_probability_prediction: false,
    has_feature_importance: true,
    note: 'predict_probaメソッドなし',
  },

  // 確率的
  naivebayes: {
    name: 'naivebayes',
    display_name: 'ナイーブベイズ',
    description: 'ナイーブベイズ - ベイズの定理に基づく確率的分類器',
    type: AlgorithmType.PROBABILISTIC,
    capabilities: [
      AlgorithmCapability.CLASSIFICATION,
      AlgorithmCapability.PROBABILITY_PREDICTION,
      AlgorithmCapability.MULTICLASS,
      AlgorithmCapability.INCREMENTAL_LEARNING,
    ],
    pros: ['高速', '少ないデータでも動作', '確率的解釈'],
    cons: ['特徴量独立性の仮定', '連続値に制限'],
    best_for: ['テキスト分類', '小規模データ', '高速学習が必要'],
    has_probability_prediction: true,
    has_feature_importance: false,
  },

  // インスタンスベース
  knn: {
    name: 'knn',
    display_name: 'K近傍法',
    description: 'K近傍法 - 近傍インスタンスに基づく分類',
    type: AlgorithmType.INSTANCE_BASED,
    capabilities: [
      AlgorithmCapability.CLASSIFICATION,
      AlgorithmCapability.PROBABILITY_PREDICTION,
      AlgorithmCapability.MULTICLASS,
    ],
    pros: ['シンプル', '非線形関係対応', '局所的パターン検出'],
    cons: ['計算コスト高', 'メモリ使用量大', '次元の呪い'],
    best_for: ['小規模データ', '局所的パターン', '非線形関係'],
    has_probability_prediction: true,
    has_feature_importance: false,
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
