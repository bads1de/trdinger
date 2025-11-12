/**
 * ML設定の定数定義
 *
 * バックエンドのML設定に対応する定数値を定義します。
 * ラベルプリセット、閾値計算方法、特徴量プロファイルなどの定数を含みます。
 *
 * 対応バックエンドファイル:
 * - backend/app/utils/label_generation/enums.py
 * - backend/app/utils/label_generation/presets.py
 */

import type {
  FeatureProfile,
  LabelPresetInfo,
  SupportedTimeframe,
  ThresholdMethod,
} from "@/types/ml-config";

/**
 * 閾値計算方法の一覧
 *
 * @see backend/app/utils/label_generation/enums.py:ThresholdMethod
 */
export const THRESHOLD_METHODS: ThresholdMethod[] = [
  "FIXED",
  "QUANTILE",
  "PERCENTILE",
  "STD_DEVIATION",
  "ADAPTIVE",
  "DYNAMIC_VOLATILITY",
  "KBINS_DISCRETIZER",
];

/**
 * 閾値計算方法の表示名マッピング
 */
export const THRESHOLD_METHOD_LABELS: Record<ThresholdMethod, string> = {
  FIXED: "固定閾値",
  QUANTILE: "分位数ベース",
  PERCENTILE: "パーセンタイルベース",
  STD_DEVIATION: "標準偏差ベース",
  ADAPTIVE: "適応的閾値",
  DYNAMIC_VOLATILITY: "動的ボラティリティベース",
  KBINS_DISCRETIZER: "KBinsDiscretizerベース（推奨）",
};

/**
 * 閾値計算方法の説明
 */
export const THRESHOLD_METHOD_DESCRIPTIONS: Record<ThresholdMethod, string> = {
  FIXED: "固定された閾値を使用します。threshold値がそのまま使用されます。",
  QUANTILE:
    "分位数に基づいて閾値を計算します。データ分布を考慮した動的な閾値設定が可能です。",
  PERCENTILE:
    "パーセンタイルに基づいて閾値を計算します（QUANTILEと同じ）。",
  STD_DEVIATION:
    "標準偏差に基づいて閾値を計算します。ボラティリティを考慮した動的な閾値設定が可能です。",
  ADAPTIVE:
    "GridSearchCVを使用して最適な閾値を自動的に探索します。計算時間がかかりますが、最も精度の高い閾値を見つけられます。",
  DYNAMIC_VOLATILITY:
    "動的なボラティリティに基づいて閾値を計算します。市場の変動性に応じた閾値設定が可能です。",
  KBINS_DISCRETIZER:
    "KBinsDiscretizerを使用してデータを3つのビンに分割します。バランスの取れたラベル分布が得られます（推奨）。",
};

/**
 * 特徴量プロファイル関連の定数（削除予定）
 * 
 * @deprecated 研究目的専用のためプロファイル概念は不要になりました
 * 後方互換性のためのみ残されています
 */
export const FEATURE_PROFILES: FeatureProfile[] = ["research", "production"];
export const FEATURE_PROFILE_LABELS: Record<FeatureProfile, string> = {
  research: "研究用（全特徴量）",
  production: "本番用（選択された特徴量）",
};
export const FEATURE_PROFILE_DESCRIPTIONS: Record<FeatureProfile, string> = {
  research:
    "全ての特徴量を使用します。研究・実験用途に適しています。計算時間が長くなる可能性があります。",
  production:
    "厳選された特徴量のみを使用します。本番運用に適しています。計算が高速で、過学習のリスクが低減されます。",
};

/**
 * サポートされる時間足の一覧
 *
 * @see backend/app/utils/label_generation/presets.py:SUPPORTED_TIMEFRAMES
 */
export const SUPPORTED_TIMEFRAMES: SupportedTimeframe[] = [
  "15m",
  "30m",
  "1h",
  "4h",
  "1d",
];

/**
 * 時間足の表示名マッピング
 */
export const TIMEFRAME_LABELS: Record<SupportedTimeframe, string> = {
  "15m": "15分足",
  "30m": "30分足",
  "1h": "1時間足",
  "4h": "4時間足",
  "1d": "1日足",
};

/**
 * ラベル生成プリセットの定義
 *
 * バックエンドのget_common_presets()に対応するプリセット定義です。
 *
 * @see backend/app/utils/label_generation/presets.py:get_common_presets()
 */
export const LABEL_PRESETS: Record<string, LabelPresetInfo> = {
  // 15分足プリセット
  "15m_4bars": {
    name: "15m_4bars",
    timeframe: "15m",
    horizonN: 4,
    threshold: 0.001,
    thresholdMethod: "FIXED",
    description: "15分足、4本先（1時間先）、±0.1%固定閾値",
  },
  "15m_8bars": {
    name: "15m_8bars",
    timeframe: "15m",
    horizonN: 8,
    threshold: 0.0015,
    thresholdMethod: "FIXED",
    description: "15分足、8本先（2時間先）、±0.15%固定閾値",
  },

  // 30分足プリセット
  "30m_4bars": {
    name: "30m_4bars",
    timeframe: "30m",
    horizonN: 4,
    threshold: 0.0015,
    thresholdMethod: "FIXED",
    description: "30分足、4本先（2時間先）、±0.15%固定閾値",
  },
  "30m_8bars": {
    name: "30m_8bars",
    timeframe: "30m",
    horizonN: 8,
    threshold: 0.002,
    thresholdMethod: "FIXED",
    description: "30分足、8本先（4時間先）、±0.2%固定閾値",
  },

  // 1時間足プリセット
  "1h_4bars": {
    name: "1h_4bars",
    timeframe: "1h",
    horizonN: 4,
    threshold: 0.002,
    thresholdMethod: "FIXED",
    description: "1時間足、4本先（4時間先）、±0.2%固定閾値",
  },
  "1h_8bars": {
    name: "1h_8bars",
    timeframe: "1h",
    horizonN: 8,
    threshold: 0.003,
    thresholdMethod: "FIXED",
    description: "1時間足、8本先（8時間先）、±0.3%固定閾値",
  },
  "1h_16bars": {
    name: "1h_16bars",
    timeframe: "1h",
    horizonN: 16,
    threshold: 0.004,
    thresholdMethod: "FIXED",
    description: "1時間足、16本先（16時間先）、±0.4%固定閾値",
  },

  // 4時間足プリセット（デフォルト推奨）
  "4h_4bars": {
    name: "4h_4bars",
    timeframe: "4h",
    horizonN: 4,
    threshold: 0.002,
    thresholdMethod: "FIXED",
    description: "4時間足、4本先（16時間先）、±0.2%固定閾値",
  },
  "4h_6bars": {
    name: "4h_6bars",
    timeframe: "4h",
    horizonN: 6,
    threshold: 0.003,
    thresholdMethod: "FIXED",
    description: "4時間足、6本先（24時間先）、±0.3%固定閾値",
  },

  // 1日足プリセット
  "1d_4bars": {
    name: "1d_4bars",
    timeframe: "1d",
    horizonN: 4,
    threshold: 0.005,
    thresholdMethod: "FIXED",
    description: "1日足、4本先（4日先）、±0.5%固定閾値",
  },
  "1d_7bars": {
    name: "1d_7bars",
    timeframe: "1d",
    horizonN: 7,
    threshold: 0.008,
    thresholdMethod: "FIXED",
    description: "1日足、7本先（1週間先）、±0.8%固定閾値",
  },

  // 動的閾値プリセット（推奨）
  "4h_4bars_dynamic": {
    name: "4h_4bars_dynamic",
    timeframe: "4h",
    horizonN: 4,
    threshold: 0.002,
    thresholdMethod: "KBINS_DISCRETIZER",
    description: "4時間足、4本先、動的閾値（KBinsDiscretizer）",
  },
  "1h_4bars_dynamic": {
    name: "1h_4bars_dynamic",
    timeframe: "1h",
    horizonN: 4,
    threshold: 0.002,
    thresholdMethod: "KBINS_DISCRETIZER",
    description: "1時間足、4本先、動的閾値（KBinsDiscretizer）",
  },
};

/**
 * デフォルトのラベル生成プリセット名
 */
export const DEFAULT_LABEL_PRESET = "4h_4bars";

/**
 * デフォルトの特徴量プロファイル（削除予定）
 * 
 * @deprecated プロファイル概念は不要になりました
 */
export const DEFAULT_FEATURE_PROFILE: FeatureProfile = "research";

/**
 * プリセットのカテゴリ分け
 */
export const PRESET_CATEGORIES = {
  "15分足": ["15m_4bars", "15m_8bars"],
  "30分足": ["30m_4bars", "30m_8bars"],
  "1時間足": ["1h_4bars", "1h_8bars", "1h_16bars"],
  "4時間足": ["4h_4bars", "4h_6bars"],
  "1日足": ["1d_4bars", "1d_7bars"],
  "動的閾値（推奨）": ["4h_4bars_dynamic", "1h_4bars_dynamic"],
};

/**
 * プリセット名の一覧を取得
 */
export const getPresetNames = (): string[] => {
  return Object.keys(LABEL_PRESETS);
};

/**
 * プリセット情報を取得
 *
 * @param presetName - プリセット名
 * @returns プリセット情報、存在しない場合はundefined
 */
export const getPresetInfo = (
  presetName: string
): LabelPresetInfo | undefined => {
  return LABEL_PRESETS[presetName];
};

/**
 * カテゴリごとのプリセットを取得
 *
 * @returns カテゴリ名をキーとするプリセット情報の配列
 */
export const getPresetsByCategory = (): Record<
  string,
  LabelPresetInfo[]
> => {
  const result: Record<string, LabelPresetInfo[]> = {};

  for (const [category, presetNames] of Object.entries(PRESET_CATEGORIES)) {
    result[category] = presetNames
      .map((name) => LABEL_PRESETS[name])
      .filter(Boolean);
  }

  return result;
};