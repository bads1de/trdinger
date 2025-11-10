/**
 * ML設定の型定義
 *
 * バックエンドのML設定（LabelGenerationConfig、FeatureEngineeringConfig）に
 * 対応するフロントエンドの型定義です。
 *
 * 対応バックエンドファイル:
 * - backend/app/config/unified_config.py
 * - backend/app/utils/label_generation/enums.py
 * - backend/app/utils/label_generation/presets.py
 */

/**
 * 閾値計算方法
 *
 * @see backend/app/utils/label_generation/enums.py:ThresholdMethod
 */
export type ThresholdMethod =
  | "FIXED"
  | "QUANTILE"
  | "PERCENTILE"
  | "STD_DEVIATION"
  | "ADAPTIVE"
  | "DYNAMIC_VOLATILITY"
  | "KBINS_DISCRETIZER";

/**
 * 特徴量プロファイル
 *
 * @see backend/app/config/unified_config.py:FeatureEngineeringConfig
 */
export type FeatureProfile = "research" | "production";

/**
 * サポートされる時間足
 *
 * @see backend/app/utils/label_generation/presets.py:SUPPORTED_TIMEFRAMES
 */
export type SupportedTimeframe = "15m" | "30m" | "1h" | "4h" | "1d";

/**
 * ラベル生成設定
 *
 * 機械学習モデルの学習に使用するラベル（目的変数）の生成方法を設定します。
 * プリセットを使用するか、カスタム設定を使用するかを選択できます。
 *
 * @see backend/app/config/unified_config.py:LabelGenerationConfig (line 367)
 *
 * @example
 * ```typescript
 * const config: LabelGenerationConfig = {
 *   usePreset: true,
 *   defaultPreset: "4h_4bars",
 *   timeframe: "4h",
 *   horizonN: 4,
 *   threshold: 0.002,
 *   priceColumn: "close",
 *   thresholdMethod: "FIXED"
 * };
 * ```
 */
export interface LabelGenerationConfig {
  /**
   * プリセットを使用するか（true）、カスタム設定を使用するか（false）
   *
   * @default true
   */
  usePreset: boolean;

  /**
   * デフォルトのラベル生成プリセット名
   *
   * 例: "4h_4bars", "1h_4bars_dynamic"
   * get_common_presets()で定義されているキーを指定します。
   *
   * @see backend/app/utils/label_generation/presets.py:get_common_presets()
   * @default "4h_4bars"
   */
  defaultPreset: string;

  /**
   * 時間足（カスタム設定時に使用）
   *
   * サポートされている値: "15m", "30m", "1h", "4h", "1d"
   *
   * @default "4h"
   */
  timeframe: SupportedTimeframe;

  /**
   * N本先を見る（カスタム設定時に使用）
   *
   * 例: 4本先を見る場合は4を指定
   *
   * @default 4
   */
  horizonN: number;

  /**
   * 閾値（カスタム設定時に使用）
   *
   * 例: 0.002 = 0.2%
   *
   * @default 0.002
   */
  threshold: number;

  /**
   * 価格カラム名（カスタム設定時に使用）
   *
   * 通常は"close"を使用
   *
   * @default "close"
   */
  priceColumn: string;

  /**
   * 閾値計算方法（カスタム設定時に使用）
   *
   * @see ThresholdMethod
   * @default "FIXED"
   */
  thresholdMethod: ThresholdMethod;
}

/**
 * 特徴量エンジニアリング設定
 *
 * 特徴量生成のプロファイル管理と特徴量選択設定を管理します。
 *
 * @see backend/app/config/unified_config.py:FeatureEngineeringConfig (line 519)
 *
 * @example
 * ```typescript
 * const config: FeatureEngineeringConfig = {
 *   profile: "production",
 *   customAllowlist: ["rsi_14", "macd_signal", "bb_upper"]
 * };
 * ```
 */
export interface FeatureEngineeringConfig {
  /**
   * 特徴量プロファイル
   *
   * - "research": 研究用（全特徴量を使用）
   * - "production": 本番用（選択された特徴量のみ使用）
   *
   * @default "research"
   */
  profile: FeatureProfile;

  /**
   * カスタム特徴量allowlist
   *
   * 指定時はproduction_allowlistを上書きします。
   * nullの場合はデフォルトのallowlistを使用します。
   *
   * @default null
   */
  customAllowlist: string[] | null;
}

/**
 * ラベルプリセット情報
 *
 * 事前定義されたラベル生成設定のメタデータです。
 *
 * @see backend/app/utils/label_generation/presets.py:get_common_presets()
 *
 * @example
 * ```typescript
 * const preset: LabelPresetInfo = {
 *   name: "4h_4bars",
 *   timeframe: "4h",
 *   horizonN: 4,
 *   threshold: 0.002,
 *   thresholdMethod: "FIXED",
 *   description: "4時間足、4本先（16時間先）、0.2%閾値"
 * };
 * ```
 */
export interface LabelPresetInfo {
  /** プリセット名 */
  name: string;

  /** 時間足 */
  timeframe: SupportedTimeframe;

  /** N本先を見る */
  horizonN: number;

  /** 閾値 */
  threshold: number;

  /** 閾値計算方法 */
  thresholdMethod: ThresholdMethod;

  /** プリセットの説明 */
  description: string;
}

/**
 * MLトレーニングリクエスト（拡張版）
 *
 * 新しいラベル生成設定と特徴量プロファイル設定を含む
 * MLトレーニングリクエストの型定義です。
 *
 * @see frontend/hooks/useMLTraining.ts:TrainingConfig
 */
export interface MLTrainingRequestExtended {
  /** 取引シンボル */
  symbol: string;

  /** 時間枠 */
  timeframe: string;

  /** トレーニング開始日 */
  startDate: string;

  /** トレーニング終了日 */
  endDate: string;

  /** モデルを保存するかどうか */
  saveModel: boolean;

  /** トレーニングデータとテストデータの分割比率 */
  trainTestSplit: number;

  /** 乱数シード */
  randomState: number;

  /**
   * ラベル生成設定（オプション）
   *
   * 指定しない場合はバックエンドのデフォルト設定が使用されます。
   */
  labelGeneration?: Partial<LabelGenerationConfig>;

  /**
   * 特徴量プロファイル（オプション）
   *
   * 指定しない場合はバックエンドのデフォルト設定（"research"）が使用されます。
   */
  featureProfile?: FeatureProfile;

  /**
   * カスタム特徴量allowlist（オプション）
   *
   * featureProfileと組み合わせて使用します。
   */
  customAllowlist?: string[] | null;
}