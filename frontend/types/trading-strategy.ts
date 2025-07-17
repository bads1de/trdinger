/**
 * トレーディング戦略関連の型定義
 */

/**
 * 売買条件の定義
 *
 * エントリーやエグジットの条件を文字列で表現します。
 *
 * @example
 * ```typescript
 * const entryCondition: TradingCondition = {
 *   condition: 'close > SMA(close, 20)',
 *   description: '終値が20期間移動平均を上回る'
 * }
 * ```
 */
export interface TradingCondition {
  /** 条件式（例: "close > SMA(close, 20)"） */
  condition: string;
  /** 条件の説明（オプション） */
  description?: string;
}

/**
 * トレーディング戦略の定義
 *
 * 完全なトレーディング戦略の設定を表現します。
 * テクニカル指標、エントリー・エグジット条件を含みます。
 */
export interface TradingStrategy {
  /** 戦略の一意識別子（オプション） */
  id?: string;
  /** 戦略名 */
  strategy_name: string;
  /** 対象通貨ペア（例: "BTC/USD"） */
  target_pair: string;
  /** 時間足（例: "1h", "1d"） */
  timeframe: string;
  /** 使用するテクニカル指標のリスト（削除済み） */
  // indicators: TechnicalIndicator[];
  /** エントリー条件のリスト（AND条件） */
  entry_rules: TradingCondition[];
  /** エグジット条件のリスト（OR条件） */
  exit_rules: TradingCondition[];
  /** 作成日時（オプション） */
  created_at?: Date;
  /** 更新日時（オプション） */
  updated_at?: Date;
}
