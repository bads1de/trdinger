/**
 * 市場データ関連の型定義
 */

/**
 * 価格データ（OHLCV）
 *
 * 仮想通貨の価格データを表現します。
 * バックテストの入力データとして使用されます。
 */
export interface PriceData {
  /** タイムスタンプ（ISO形式） */
  timestamp: string;
  /** 始値（USD） */
  open: number;
  /** 高値（USD） */
  high: number;
  /** 安値（USD） */
  low: number;
  /** 終値（USD） */
  close: number;
  /** 出来高 */
  volume: number;
}

/**
 * 時間軸の定義
 *
 * データ表示で使用可能な時間軸を定義します。
 */
export type TimeFrame = "15m" | "30m" | "1h" | "4h" | "1d";

/**
 * 時間軸の表示情報
 *
 * 時間軸の表示名と説明を含みます。
 */
export interface TimeFrameInfo {
  /** 時間軸の値 */
  value: TimeFrame;
  /** 表示名 */
  label: string;
  /** 説明 */
  description: string;
}

/**
 * OHLCVデータのAPIレスポンス
 *
 * APIから返されるOHLCVデータの形式を定義します。
 */
export interface OHLCVResponse {
  /** 成功フラグ */
  success: boolean;
  /** データ */
  data: {
    /** 通貨ペア */
    symbol: string;
    /** 時間軸 */
    timeframe: TimeFrame;
    /** OHLCVデータの配列 */
    ohlcv: PriceData[];
  };
  /** メッセージ */
  message?: string;
  /** タイムスタンプ */
  timestamp: string;
}

/**
 * 利用可能な通貨ペア
 *
 * システムで取引可能な通貨ペアの情報を定義します。
 */
export interface TradingPair {
  /** 通貨ペアのシンボル（例: "BTC/USD"） */
  symbol: string;
  /** 表示名（例: "Bitcoin / US Dollar"） */
  name: string;
  /** ベース通貨（例: "BTC"） */
  base: string;
  /** クォート通貨（例: "USD"） */
  quote: string;
}
