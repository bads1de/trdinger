export interface LongShortRatioData {
  symbol: string;
  period: string;
  buy_ratio: number;
  sell_ratio: number;
  timestamp: string; // ISO string
  ls_ratio?: number; // 計算値 (buy / sell)
}

export interface LongShortRatioCollectionResponse {
  message: string;
  symbol: string;
}
