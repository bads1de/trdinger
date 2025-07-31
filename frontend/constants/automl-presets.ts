// 市場条件の表示名マッピング
export const MARKET_CONDITION_LABELS: Record<string, string> = {
  bull_market: "強気市場",
  bear_market: "弱気市場",
  sideways: "横ばい市場",
  high_volatility: "高ボラティリティ",
  low_volatility: "低ボラティリティ",
};

// 取引戦略の表示名マッピング
export const TRADING_STRATEGY_LABELS: Record<string, string> = {
  scalping: "スキャルピング",
  day_trading: "デイトレード",
  swing_trading: "スイングトレード",
  position_trading: "ポジショントレード",
  arbitrage: "アービトラージ",
};

// データサイズの表示名マッピング
export const DATA_SIZE_LABELS: Record<string, string> = {
  small: "小規模 (< 1000サンプル)",
  medium: "中規模 (1000-10000サンプル)",
  large: "大規模 (> 10000サンプル)",
};
