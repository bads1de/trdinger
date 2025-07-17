/**
 * データリセットボタンの設定定数
 */
export const RESET_CONFIGS = {
  all: {
    label: "全データリセット",
    endpoint: "/api/data-reset/all",
    confirmMessage:
      "⚠️ 全てのデータ（OHLCV・ファンディングレート・オープンインタレスト）を削除します。\n\n" +
      "この操作は取り消すことができません。\n" +
      "本当に実行しますか？",
    variant: "danger" as const,
    icon: "🗑️",
  },
  ohlcv: {
    label: "OHLCVリセット",
    endpoint: "/api/data-reset/ohlcv",
    confirmMessage:
      "⚠️ 全てのOHLCVデータを削除します。\n\n" +
      "この操作は取り消すことができません。\n" +
      "本当に実行しますか？",
    variant: "warning" as const,
    icon: "📊",
  },
  "funding-rates": {
    label: "FRリセット",
    endpoint: "/api/data-reset/funding-rates",
    confirmMessage:
      "⚠️ 全てのファンディングレートデータを削除します。\n\n" +
      "この操作は取り消すことができません。\n" +
      "本当に実行しますか？",
    variant: "warning" as const,
    icon: "💰",
  },
  "open-interest": {
    label: "OIリセット",
    endpoint: "/api/data-reset/open-interest",
    confirmMessage:
      "⚠️ 全てのオープンインタレストデータを削除します。\n\n" +
      "この操作は取り消すことができません。\n" +
      "本当に実行しますか？",
    variant: "warning" as const,
    icon: "📈",
  },
  symbol: {
    label: "シンボル別リセット",
    endpoint: "/api/data-reset/symbol",
    confirmMessage:
      "⚠️ 指定されたシンボルの全データを削除します。\n\n" +
      "この操作は取り消すことができません。\n" +
      "本当に実行しますか？",
    variant: "warning" as const,
    icon: "🎯",
  },
};
