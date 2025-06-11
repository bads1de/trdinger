/**
 * API バリデーション関数
 *
 * BTC/USDT:USDT無期限先物のみをサポートするためのバリデーション機能
 */

import { SUPPORTED_TRADING_PAIRS, SUPPORTED_TIMEFRAMES } from "@/constants";
import { TimeFrame } from "@/types/strategy";

/**
 * サポートされているシンボルかどうかを検証
 *
 * @param symbol 検証するシンボル
 * @returns 有効な場合true、無効な場合false
 */
export function validateSymbol(symbol: string): boolean {
  const supportedSymbols = SUPPORTED_TRADING_PAIRS.map((pair) => pair.symbol);
  return supportedSymbols.includes(symbol);
}

/**
 * サポートされている時間足かどうかを検証
 *
 * @param timeframe 検証する時間足
 * @returns 有効な場合true、無効な場合false
 */
export function validateTimeframe(timeframe: string): timeframe is TimeFrame {
  const supportedTimeframes = SUPPORTED_TIMEFRAMES.map((tf) => tf.value);
  return supportedTimeframes.includes(timeframe as TimeFrame);
}

/**
 * シンボルバリデーションエラーレスポンスを生成
 *
 * @param symbol 無効なシンボル
 * @returns エラーレスポンス
 */
export function createSymbolValidationError(symbol: string) {
  const supportedSymbols = SUPPORTED_TRADING_PAIRS.map((pair) => pair.symbol);

  return {
    success: false,
    message: `サポートされていないシンボルです: ${symbol}. サポートされているシンボル: ${supportedSymbols.join(
      ", "
    )}`,
    error: "UNSUPPORTED_SYMBOL",
    supported_symbols: supportedSymbols,
    timestamp: new Date().toISOString(),
  };
}

/**
 * 時間足バリデーションエラーレスポンスを生成
 *
 * @param timeframe 無効な時間足
 * @returns エラーレスポンス
 */
export function createTimeframeValidationError(timeframe: string) {
  const supportedTimeframes = SUPPORTED_TIMEFRAMES.map((tf) => tf.value);

  return {
    success: false,
    message: `サポートされていない時間足です: ${timeframe}. サポートされている時間足: ${supportedTimeframes.join(
      ", "
    )}`,
    error: "UNSUPPORTED_TIMEFRAME",
    supported_timeframes: supportedTimeframes,
    timestamp: new Date().toISOString(),
  };
}

/**
 * リクエストパラメータを検証し、デフォルト値を適用
 *
 * @param searchParams URLSearchParams
 * @returns 検証済みパラメータまたはエラー
 */
export function validateAndNormalizeParams(searchParams: URLSearchParams) {
  const symbol = searchParams.get("symbol") || "BTC/USDT:USDT";
  const timeframe = searchParams.get("timeframe") || "1h";
  const limit = searchParams.get("limit") || "100";

  // シンボルバリデーション
  if (!validateSymbol(symbol)) {
    return {
      isValid: false,
      error: createSymbolValidationError(symbol),
    };
  }

  // 時間足バリデーション（時間足が指定されている場合のみ）
  if (searchParams.has("timeframe") && !validateTimeframe(timeframe)) {
    return {
      isValid: false,
      error: createTimeframeValidationError(timeframe),
    };
  }

  // 制限値バリデーション
  const limitNum = parseInt(limit, 10);
  if (isNaN(limitNum) || limitNum < 1 || limitNum > 1000) {
    return {
      isValid: false,
      error: {
        success: false,
        message: `無効な制限値です: ${limit}. 1-1000の範囲で指定してください。`,
        error: "INVALID_LIMIT",
        timestamp: new Date().toISOString(),
      },
    };
  }

  return {
    isValid: true,
    params: {
      symbol,
      timeframe,
      limit: limitNum.toString(),
    },
  };
}

/**
 * サポートされているシンボル一覧を取得
 *
 * @returns サポートされているシンボル一覧
 */
export function getSupportedSymbols() {
  return SUPPORTED_TRADING_PAIRS;
}

/**
 * サポートされている時間足一覧を取得
 *
 * @returns サポートされている時間足一覧
 */
export function getSupportedTimeframes() {
  return SUPPORTED_TIMEFRAMES;
}

/**
 * シンボルを正規化（大文字小文字、スペースの処理）
 *
 * @param symbol 正規化するシンボル
 * @returns 正規化されたシンボル
 */
export function normalizeSymbol(symbol: string): string {
  return symbol.trim().toUpperCase();
}

/**
 * 時間足を正規化
 *
 * @param timeframe 正規化する時間足
 * @returns 正規化された時間足
 */
export function normalizeTimeframe(timeframe: string): string {
  return timeframe.trim().toLowerCase();
}

/**
 * APIレスポンスの成功レスポンスを生成
 *
 * @param data レスポンスデータ
 * @param message メッセージ（オプション）
 * @returns 成功レスポンス
 */
export function createSuccessResponse(data: any, message?: string) {
  return {
    success: true,
    data,
    message,
    timestamp: new Date().toISOString(),
  };
}

/**
 * APIレスポンスのエラーレスポンスを生成
 *
 * @param message エラーメッセージ
 * @param error エラーコード
 * @param details 詳細情報（オプション）
 * @returns エラーレスポンス
 */
export function createErrorResponse(
  message: string,
  error: string,
  details?: any
) {
  return {
    success: false,
    message,
    error,
    details,
    timestamp: new Date().toISOString(),
  };
}
