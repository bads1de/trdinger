/**
 * シンボル変換ユーティリティ
 * 
 * フロントエンドとバックエンド間でのシンボル形式変換を行います。
 */

/**
 * シンボル変換関数
 * フロントエンド形式（BTC/USDT:USDT）はそのままバックエンドに送信
 * データベースには BTC/USDT:USDT 形式で保存されているため
 * 
 * @param symbol - 変換するシンボル
 * @returns バックエンド用に変換されたシンボル
 */
export function convertSymbolForBackend(symbol: string): string {
  // データベースの形式と一致させるため、そのまま返す
  return symbol;
}

/**
 * バックエンドからのシンボルをフロントエンド表示用に変換
 * 
 * @param symbol - バックエンドからのシンボル
 * @returns フロントエンド表示用のシンボル
 */
export function convertSymbolForFrontend(symbol: string): string {
  // 現在は同じ形式を使用しているため、そのまま返す
  return symbol;
}

/**
 * シンボルを正規化（大文字・空白除去）
 * 
 * @param symbol - 正規化するシンボル
 * @returns 正規化されたシンボル
 */
export function normalizeSymbol(symbol: string): string {
  return symbol.trim().toUpperCase();
}

/**
 * シンボルが有効な形式かチェック
 * 
 * @param symbol - チェックするシンボル
 * @returns 有効な場合true
 */
export function isValidSymbol(symbol: string): boolean {
  // BTC/USDT:USDT 形式をチェック
  const pattern = /^[A-Z]+\/[A-Z]+:[A-Z]+$/;
  return pattern.test(symbol);
}
