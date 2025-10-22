import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

/**
 * Tailwind CSSクラス名のユーティリティ関数
 * clsxとtailwind-mergeを組み合わせ、クラス名の条件付き結合と競合解決を行う
 * @param inputs - 結合するクラス名（文字列、配列、オブジェクト）
 * @returns 結合および最適化されたクラス名文字列
 */
export function cn(...inputs: ClassValue[]) {
   return twMerge(clsx(inputs))
}
