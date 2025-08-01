import * as React from "react"

/**
 * モバイルデバイス検出用カスタムフック
 *
 * 画面幅が768px未満の場合にモバイルデバイスと判定します。
 * ウィンドウサイズの変更を監視し、リアルタイムで判定結果を更新します。
 *
 * @returns {boolean} モバイルデバイスの場合はtrue、デスクトップの場合はfalse
 */
const MOBILE_BREAKPOINT = 768

/**
 * モバイルデバイス判定フック
 *
 * 現在の画面幅がモバイルデバイスと見なされるかどうかを判定します。
 * 768px未満をモバイルデバイスとして判定し、ウィンドウサイズの変更を監視します。
 *
 * @example
 * ```tsx
 * const isMobile = useIsMobile();
 * if (isMobile) {
 *   // モバイル用のUIを表示
 * } else {
 *   // デスクトップ用のUIを表示
 * }
 * ```
 *
 * @returns {boolean} モバイルデバイスの場合はtrue、デスクトップの場合はfalse
 */
export function useIsMobile() {
  const [isMobile, setIsMobile] = React.useState<boolean | undefined>(undefined)

  React.useEffect(() => {
    const mql = window.matchMedia(`(max-width: ${MOBILE_BREAKPOINT - 1}px)`)
    const onChange = () => {
      setIsMobile(window.innerWidth < MOBILE_BREAKPOINT)
    }
    mql.addEventListener("change", onChange)
    setIsMobile(window.innerWidth < MOBILE_BREAKPOINT)
    return () => mql.removeEventListener("change", onChange)
  }, [])

  return !!isMobile
}
