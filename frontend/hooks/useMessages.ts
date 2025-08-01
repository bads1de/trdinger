import { useCallback, useRef, useEffect, useState } from "react";

export type MessageMap = Record<string, string>;

export interface UseMessagesOptions {
  defaultDurations?: {
    SHORT: number;
    MEDIUM: number;
    LONG: number;
  };
}

export const DefaultMessageDurations = {
  SHORT: 10000,
  MEDIUM: 15000,
  LONG: 20000,
} as const;

/**
 * メッセージ管理フック
 *
 * アプリケーション内のメッセージ表示を管理します。
 * メッセージの設定、自動削除、クリアなどの機能を提供します。
 * タイマーによる自動消去や、手動でのメッセージ操作をサポートします。
 *
 * @example
 * ```tsx
 * const {
 *   messages,
 *   setMessage,
 *   removeMessage,
 *   clearAllMessages,
 *   durations
 * } = useMessages({
 *   defaultDurations: {
 *     SHORT: 5000,
 *     MEDIUM: 10000,
 *     LONG: 15000
 *   }
 * });
 *
 * // メッセージを設定
 * setMessage('success', '操作が成功しました', durations.MEDIUM);
 *
 * // メッセージを削除
 * removeMessage('success');
 *
 * // 全メッセージをクリア
 * clearAllMessages();
 * ```
 *
 * @param {UseMessagesOptions} [options] - メッセージ管理オプション
 * @returns {{
 *   messages: MessageMap,
 *   setMessage: (key: string, message: string, duration?: number) => void,
 *   removeMessage: (key: string) => void,
 *   clearAllMessages: () => void,
 *   durations: Record<"SHORT" | "MEDIUM" | "LONG", number>
 * }} メッセージ管理関連の状態と操作関数
 */
export const useMessages = (options?: UseMessagesOptions) => {
  const [messages, setMessages] = useState<MessageMap>({});
  const timersRef = useRef<Record<string, number>>({});
  const DURATIONS = options?.defaultDurations || DefaultMessageDurations;

  const clearTimer = useCallback((key: string) => {
    const tid = timersRef.current[key];

    if (tid) {
      clearTimeout(tid);
      delete timersRef.current[key];
    }
  }, []);

  const setMessage = useCallback(
    (key: string, message: string, duration: number = DURATIONS.SHORT) => {
      setMessages((prev) => ({ ...prev, [key]: message }));

      clearTimer(key);

      if (duration > 0) {
        const tid = window.setTimeout(() => {
          setMessages((prev) => {
            const next = { ...prev };
            delete next[key];
            return next;
          });

          delete timersRef.current[key];
        }, duration);

        timersRef.current[key] = tid;
      }
    },
    [DURATIONS.SHORT, clearTimer]
  );

  const removeMessage = useCallback(
    (key: string) => {
      clearTimer(key);
      setMessages((prev) => {
        const next = { ...prev };
        delete next[key];
        return next;
      });
    },
    [clearTimer]
  );

  const clearAllMessages = useCallback(() => {
    Object.keys(timersRef.current).forEach((k) => clearTimer(k));
    timersRef.current = {};
    setMessages({});
  }, [clearTimer]);

  useEffect(() => {
    return () => {
      Object.keys(timersRef.current).forEach((k) => clearTimer(k));
      timersRef.current = {};
    };
  }, [clearTimer]);

  return {
    /** メッセージのマップ */
    messages,
    /** メッセージを設定する関数 */
    setMessage,
    /** メッセージを削除する関数 */
    removeMessage,
    /** 全メッセージをクリアする関数 */
    clearAllMessages,
    /** メッセージ表示期間の設定 */
    durations: DURATIONS,
  };
};
