import { useCallback } from "react";
import { toast } from "sonner";

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
 * メッセージ管理フック (sonner/toast版)
 *
 * アプリケーション内のメッセージ表示を `sonner/toast` を用いて管理します。
 *
 * @param {UseMessagesOptions} [options] - メッセージ管理オプション
 * @returns {{
 *   messages: MessageMap,
 *   setMessage: (key: string, message: string, duration?: number, type?: 'success' | 'error' | 'info' | 'warning') => void,
 *   removeMessage: (key: string) => void,
 *   clearAllMessages: () => void,
 *   durations: Record<"SHORT" | "MEDIUM" | "LONG", number>
 * }} メッセージ管理関連の状態と操作関数
 */
export const useMessages = (options?: UseMessagesOptions) => {
  const DURATIONS = options?.defaultDurations || DefaultMessageDurations;

  const setMessage = useCallback(
    (
      key: string,
      message: string,
      duration: number = DURATIONS.SHORT,
      type: "success" | "error" | "info" | "warning" = "info"
    ) => {
      const toastOptions = {
        id: key,
        duration: duration > 0 ? duration : Infinity,
      };

      switch (type) {
        case "success":
          toast.success(message, toastOptions);
          break;
        case "error":
          toast.error(message, toastOptions);
          break;
        case "warning":
          toast.warning(message, toastOptions);
          break;
        case "info":
        default:
          toast.info(message, toastOptions);
          break;
      }
    },
    [DURATIONS]
  );

  const removeMessage = useCallback((key: string) => {
    toast.dismiss(key);
  }, []);

  const clearAllMessages = useCallback(() => {
    toast.dismiss();
  }, []);

  return {
    /** メッセージのマップ (sonnerに移行したため、常に空) */
    messages: {},
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