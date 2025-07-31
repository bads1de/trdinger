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
      // cleanup timers on unmount
      Object.keys(timersRef.current).forEach((k) => clearTimer(k));
      timersRef.current = {};
    };
  }, [clearTimer]);

  return {
    messages,
    setMessage,
    removeMessage,
    clearAllMessages,
    durations: DURATIONS,
  };
};
