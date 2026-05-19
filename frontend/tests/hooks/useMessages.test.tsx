import { renderHook, act } from "@testing-library/react";
import { useMessages, DefaultMessageDurations } from "@/hooks/useMessages";
import { toast } from "sonner";

jest.mock("sonner", () => ({
  toast: {
    success: jest.fn(),
    error: jest.fn(),
    warning: jest.fn(),
    info: jest.fn(),
    dismiss: jest.fn(),
  },
}));

describe("useMessages", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it("初期状態が正しいこと", () => {
    const { result } = renderHook(() => useMessages());

    expect(result.current.messages).toEqual({});
    expect(typeof result.current.setMessage).toBe("function");
    expect(typeof result.current.removeMessage).toBe("function");
    expect(typeof result.current.clearAllMessages).toBe("function");
    expect(result.current.durations).toEqual(DefaultMessageDurations);
  });

  it("successメッセージを表示すること", () => {
    const { result } = renderHook(() => useMessages());

    act(() => {
      result.current.setMessage("test", "成功しました", 5000, "success");
    });

    expect(toast.success).toHaveBeenCalledWith("成功しました", {
      id: "test",
      duration: 5000,
    });
  });

  it("errorメッセージを表示すること", () => {
    const { result } = renderHook(() => useMessages());

    act(() => {
      result.current.setMessage("test", "エラーが発生しました", 5000, "error");
    });

    expect(toast.error).toHaveBeenCalledWith("エラーが発生しました", {
      id: "test",
      duration: 5000,
    });
  });

  it("warningメッセージを表示すること", () => {
    const { result } = renderHook(() => useMessages());

    act(() => {
      result.current.setMessage("test", "警告です", 5000, "warning");
    });

    expect(toast.warning).toHaveBeenCalledWith("警告です", {
      id: "test",
      duration: 5000,
    });
  });

  it("infoメッセージを表示すること（デフォルト）", () => {
    const { result } = renderHook(() => useMessages());

    act(() => {
      result.current.setMessage("test", "情報です");
    });

    expect(toast.info).toHaveBeenCalledWith("情報です", {
      id: "test",
      duration: DefaultMessageDurations.SHORT,
    });
  });

  it("メッセージを削除すること", () => {
    const { result } = renderHook(() => useMessages());

    act(() => {
      result.current.removeMessage("test");
    });

    expect(toast.dismiss).toHaveBeenCalledWith("test");
  });

  it("全メッセージをクリアすること", () => {
    const { result } = renderHook(() => useMessages());

    act(() => {
      result.current.clearAllMessages();
    });

    expect(toast.dismiss).toHaveBeenCalledWith();
  });

  it("カスタムデュレーションを適用すること", () => {
    const customDurations = { SHORT: 5000, MEDIUM: 10000, LONG: 15000 };
    const { result } = renderHook(() =>
      useMessages({ defaultDurations: customDurations })
    );

    expect(result.current.durations).toEqual(customDurations);
  });
});
