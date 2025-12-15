import { renderHook, act } from "@testing-library/react";
import { useApiCall } from "@/hooks/useApiCall";

describe("useApiCall", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it("初期状態が正しいこと", () => {
    const { result } = renderHook(() => useApiCall());

    expect(result.current.loading).toBe(false);
    expect(result.current.error).toBeNull();
    expect(typeof result.current.execute).toBe("function");
    expect(typeof result.current.reset).toBe("function");
  });

  it("API呼び出しが成功した場合、loading状態が変化し、結果を返すこと", async () => {
    const mockData = { success: true, data: "test data" };
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      status: 200,
      text: () => Promise.resolve(JSON.stringify(mockData)),
    });

    const { result } = renderHook(() => useApiCall());

    let promise: Promise<any>;
    act(() => {
      promise = result.current.execute("/test-endpoint");
    });

    expect(result.current.loading).toBe(true);

    await act(async () => {
      const response = await promise;
      expect(response).toEqual(mockData);
    });

    expect(result.current.loading).toBe(false);
    expect(result.current.error).toBeNull();
  });

  it("API呼び出しが失敗した場合（ステータス200以外）、エラーメッセージを設定すること", async () => {
    const errorMessage = "Bad Request";
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: false,
      status: 400,
      statusText: "Bad Request",
      text: () => Promise.resolve(JSON.stringify({ message: errorMessage })),
    });

    const { result } = renderHook(() => useApiCall());

    await act(async () => {
      const response = await result.current.execute("/test-endpoint");
      expect(response).toBeNull();
    });

    expect(result.current.loading).toBe(false);
    expect(result.current.error).toBe(errorMessage);
  });

  it("JSONパースエラーが発生した場合、エラーメッセージを設定すること", async () => {
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      status: 200,
      text: () => Promise.resolve("invalid json"),
    });

    const { result } = renderHook(() => useApiCall());

    await act(async () => {
      const response = await result.current.execute("/test-endpoint");
      expect(response).toBeNull();
    });

    expect(result.current.loading).toBe(false);
    expect(result.current.error).toContain("レスポンスが無効なJSON形式です");
  });

  it("422バリデーションエラーの場合、詳細なエラーがログ出力されること", async () => {
    const consoleSpy = jest.spyOn(console, "error").mockImplementation();
    const mockDetailedError = {
      detail: [{ loc: ["body", "field"], msg: "field required" }],
    };

    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: false,
      status: 422,
      statusText: "Unprocessable Entity",
      text: () => Promise.resolve(JSON.stringify(mockDetailedError)),
    });

    const { result } = renderHook(() => useApiCall());

    await act(async () => {
      await result.current.execute("/test-endpoint");
    });

    expect(consoleSpy).toHaveBeenCalledWith(
      "Validation Errors:",
      "body.field - field required"
    );

    consoleSpy.mockRestore();
  });

  it("reset関数を呼び出すと状態が初期化されること", async () => {
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: false,
      status: 500,
      text: () => Promise.resolve(JSON.stringify({ error: "Server Error" })),
    });

    const { result } = renderHook(() => useApiCall());

    await act(async () => {
      await result.current.execute("/test-endpoint");
    });

    expect(result.current.error).toBe("Server Error");

    act(() => {
      result.current.reset();
    });

    expect(result.current.error).toBeNull();
    expect(result.current.loading).toBe(false);
  });

  it("confirmMessageが指定され、キャンセルされた場合、API呼び出しを行わないこと", async () => {
    const confirmSpy = jest.spyOn(window, "confirm").mockReturnValue(false);
    const { result } = renderHook(() => useApiCall());

    await act(async () => {
      await result.current.execute("/test-endpoint", {
        confirmMessage: "本当によろしいですか？",
      });
    });

    expect(confirmSpy).toHaveBeenCalledWith("本当によろしいですか？");
    expect(global.fetch).not.toHaveBeenCalled();

    confirmSpy.mockRestore();
  });

  it("confirmMessageが指定され、OKされた場合、API呼び出しを行うこと", async () => {
    const confirmSpy = jest.spyOn(window, "confirm").mockReturnValue(true);
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      text: () => Promise.resolve(JSON.stringify({})),
    });

    const { result } = renderHook(() => useApiCall());

    await act(async () => {
      await result.current.execute("/test-endpoint", {
        confirmMessage: "本当によろしいですか？",
      });
    });

    expect(confirmSpy).toHaveBeenCalledWith("本当によろしいですか？");
    expect(global.fetch).toHaveBeenCalled();

    confirmSpy.mockRestore();
  });
});
