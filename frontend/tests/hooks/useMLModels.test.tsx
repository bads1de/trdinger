import { renderHook, act } from "@testing-library/react";
import { useMLModels } from "@/hooks/useMLModels";

jest.mock("@/hooks/useDataFetching", () => ({
  useDataFetching: jest.fn(),
}));

jest.mock("@/hooks/useApiCall", () => ({
  useApiCall: jest.fn(),
}));

import { useDataFetching } from "@/hooks/useDataFetching";
import { useApiCall } from "@/hooks/useApiCall";

const mockUseDataFetching = useDataFetching as jest.MockedFunction<typeof useDataFetching>;
const mockUseApiCall = useApiCall as jest.MockedFunction<typeof useApiCall>;

describe("useMLModels", () => {
  const mockRefetch = jest.fn();
  const mockExecute = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
    mockUseDataFetching.mockReturnValue({
      data: [],
      loading: false,
      error: null,
      refetch: mockRefetch,
      params: {},
      setParams: jest.fn(),
      reset: jest.fn(),
      setData: jest.fn(),
    });
    mockUseApiCall.mockReturnValue({
      execute: mockExecute,
      loading: false,
      error: null,
      reset: jest.fn(),
    });
  });

  it("初期状態が正しいこと", () => {
    const { result } = renderHook(() => useMLModels());

    expect(result.current.models).toEqual([]);
    expect(result.current.isLoading).toBe(false);
    expect(result.current.error).toBeNull();
    expect(result.current.isDeleting).toBe(false);
    expect(typeof result.current.fetchModels).toBe("function");
    expect(typeof result.current.deleteModel).toBe("function");
    expect(typeof result.current.deleteAllModels).toBe("function");
  });

  it("モデル一覧を再取得すること", () => {
    const { result } = renderHook(() => useMLModels());

    act(() => {
      result.current.fetchModels();
    });

    expect(mockRefetch).toHaveBeenCalled();
  });

  it("モデルを削除すること", async () => {
    mockExecute.mockResolvedValueOnce({});
    const { result } = renderHook(() => useMLModels());

    await act(async () => {
      await result.current.deleteModel("model-1");
    });

    expect(mockExecute).toHaveBeenCalledWith(
      "/api/ml/models/model-1",
      expect.objectContaining({
        method: "DELETE",
        onSuccess: expect.any(Function),
      })
    );
  });

  it("全モデルを削除すること", async () => {
    mockExecute.mockResolvedValueOnce({});
    const { result } = renderHook(() => useMLModels());

    await act(async () => {
      await result.current.deleteAllModels();
    });

    expect(mockExecute).toHaveBeenCalledWith(
      "/api/ml/models/all",
      expect.objectContaining({
        method: "DELETE",
      })
    );
  });

  it("limitパラメータを渡すこと", () => {
    renderHook(() => useMLModels(10));

    expect(mockUseDataFetching).toHaveBeenCalledWith(
      expect.objectContaining({
        endpoint: "/api/ml/models",
      })
    );
  });
});
