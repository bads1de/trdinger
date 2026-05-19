import { renderHook, act } from "@testing-library/react";
import { useModelManagement } from "@/hooks/useModelManagement";

jest.mock("@/hooks/useApiCall", () => ({
  useApiCall: jest.fn(),
}));

import { useApiCall } from "@/hooks/useApiCall";

const mockUseApiCall = useApiCall as jest.MockedFunction<typeof useApiCall>;

describe("useModelManagement", () => {
  const mockFetchData = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
    mockUseApiCall.mockReturnValue({
      execute: mockFetchData,
      loading: false,
      error: null,
      reset: jest.fn(),
    });
  });

  it("初期状態が正しいこと", () => {
    const { result } = renderHook(() => useModelManagement());

    expect(result.current.models).toEqual([]);
    expect(result.current.currentModel).toBeNull();
    expect(result.current.loading).toBe(false);
    expect(result.current.error).toBeNull();
    expect(typeof result.current.fetchModels).toBe("function");
    expect(typeof result.current.fetchCurrentModel).toBe("function");
    expect(typeof result.current.loadModel).toBe("function");
    expect(typeof result.current.refreshModels).toBe("function");
  });

  it("モデル一覧を取得すること", async () => {
    const mockModels = [
      { name: "model1", path: "/path/model1", size_mb: 10, modified_at: "2023-01-01", model_type: "lightgbm", trainer_type: "default", feature_count: 5, has_feature_importance: true, feature_importance_count: 5 },
    ];
    mockFetchData.mockImplementation(async (_url: string, options: any) => {
      options.onSuccess({ models: mockModels });
      options.onFinally();
    });

    const { result } = renderHook(() => useModelManagement());

    await act(async () => {
      await result.current.fetchModels();
    });

    expect(result.current.models).toEqual(mockModels);
    expect(result.current.loading).toBe(false);
  });

  it("現在のモデル情報を取得すること", async () => {
    const mockCurrentModel = { loaded: true, model_type: "lightgbm" };
    mockFetchData.mockImplementation(async (_url: string, options: any) => {
      options.onSuccess(mockCurrentModel);
    });

    const { result } = renderHook(() => useModelManagement());

    await act(async () => {
      await result.current.fetchCurrentModel();
    });

    expect(result.current.currentModel).toEqual(mockCurrentModel);
  });

  it("モデル読み込みが成功した場合、currentModelを更新すること", async () => {
    const mockResponse = { success: true, current_model: { loaded: true } };
    mockFetchData.mockImplementation(async (_url: string, options: any) => {
      options.onSuccess(mockResponse);
      options.onFinally();
    });

    const { result } = renderHook(() => useModelManagement());

    await act(async () => {
      await result.current.loadModel("my_model");
    });

    expect(result.current.currentModel).toEqual({ loaded: true });
  });

  it("モデル読み込みが失敗した場合、エラーを設定すること", async () => {
    mockFetchData.mockImplementation(async (_url: string, options: any) => {
      options.onSuccess({ success: false, error: "読み込み失敗" });
      options.onFinally();
    });

    const { result } = renderHook(() => useModelManagement());

    await act(async () => {
      await result.current.loadModel("my_model");
    });

    expect(result.current.error).toBe("読み込み失敗");
  });

  it("refreshModelsでモデル一覧と現在のモデル情報を同時に更新すること", async () => {
    mockFetchData.mockResolvedValue(undefined);

    const { result } = renderHook(() => useModelManagement());

    await act(async () => {
      await result.current.refreshModels();
    });

    expect(mockFetchData).toHaveBeenCalledTimes(2);
  });
});
