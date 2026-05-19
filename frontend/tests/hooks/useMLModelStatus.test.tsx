import { renderHook, act } from "@testing-library/react";
import { useMLModelStatus } from "@/hooks/useMLModelStatus";

jest.mock("@/hooks/useDataFetching", () => ({
  useDataFetching: jest.fn(),
}));

import { useDataFetching } from "@/hooks/useDataFetching";

const mockUseDataFetching = useDataFetching as jest.MockedFunction<typeof useDataFetching>;

describe("useMLModelStatus", () => {
  const mockRefetchStatus = jest.fn();
  const mockRefetchImportance = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
    let callCount = 0;
    mockUseDataFetching.mockImplementation(() => {
      callCount++;
      if (callCount === 1) {
        return {
          data: [],
          loading: false,
          error: null,
          refetch: mockRefetchStatus,
          params: {},
          setParams: jest.fn(),
          reset: jest.fn(),
          setData: jest.fn(),
        };
      }
      return {
        data: [],
        loading: false,
        error: null,
        refetch: mockRefetchImportance,
        params: {},
        setParams: jest.fn(),
        reset: jest.fn(),
        setData: jest.fn(),
      };
    });
  });

  it("初期状態が正しいこと", () => {
    const { result } = renderHook(() => useMLModelStatus());

    expect(result.current.modelStatus).toBeNull();
    expect(result.current.featureImportance).toEqual({});
    expect(result.current.isLoading).toBe(false);
    expect(result.current.error).toBeNull();
    expect(typeof result.current.fetchModelStatus).toBe("function");
    expect(typeof result.current.fetchFeatureImportance).toBe("function");
  });

  it("モデル状態を再取得すること", () => {
    const { result } = renderHook(() => useMLModelStatus());

    act(() => {
      result.current.fetchModelStatus();
    });

    expect(mockRefetchStatus).toHaveBeenCalled();
  });

  it("特徴量重要度を再取得すること", () => {
    const { result } = renderHook(() => useMLModelStatus());

    act(() => {
      result.current.fetchFeatureImportance();
    });

    expect(mockRefetchImportance).toHaveBeenCalled();
  });

  it("モデル状態データを正しく処理すること", () => {
    const mockStatus = {
      is_model_loaded: true,
      is_trained: true,
      feature_count: 10,
    };
    mockUseDataFetching.mockImplementationOnce(() => ({
      data: [mockStatus],
      loading: false,
      error: null,
      refetch: mockRefetchStatus,
      params: {},
      setParams: jest.fn(),
      reset: jest.fn(),
      setData: jest.fn(),
    }));

    const { result } = renderHook(() => useMLModelStatus());

    expect(result.current.modelStatus).toEqual(mockStatus);
  });

  it("特徴量重要度データを正しく処理すること", () => {
    const mockImportance = { feature_importance: { feature1: 0.5, feature2: 0.3 } };
    let callCount = 0;
    mockUseDataFetching.mockImplementation(() => {
      callCount++;
      if (callCount === 1) {
        return {
          data: [],
          loading: false,
          error: null,
          refetch: mockRefetchStatus,
          params: {},
          setParams: jest.fn(),
          reset: jest.fn(),
          setData: jest.fn(),
        };
      }
      return {
        data: [mockImportance],
        loading: false,
        error: null,
        refetch: mockRefetchImportance,
        params: {},
        setParams: jest.fn(),
        reset: jest.fn(),
        setData: jest.fn(),
      };
    });

    const { result } = renderHook(() => useMLModelStatus());

    expect(result.current.featureImportance).toEqual({ feature1: 0.5, feature2: 0.3 });
  });
});
