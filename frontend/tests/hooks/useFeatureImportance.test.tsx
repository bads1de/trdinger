import { renderHook, act } from "@testing-library/react";
import { useFeatureImportance } from "@/hooks/useFeatureImportance";

jest.mock("@/hooks/useDataFetching", () => ({
  useDataFetching: jest.fn(),
}));

jest.mock("@/utils/colorUtils", () => ({
  getBarColor: jest.fn().mockReturnValue("hsl(200, 70%, 50%)"),
}));

import { useDataFetching } from "@/hooks/useDataFetching";

const mockUseDataFetching = useDataFetching as jest.MockedFunction<
  typeof useDataFetching
>;

describe("useFeatureImportance", () => {
  const mockRefetch = jest.fn();
  const mockSetParams = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
    jest.useFakeTimers();
    mockUseDataFetching.mockReturnValue({
      data: [],
      loading: false,
      error: null,
      params: {},
      setParams: mockSetParams,
      setData: jest.fn(),
      refetch: mockRefetch,
      reset: jest.fn(),
    } as any);
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it("初期状態が正しいこと", () => {
    const { result } = renderHook(() => useFeatureImportance(20));

    expect(result.current.data).toEqual([]);
    expect(result.current.chartData).toEqual([]);
    expect(result.current.loading).toBe(false);
    expect(result.current.error).toBeNull();
    expect(result.current.displayCount).toBe(20);
    expect(result.current.sortOrder).toBe("desc");
  });

  it("useDataFetchingが正しいパラメータで呼ばれること", () => {
    renderHook(() => useFeatureImportance(20));

    expect(mockUseDataFetching).toHaveBeenCalledWith({
      endpoint: "/api/ml/feature-importance",
      initialParams: { top_n: 20 },
      transform: expect.any(Function),
      dependencies: [20, "desc"],
      errorMessage: "特徴量重要度の取得中にエラーが発生しました",
    });
  });

  it("transform関数がレスポンスを正しく変換すること（降順）", () => {
    const response = {
      feature_importance: {
        feature_a: 0.5,
        feature_b: 0.8,
        feature_c: 0.3,
      },
    };

    mockUseDataFetching.mockImplementation((options: any) => {
      const transformed = options.transform(response);
      return {
        data: transformed,
        loading: false,
        error: null,
        params: {},
        setParams: mockSetParams,
        setData: jest.fn(),
        refetch: mockRefetch,
        reset: jest.fn(),
      };
    });

    const { result } = renderHook(() => useFeatureImportance(20));

    expect(result.current.data).toHaveLength(3);
    // rankはtransform内でソート前に割り当てられるため、元のindex+1となる
    // feature_bは元々2番目(index=1)なのでrank=2
    expect(result.current.data[0]).toEqual({
      feature_name: "feature_b",
      importance: 0.8,
      rank: 2,
    });
    // feature_aは元々最初(index=0)なのでrank=1
    expect(result.current.data[1]).toEqual({
      feature_name: "feature_a",
      importance: 0.5,
      rank: 1,
    });
    expect(result.current.data[2]).toEqual({
      feature_name: "feature_c",
      importance: 0.3,
      rank: 3,
    });
  });

  it("昇順ソート済みデータが正しく表示されること", () => {
    const response = {
      feature_importance: {
        feature_a: 0.5,
        feature_b: 0.8,
        feature_c: 0.3,
      },
    };

    mockUseDataFetching.mockImplementation((options: any) => {
      // sortOrderが"asc"の時の変換をシミュレート
      return {
        data: Object.entries(response.feature_importance)
          .map(([feature_name, importance], index) => ({
            feature_name,
            importance: Number(importance),
            rank: index + 1,
          }))
          .sort((a, b) => a.importance - b.importance),
        loading: false,
        error: null,
        params: { top_n: 20 },
        setParams: mockSetParams,
        setData: jest.fn(),
        refetch: mockRefetch,
        reset: jest.fn(),
      };
    });

    const { result } = renderHook(() => useFeatureImportance(20));

    expect(result.current.data[0].feature_name).toBe("feature_c");
    expect(result.current.data[2].feature_name).toBe("feature_b");
  });

  it("displayCountを変更するとsetParamsが呼ばれること", () => {
    const { result } = renderHook(() => useFeatureImportance(20));

    act(() => {
      result.current.setDisplayCount(50);
    });

    expect(mockSetParams).toHaveBeenCalledWith({ top_n: 50 });
  });

  it("sortOrderを変更できること", () => {
    const { result } = renderHook(() => useFeatureImportance(20));

    act(() => {
      result.current.setSortOrder("asc");
    });

    expect(result.current.sortOrder).toBe("asc");
  });

  it("chartDataが正しく計算されること", () => {
    const mockData = [
      { feature_name: "feature_a", importance: 0.5, rank: 1 },
      { feature_name: "feature_b", importance: 0.8, rank: 2 },
      { feature_name: "feature_c", importance: 0.3, rank: 3 },
    ];

    mockUseDataFetching.mockReturnValue({
      data: mockData,
      loading: false,
      error: null,
      params: { top_n: 20 },
      setParams: mockSetParams,
      setData: jest.fn(),
      refetch: mockRefetch,
      reset: jest.fn(),
    } as any);

    const { result } = renderHook(() => useFeatureImportance(20));

    expect(result.current.chartData).toHaveLength(3);

    // feature_bが最大値（0.8）なので、それが100%になる
    const maxItem = result.current.chartData.find(
      (d: any) => d.feature_name === "feature_b"
    );
    expect(maxItem.importancePercent).toBe("100.00");
    expect(maxItem.normalizedImportance).toBe(1);

    // feature_aは相対的に 0.5/0.8 = 62.5%
    const midItem = result.current.chartData.find(
      (d: any) => d.feature_name === "feature_a"
    );
    expect(midItem.importancePercent).toBe("62.50");
    expect(midItem.normalizedImportance).toBe(0.625);
  });

  it("chartDataのabsoluteImportancePercentが正しいこと", () => {
    const mockData = [
      { feature_name: "feature_a", importance: 0.5, rank: 1 },
      { feature_name: "feature_b", importance: 0.8, rank: 2 },
      { feature_name: "feature_c", importance: 0.3, rank: 3 },
    ];
    // total = 0.5 + 0.8 + 0.3 = 1.6

    mockUseDataFetching.mockReturnValue({
      data: mockData,
      loading: false,
      error: null,
      params: { top_n: 20 },
      setParams: mockSetParams,
      setData: jest.fn(),
      refetch: mockRefetch,
      reset: jest.fn(),
    } as any);

    const { result } = renderHook(() => useFeatureImportance(20));

    // feature_a: 0.5/1.6 = 31.25%
    const item = result.current.chartData.find(
      (d: any) => d.feature_name === "feature_a"
    );
    expect(item.absoluteImportancePercent).toBe("31.25");
  });

  it("長いfeature_nameがshortNameで切り詰められること", () => {
    const mockData = [
      {
        feature_name: "this_is_a_very_long_feature_name",
        importance: 0.5,
        rank: 1,
      },
    ];

    mockUseDataFetching.mockReturnValue({
      data: mockData,
      loading: false,
      error: null,
      params: { top_n: 20 },
      setParams: mockSetParams,
      setData: jest.fn(),
      refetch: mockRefetch,
      reset: jest.fn(),
    } as any);

    const { result } = renderHook(() => useFeatureImportance(20));

    // substring(0, 12) + "..."
    expect(result.current.chartData[0].shortName).toBe(
      "this_is_a_ve..."
    );
  });

  it("短いfeature_nameはそのまま表示されること", () => {
    const mockData = [
      { feature_name: "short_name", importance: 0.5, rank: 1 },
    ];

    mockUseDataFetching.mockReturnValue({
      data: mockData,
      loading: false,
      error: null,
      params: { top_n: 20 },
      setParams: mockSetParams,
      setData: jest.fn(),
      refetch: mockRefetch,
      reset: jest.fn(),
    } as any);

    const { result } = renderHook(() => useFeatureImportance(20));

    expect(result.current.chartData[0].shortName).toBe("short_name");
  });

  it("空のデータの場合、chartDataが空配列になること", () => {
    mockUseDataFetching.mockReturnValue({
      data: [],
      loading: false,
      error: null,
      params: { top_n: 20 },
      setParams: mockSetParams,
      setData: jest.fn(),
      refetch: mockRefetch,
      reset: jest.fn(),
    } as any);

    const { result } = renderHook(() => useFeatureImportance(20));

    expect(result.current.chartData).toEqual([]);
  });

  it("loadFeatureImportanceが呼び出せること", () => {
    const { result } = renderHook(() => useFeatureImportance(20));

    act(() => {
      result.current.loadFeatureImportance();
    });

    expect(mockRefetch).toHaveBeenCalledTimes(1);
  });

  it("autoRefreshIntervalが指定された場合、自動更新が設定されること", () => {
    renderHook(() => useFeatureImportance(20, 30));

    expect(mockRefetch).not.toHaveBeenCalled();

    act(() => {
      jest.advanceTimersByTime(30000);
    });

    expect(mockRefetch).toHaveBeenCalledTimes(1);

    act(() => {
      jest.advanceTimersByTime(30000);
    });

    expect(mockRefetch).toHaveBeenCalledTimes(2);
  });

  it("autoRefreshIntervalが0の場合、自動更新されないこと", () => {
    renderHook(() => useFeatureImportance(20, 0));

    act(() => {
      jest.advanceTimersByTime(60000);
    });

    expect(mockRefetch).not.toHaveBeenCalled();
  });

  it("autoRefreshIntervalがundefinedの場合、自動更新されないこと", () => {
    renderHook(() => useFeatureImportance(20));

    act(() => {
      jest.advanceTimersByTime(60000);
    });

    expect(mockRefetch).not.toHaveBeenCalled();
  });

  it("getBarColorを返すこと", () => {
    const { result } = renderHook(() => useFeatureImportance(20));
    expect(result.current.getBarColor).toBeDefined();
  });

  it("全てのimportanceが0の場合、percentagesが0になること", () => {
    const mockData = [
      { feature_name: "feature_a", importance: 0, rank: 1 },
      { feature_name: "feature_b", importance: 0, rank: 2 },
    ];

    mockUseDataFetching.mockReturnValue({
      data: mockData,
      loading: false,
      error: null,
      params: { top_n: 20 },
      setParams: mockSetParams,
      setData: jest.fn(),
      refetch: mockRefetch,
      reset: jest.fn(),
    } as any);

    const { result } = renderHook(() => useFeatureImportance(20));

    expect(result.current.chartData[0].importancePercent).toBe("0.00");
    expect(result.current.chartData[0].normalizedImportance).toBe(0);
    expect(result.current.chartData[0].absoluteImportancePercent).toBe("0.00");
  });
});
