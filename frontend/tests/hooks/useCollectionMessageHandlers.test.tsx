import { renderHook, act } from "@testing-library/react";
import { useCollectionMessageHandlers } from "@/hooks/useCollectionMessageHandlers";

describe("useCollectionMessageHandlers", () => {
  const MESSAGE_KEYS = {
    BULK_COLLECTION: "bulk_collection",
    FUNDING_RATE_COLLECTION: "funding_rate_collection",
    OPEN_INTEREST_COLLECTION: "open_interest_collection",
    ALL_DATA_COLLECTION: "all_data_collection",
  };

  const MESSAGE_DURATION = {
    SHORT: 2000,
    MEDIUM: 5000,
    LONG: 10000,
  };

  const createDefaultDeps = (overrides = {}) => ({
    setMessage: jest.fn(),
    fetchDataStatus: jest.fn(),
    fetchOHLCVData: jest.fn(),
    fetchFundingRateData: jest.fn(),
    fetchOpenInterestData: jest.fn(),
    MESSAGE_KEYS,
    MESSAGE_DURATION,
    ...overrides,
  });

  it("handleCollectionStartがbulkメッセージを正しく生成すること", () => {
    const setMessage = jest.fn();
    const deps = createDefaultDeps({ setMessage });
    const { result } = renderHook(() => useCollectionMessageHandlers(deps));

    const resultData = {
      message: "BULK収集完了",
      actual_tasks: 5,
    };

    act(() => {
      result.current.handleCollectionStart(
        MESSAGE_KEYS.BULK_COLLECTION,
        "bulk",
        resultData,
        MESSAGE_DURATION.MEDIUM
      );
    });

    expect(setMessage).toHaveBeenCalledWith(
      MESSAGE_KEYS.BULK_COLLECTION,
      "🚀 BULK収集完了 (5タスク)",
      MESSAGE_DURATION.MEDIUM,
      "success"
    );
  });

  it("handleCollectionStartがfundingメッセージ（複数シンボル）を正しく生成すること", () => {
    const setMessage = jest.fn();
    const deps = createDefaultDeps({ setMessage });
    const { result } = renderHook(() => useCollectionMessageHandlers(deps));

    const resultData = {
      message: "FR収集完了",
      total_symbols: 10,
      successful_symbols: 8,
    };

    act(() => {
      result.current.handleCollectionStart(
        MESSAGE_KEYS.FUNDING_RATE_COLLECTION,
        "funding",
        resultData
      );
    });

    expect(setMessage).toHaveBeenCalledWith(
      MESSAGE_KEYS.FUNDING_RATE_COLLECTION,
      "🚀 FR収集完了 (8/10シンボル成功)",
      undefined,
      "success"
    );
  });

  it("handleCollectionStartがfundingメッセージ（単一シンボル）を正しく生成すること", () => {
    const setMessage = jest.fn();
    const deps = createDefaultDeps({ setMessage });
    const { result } = renderHook(() => useCollectionMessageHandlers(deps));

    const resultData = {
      symbol: "BTC/USDT:USDT",
      message: "FR収集完了",
      saved_count: 500,
    };

    act(() => {
      result.current.handleCollectionStart(
        MESSAGE_KEYS.FUNDING_RATE_COLLECTION,
        "funding",
        resultData
      );
    });

    expect(setMessage).toHaveBeenCalledWith(
      MESSAGE_KEYS.FUNDING_RATE_COLLECTION,
      "🚀 BTC/USDT:USDTのFRデータ収集完了 (500件保存)",
      undefined,
      "success"
    );
  });

  it("handleCollectionStartがopeninterestメッセージ（複数シンボル）を正しく生成すること", () => {
    const setMessage = jest.fn();
    const deps = createDefaultDeps({ setMessage });
    const { result } = renderHook(() => useCollectionMessageHandlers(deps));

    const resultData = {
      message: "OI収集完了",
      total_symbols: 5,
      successful_symbols: 5,
    };

    act(() => {
      result.current.handleCollectionStart(
        MESSAGE_KEYS.OPEN_INTEREST_COLLECTION,
        "openinterest",
        resultData
      );
    });

    expect(setMessage).toHaveBeenCalledWith(
      MESSAGE_KEYS.OPEN_INTEREST_COLLECTION,
      "🚀 OI収集完了 (5/5シンボル成功)",
      undefined,
      "success"
    );
  });

  it("handleCollectionStartがopeninterestメッセージ（単一シンボル）を正しく生成すること", () => {
    const setMessage = jest.fn();
    const deps = createDefaultDeps({ setMessage });
    const { result } = renderHook(() => useCollectionMessageHandlers(deps));

    const resultData = {
      symbol: "ETH/USDT:USDT",
      message: "OI収集完了",
      saved_count: 300,
    };

    act(() => {
      result.current.handleCollectionStart(
        MESSAGE_KEYS.OPEN_INTEREST_COLLECTION,
        "openinterest",
        resultData
      );
    });

    expect(setMessage).toHaveBeenCalledWith(
      MESSAGE_KEYS.OPEN_INTEREST_COLLECTION,
      "🚀 ETH/USDT:USDTのOIデータ収集完了 (300件保存)",
      undefined,
      "success"
    );
  });

  it("handleCollectionStartがalldataメッセージ（完了時）を正しく生成すること", () => {
    const setMessage = jest.fn();
    const deps = createDefaultDeps({ setMessage });
    const { result } = renderHook(() => useCollectionMessageHandlers(deps));

    const resultData = {
      ohlcv_result: {
        status: "completed",
        actual_tasks: 3,
      },
      funding_rate_result: {
        total_saved_records: 150,
      },
      open_interest_result: {
        total_saved_records: 75,
      },
    };

    act(() => {
      result.current.handleCollectionStart(
        MESSAGE_KEYS.ALL_DATA_COLLECTION,
        "alldata",
        resultData,
        MESSAGE_DURATION.MEDIUM
      );
    });

    expect(setMessage).toHaveBeenCalledWith(
      MESSAGE_KEYS.ALL_DATA_COLLECTION,
      "🚀 全データ収集完了！ OHLCV:3タスク, FR:150件, OI:75件, TI:自動計算済み",
      MESSAGE_DURATION.MEDIUM,
      "success"
    );
  });

  it("handleCollectionStartがalldataメッセージ（処理中）を正しく生成すること", () => {
    const setMessage = jest.fn();
    const deps = createDefaultDeps({ setMessage });
    const { result } = renderHook(() => useCollectionMessageHandlers(deps));

    const resultData = {
      ohlcv_result: {
        status: "processing",
        message: "データ収集中...",
      },
    };

    act(() => {
      result.current.handleCollectionStart(
        MESSAGE_KEYS.ALL_DATA_COLLECTION,
        "alldata",
        resultData
      );
    });

    expect(setMessage).toHaveBeenCalledWith(
      MESSAGE_KEYS.ALL_DATA_COLLECTION,
      expect.stringContaining("🔄"),
      undefined,
      "info"
    );
  });

  it("handleCollectionStartがデフォルトメッセージを正しく生成すること", () => {
    const setMessage = jest.fn();
    const deps = createDefaultDeps({ setMessage });
    const { result } = renderHook(() => useCollectionMessageHandlers(deps));

    const resultData = { message: "未知の処理完了" };

    act(() => {
      result.current.handleCollectionStart(
        "custom_key",
        "unknown_type",
        resultData
      );
    });

    expect(setMessage).toHaveBeenCalledWith(
      "custom_key",
      "🚀 未知の処理完了",
      undefined,
      "success"
    );
  });

  it("handleCollectionStartがonSuccessコールバックを呼ぶこと", () => {
    const setMessage = jest.fn();
    const onSuccess = jest.fn();
    const deps = createDefaultDeps({ setMessage });
    const { result } = renderHook(() => useCollectionMessageHandlers(deps));

    const resultData = { message: "完了", actual_tasks: 1 };

    act(() => {
      result.current.handleCollectionStart(
        MESSAGE_KEYS.BULK_COLLECTION,
        "bulk",
        resultData,
        undefined,
        onSuccess
      );
    });

    expect(onSuccess).toHaveBeenCalledWith(resultData);
  });

  it("handleCollectionErrorがエラーメッセージを正しく設定すること", () => {
    const setMessage = jest.fn();
    const deps = createDefaultDeps({ setMessage });
    const { result } = renderHook(() => useCollectionMessageHandlers(deps));

    act(() => {
      result.current.handleCollectionError(
        MESSAGE_KEYS.BULK_COLLECTION,
        "収集に失敗しました",
        MESSAGE_DURATION.SHORT
      );
    });

    expect(setMessage).toHaveBeenCalledWith(
      MESSAGE_KEYS.BULK_COLLECTION,
      "収集に失敗しました",
      MESSAGE_DURATION.SHORT,
      "error"
    );
  });

  it("collectionHandlersに正しい構造が含まれていること", () => {
    const deps = createDefaultDeps();
    const { result } = renderHook(() => useCollectionMessageHandlers(deps));

    expect(result.current.collectionHandlers).toHaveProperty("bulk");
    expect(result.current.collectionHandlers).toHaveProperty("funding");
    expect(result.current.collectionHandlers).toHaveProperty("openinterest");
    expect(result.current.collectionHandlers).toHaveProperty("alldata");
  });

  it("collectionHandlers.bulkのonSuccessがfetchDataStatusを呼ぶこと", () => {
    const fetchDataStatus = jest.fn();
    const deps = createDefaultDeps({ fetchDataStatus });
    const { result } = renderHook(() => useCollectionMessageHandlers(deps));

    result.current.collectionHandlers.bulk.onSuccess();
    expect(fetchDataStatus).toHaveBeenCalledTimes(1);
  });

  it("handleCollectionErrorがデフォルトdurationで呼び出せること", () => {
    const setMessage = jest.fn();
    const deps = createDefaultDeps({ setMessage });
    const { result } = renderHook(() => useCollectionMessageHandlers(deps));

    act(() => {
      result.current.handleCollectionError(
        MESSAGE_KEYS.BULK_COLLECTION,
        "エラー"
      );
    });

    expect(setMessage).toHaveBeenCalledWith(
      MESSAGE_KEYS.BULK_COLLECTION,
      "エラー",
      undefined,
      "error"
    );
  });
});
