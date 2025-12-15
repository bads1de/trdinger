import { renderHook, act } from "@testing-library/react";
import { useAutoStrategy } from "@/hooks/useAutoStrategy";
import { useApiCall } from "@/hooks/useApiCall";

jest.mock("@/hooks/useApiCall", () => ({
  useApiCall: jest.fn(),
}));

// window.confirm, window.alertをモック
const mockAlert = jest.fn();
Object.defineProperty(window, "alert", { value: mockAlert });

// crypto.randomUUIDをモック
Object.defineProperty(crypto, "randomUUID", {
  value: jest.fn(() => "mock-uuid-1234"),
});

describe("useAutoStrategy", () => {
  const mockExecute = jest.fn();
  const mockLoadResults = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
    (useApiCall as jest.Mock).mockReturnValue({
      execute: mockExecute,
      loading: false,
    });
  });

  const validConfig: any = {
    experiment_name: "Test Experiment",
    base_config: {
      symbol: "BTC/USDT",
      timeframe: "1h",
      start_date: "2023-01-01",
      end_date: "2023-12-31",
      initial_capital: 10000,
      commission_rate: 0.001,
    },
    ga_config: {
      population_size: 50,
      generations: 20,
      crossover_rate: 0.8,
      mutation_rate: 0.2,
      elite_size: 2,
      enable_multi_objective: false,
    },
  };

  it("初期状態が正しいこと", () => {
    const { result } = renderHook(() => useAutoStrategy(mockLoadResults));
    expect(result.current.showAutoStrategyModal).toBe(false);
    expect(result.current.autoStrategyLoading).toBe(false);
  });

  it("モーダルの開閉ができること", () => {
    const { result } = renderHook(() => useAutoStrategy(mockLoadResults));

    act(() => {
      result.current.openAutoStrategyModal();
    });
    expect(result.current.showAutoStrategyModal).toBe(true);

    act(() => {
      result.current.setShowAutoStrategyModal(false);
    });
    expect(result.current.showAutoStrategyModal).toBe(false);
  });

  it("必須フィールドが不足している場合、アラートが表示されAPIが呼ばれないこと（ルートレベル）", async () => {
    const { result } = renderHook(() => useAutoStrategy(mockLoadResults));
    const invalidConfig = { ...validConfig, experiment_name: "" };

    await act(async () => {
      await result.current.handleAutoStrategy(invalidConfig);
    });

    expect(mockAlert).toHaveBeenCalledWith(
      expect.stringContaining("必須フィールドが不足しています")
    );
    expect(mockExecute).not.toHaveBeenCalled();
  });

  it("base_configの必須フィールドが不足している場合、エラーになること", async () => {
    const { result } = renderHook(() => useAutoStrategy(mockLoadResults));
    const invalidConfig = {
      ...validConfig,
      base_config: { ...validConfig.base_config },
    };
    delete invalidConfig.base_config.symbol;

    await act(async () => {
      await result.current.handleAutoStrategy(invalidConfig);
    });

    expect(mockAlert).toHaveBeenCalledWith(
      expect.stringContaining("base_configに必須フィールドがありません: symbol")
    );
    expect(mockExecute).not.toHaveBeenCalled();
  });

  it("ga_configの必須フィールドが不足している場合、エラーになること", async () => {
    const { result } = renderHook(() => useAutoStrategy(mockLoadResults));
    const invalidConfig = {
      ...validConfig,
      ga_config: { ...validConfig.ga_config },
    };
    delete invalidConfig.ga_config.population_size;

    await act(async () => {
      await result.current.handleAutoStrategy(invalidConfig);
    });

    expect(mockAlert).toHaveBeenCalledWith(
      expect.stringContaining(
        "ga_configに必須フィールドがありません: population_size"
      )
    );
    expect(mockExecute).not.toHaveBeenCalled();
  });

  it("正常な設定の場合、APIが呼び出され、成功時に適切な処理が行われること", async () => {
    const { result } = renderHook(() => useAutoStrategy(mockLoadResults));

    mockExecute.mockImplementation(async (_, options) => {
      options.onSuccess({});
      return {};
    });

    act(() => {
      result.current.openAutoStrategyModal();
    });

    await act(async () => {
      await result.current.handleAutoStrategy(validConfig);
    });

    // API呼び出しの確認（experiment_idが含まれているか）
    expect(mockExecute).toHaveBeenCalledWith(
      expect.stringContaining("/api/auto-strategy/generate"),
      expect.objectContaining({
        method: "POST",
        body: expect.objectContaining({
          experiment_name: validConfig.experiment_name,
          base_config: validConfig.base_config,
          ga_config: validConfig.ga_config,
          experiment_id: "mock-uuid-1234",
        }),
      })
    );

    // 成功後の処理確認
    expect(result.current.showAutoStrategyModal).toBe(false);
    expect(mockAlert).toHaveBeenCalledWith(
      expect.stringContaining("戦略生成を開始しました")
    );
    expect(mockLoadResults).toHaveBeenCalled();
  });

  it("多目的最適化の場合、アラートメッセージが変化すること", async () => {
    const { result } = renderHook(() => useAutoStrategy(mockLoadResults));
    const multiObjConfig = {
      ...validConfig,
      ga_config: { ...validConfig.ga_config, enable_multi_objective: true },
    };

    mockExecute.mockImplementation(async (_, options) => {
      options.onSuccess({});
      return {};
    });

    await act(async () => {
      await result.current.handleAutoStrategy(multiObjConfig);
    });

    expect(mockAlert).toHaveBeenCalledWith(
      expect.stringContaining("多目的最適化GA戦略生成を開始しました")
    );
  });

  it("APIエラーの場合、アラートが表示されること", async () => {
    const { result } = renderHook(() => useAutoStrategy(mockLoadResults));

    mockExecute.mockImplementation(async (_, options) => {
      options.onError("API Error");
      return {};
    });

    await act(async () => {
      await result.current.handleAutoStrategy(validConfig);
    });

    expect(mockAlert).toHaveBeenCalledWith(
      expect.stringContaining(
        "オートストラテジーの生成に失敗しました: API Error"
      )
    );
  });
});
