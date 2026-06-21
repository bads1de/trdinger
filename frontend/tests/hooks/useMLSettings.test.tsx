import { renderHook, act } from "@testing-library/react";
import { useMLSettings, MLConfig } from "@/hooks/useMLSettings";

jest.mock("@/hooks/useApiCall", () => ({
  useApiCall: jest.fn(),
}));

import { useApiCall } from "@/hooks/useApiCall";

const mockUseApiCall = useApiCall as jest.MockedFunction<typeof useApiCall>;

const DEFAULT_ML_CONFIG: MLConfig = {
  data_processing: {
    max_ohlcv_rows: 100000,
    max_feature_rows: 50000,
    feature_calculation_timeout: 300,
    model_training_timeout: 600,
    model_prediction_timeout: 60,
    memory_warning_threshold: 0.8,
    memory_limit_threshold: 0.9,
    debug_mode: false,
    log_level: "INFO",
  },
  model: {
    model_save_path: "/models",
    model_file_extension: ".pkl",
    model_name_prefix: "model_",
    auto_strategy_model_name: "auto_strategy",
    max_model_versions: 5,
    model_retention_days: 30,
  },
  prediction: {
    default_up_prob: 0.4,
    default_down_prob: 0.4,
    default_range_prob: 0.2,
    fallback_up_prob: 0.33,
    fallback_down_prob: 0.34,
    fallback_range_prob: 0.33,
    min_probability: 0.1,
    max_probability: 0.9,
    probability_sum_min: 0.95,
    probability_sum_max: 1.05,
    expand_to_data_length: true,
    default_indicator_length: 14,
  },
  training: {
    train_test_split: 0.8,
    cross_validation_folds: 5,
    prediction_horizon: 4,
    label_method: "triple_barrier",
    volatility_window: 20,
    threshold_multiplier: 1.5,
    min_threshold: 0.001,
    max_threshold: 0.01,
    threshold_up: 0.02,
    threshold_down: 0.01,
    label_generation: {
      usePreset: true,
      defaultPreset: "TBB",
      timeframe: "4h",
      horizonN: 4,
      threshold: 0.002,
      priceColumn: "close",
      thresholdMethod: "FIXED",
    },
    cv_folds: 5,
    random_state: 42,
  },
  feature_engineering: {
    featureAllowlist: null,
  },
  ensemble: {
    enabled: false,
    algorithms: ["lightgbm", "xgboost"],
    voting_method: "soft",
    default_method: "soft",
    stacking_cv_folds: 5,
    stacking_use_probas: true,
  },
  retraining: {
    check_interval_seconds: 3600,
    max_concurrent_jobs: 1,
    job_timeout_seconds: 1800,
    data_retention_days: 90,
    incremental_training_enabled: true,
    performance_degradation_threshold: 0.05,
    data_drift_threshold: 0.1,
  },
};

describe("useMLSettings", () => {
  let mockExecute: jest.Mock;

  beforeEach(() => {
    jest.clearAllMocks();
    jest.useFakeTimers();

    mockExecute = jest.fn();
    mockUseApiCall.mockReturnValue({
      execute: mockExecute,
      loading: false,
      error: null,
      reset: jest.fn(),
    });
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it("初期状態が正しいこと", () => {
    const { result } = renderHook(() => useMLSettings());

    expect(result.current.config).toBeNull();
    expect(result.current.isLoading).toBe(false);
    expect(result.current.isSaving).toBe(false);
    expect(result.current.error).toBeNull();
    expect(result.current.successMessage).toBeNull();
  });

  it("マウント時に自動的に設定を取得すること", () => {
    renderHook(() => useMLSettings());

    expect(mockExecute).toHaveBeenCalledWith(
      "/api/ml/config",
      expect.objectContaining({
        onSuccess: expect.any(Function),
      })
    );
  });

  it("設定取得成功時にconfigが更新されること", () => {
    mockExecute.mockImplementation(
      async (_url: string, options: { onSuccess: (data: MLConfig) => void }) => {
        options.onSuccess(DEFAULT_ML_CONFIG);
      }
    );

    const { result } = renderHook(() => useMLSettings());

    expect(result.current.config).toEqual(
      expect.objectContaining({
        data_processing: DEFAULT_ML_CONFIG.data_processing,
        model: DEFAULT_ML_CONFIG.model,
      })
    );
  });

  it("設定取得成功時にlabel_generationのデフォルト値が補完されること", () => {
    const configWithoutLabelGen: any = {
      ...DEFAULT_ML_CONFIG,
      training: {
        ...DEFAULT_ML_CONFIG.training,
        label_generation: undefined,
      },
    };

    mockExecute.mockImplementation(
      async (_url: string, options: { onSuccess: (data: any) => void }) => {
        options.onSuccess(configWithoutLabelGen);
      }
    );

    const { result } = renderHook(() => useMLSettings());

    expect(result.current.config?.training.label_generation).toEqual({
      usePreset: true,
      defaultPreset: "4h_4bars",
      timeframe: "4h",
      horizonN: 4,
      threshold: 0.002,
      priceColumn: "close",
      thresholdMethod: "FIXED",
    });
  });

  it("設定取得成功時にfeature_engineeringのデフォルト値が補完されること", () => {
    const configWithoutFe: any = {
      ...DEFAULT_ML_CONFIG,
      feature_engineering: undefined,
    };

    mockExecute.mockImplementation(
      async (_url: string, options: { onSuccess: (data: any) => void }) => {
        options.onSuccess(configWithoutFe);
      }
    );

    const { result } = renderHook(() => useMLSettings());

    expect(result.current.config?.feature_engineering).toEqual({
      featureAllowlist: null,
    });
  });

  it("saveConfigが正しく動作すること", async () => {
    mockExecute.mockImplementation(
      async (_url: string, options: { onSuccess: (data: MLConfig) => void }) => {
        options.onSuccess(DEFAULT_ML_CONFIG);
      }
    );

    const { result } = renderHook(() => useMLSettings());

    mockExecute.mockClear();

    mockExecute.mockImplementation(
      async (_url: string, options: { onSuccess: (data: MLConfig) => void }) => {
        options.onSuccess(DEFAULT_ML_CONFIG);
      }
    );

    await act(async () => {
      await result.current.saveConfig(DEFAULT_ML_CONFIG);
    });

    expect(mockExecute).toHaveBeenCalledWith(
      "/api/ml/config",
      expect.objectContaining({
        method: "PUT",
        body: DEFAULT_ML_CONFIG,
      })
    );
  });

  it("saveConfigが成功メッセージを表示すること", async () => {
    mockExecute.mockImplementation(
      async (_url: string, options: { onSuccess: (data: MLConfig) => void }) => {
        options.onSuccess(DEFAULT_ML_CONFIG);
      }
    );

    const { result } = renderHook(() => useMLSettings());

    mockExecute.mockClear();

    mockExecute.mockImplementation(
      async (_url: string, options: { onSuccess: (data: MLConfig) => void }) => {
        options.onSuccess(DEFAULT_ML_CONFIG);
      }
    );

    await act(async () => {
      await result.current.saveConfig(DEFAULT_ML_CONFIG);
    });

    expect(result.current.successMessage).toBe("設定が正常に保存されました");
  });

  it("successMessageが3秒後にクリアされること", async () => {
    mockExecute.mockImplementation(
      async (_url: string, options: { onSuccess: (data: MLConfig) => void }) => {
        options.onSuccess(DEFAULT_ML_CONFIG);
      }
    );

    const { result } = renderHook(() => useMLSettings());

    mockExecute.mockClear();
    mockExecute.mockImplementation(
      async (_url: string, options: { onSuccess: (data: MLConfig) => void }) => {
        options.onSuccess(DEFAULT_ML_CONFIG);
      }
    );

    await act(async () => {
      await result.current.saveConfig(DEFAULT_ML_CONFIG);
    });

    expect(result.current.successMessage).toBe("設定が正常に保存されました");

    act(() => {
      jest.advanceTimersByTime(3000);
    });

    expect(result.current.successMessage).toBeNull();
  });

  it("resetToDefaultsがconfirmMessage付きでPOSTリクエストを送ること", async () => {
    mockExecute.mockImplementation(
      async (_url: string, options: { onSuccess: (data: MLConfig) => void }) => {
        options.onSuccess(DEFAULT_ML_CONFIG);
      }
    );

    const { result } = renderHook(() => useMLSettings());

    mockExecute.mockClear();

    mockExecute.mockImplementation(
      async (_url: string, options: { onSuccess: (data: MLConfig) => void }) => {
        options.onSuccess(DEFAULT_ML_CONFIG);
      }
    );

    await act(async () => {
      await result.current.resetToDefaults();
    });

    expect(mockExecute).toHaveBeenCalledWith(
      "/api/ml/config/reset",
      expect.objectContaining({
        method: "POST",
        confirmMessage: "設定をデフォルト値にリセットしますか？",
      })
    );
  });

  it("cleanupOldModelsがconfirmMessage付きでPOSTリクエストを送ること", async () => {
    mockExecute.mockImplementation(
      async (_url: string, options: { onSuccess: (data: any) => void }) => {
        options.onSuccess({});
      }
    );

    const { result } = renderHook(() => useMLSettings());

    mockExecute.mockClear();

    await act(async () => {
      await result.current.cleanupOldModels();
    });

    expect(mockExecute).toHaveBeenCalledWith(
      "/api/ml/models/cleanup",
      expect.objectContaining({
        method: "POST",
        confirmMessage: "古いモデルファイルを削除しますか？この操作は取り消せません。",
      })
    );
  });

  it("updateConfigがconfigの特定セクションを更新すること", () => {
    mockExecute.mockImplementation(
      async (_url: string, options: { onSuccess: (data: MLConfig) => void }) => {
        options.onSuccess(DEFAULT_ML_CONFIG);
      }
    );

    const { result } = renderHook(() => useMLSettings());

    act(() => {
      result.current.updateConfig(
        "training",
        "prediction_horizon",
        8
      );
    });

    expect(result.current.config?.training.prediction_horizon).toBe(8);
    // 他の設定は変わらないことを確認
    expect(result.current.config?.training.cv_folds).toBe(5);
  });

  it("updateConfigでconfigがnullの場合に何も起こらないこと", () => {
    const { result } = renderHook(() => useMLSettings());

    act(() => {
      // configがnullの状態で呼び出してもエラーにならない
      result.current.updateConfig("training", "cv_folds", 10);
    });

    expect(result.current.config).toBeNull();
  });

  it("setConfigが直接configを設定できること", () => {
    mockExecute.mockImplementation(
      async (_url: string, options: { onSuccess: (data: MLConfig) => void }) => {
        options.onSuccess(DEFAULT_ML_CONFIG);
      }
    );

    const newConfig = {
      ...DEFAULT_ML_CONFIG,
      model: { ...DEFAULT_ML_CONFIG.model, max_model_versions: 10 },
    };

    const { result } = renderHook(() => useMLSettings());

    act(() => {
      result.current.setConfig(newConfig);
    });

    expect(result.current.config?.model.max_model_versions).toBe(10);
  });

  it("saveConfigでconfigがnullの場合に何もしないこと", async () => {
    const { result } = renderHook(() => useMLSettings());

    // 初回マウント時のauto-fetch呼び出しをクリア
    mockExecute.mockClear();

    await act(async () => {
      await result.current.saveConfig(null as unknown as MLConfig);
    });

    expect(mockExecute).not.toHaveBeenCalled();
  });

  it("複数のuseApiCallエラーが統合されること", () => {
    mockUseApiCall
      .mockReturnValueOnce({
        execute: jest.fn(),
        loading: false,
        error: "Fetch Error",
        reset: jest.fn(),
      })
      .mockReturnValueOnce({
        execute: jest.fn(),
        loading: false,
        error: null,
        reset: jest.fn(),
      })
      .mockReturnValueOnce({
        execute: jest.fn(),
        loading: false,
        error: null,
        reset: jest.fn(),
      })
      .mockReturnValueOnce({
        execute: jest.fn(),
        loading: false,
        error: null,
        reset: jest.fn(),
      });

    const { result } = renderHook(() => useMLSettings());

    expect(result.current.error).toBe("Fetch Error");
  });

  it("ローディング状態が正しく返されること", () => {
    mockUseApiCall
      .mockReturnValueOnce({
        execute: jest.fn(),
        loading: true,
        error: null,
        reset: jest.fn(),
      })
      .mockReturnValueOnce({
        execute: jest.fn(),
        loading: false,
        error: null,
        reset: jest.fn(),
      })
      .mockReturnValueOnce({
        execute: jest.fn(),
        loading: false,
        error: null,
        reset: jest.fn(),
      })
      .mockReturnValueOnce({
        execute: jest.fn(),
        loading: false,
        error: null,
        reset: jest.fn(),
      });

    const { result } = renderHook(() => useMLSettings());
    expect(result.current.isLoading).toBe(true);
  });
});
