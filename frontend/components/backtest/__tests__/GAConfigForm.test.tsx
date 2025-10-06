/**
 * GAConfigForm コンポーネントのテスト
 *
 * 指標モード選択と自動最適化設定のUIテストを行います。
 */

import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import "@testing-library/jest-dom";
import { TooltipProvider } from "@/components/ui/tooltip";
import GAConfigForm from "../GAConfigForm";

// モック関数
const mockOnSubmit = jest.fn();
const mockOnClose = jest.fn();

// テスト用の初期設定
const initialConfig = {
  experiment_name: "Test GA Experiment",
  base_config: {
    strategy_name: "Test Strategy",
    symbol: "BTC/USDT:USDT",
    timeframe: "1h",
    start_date: "2024-01-01",
    end_date: "2024-12-31",
    initial_capital: 100000,
    commission_rate: 0.001,
    strategy_config: {
      strategy_type: "",
      parameters: {},
    },
  },
  ga_config: {
    population_size: 20,
    generations: 15,
    mutation_rate: 0.1,
    crossover_rate: 0.8,
    elite_size: 5,
    max_indicators: 5,
    allowed_indicators: [],
    indicator_mode: "technical_only",
    fitness_weights: {
      total_return: 0.3,
      sharpe_ratio: 0.4,
      max_drawdown: 0.2,
      win_rate: 0.1,
    },
    fitness_constraints: {
      min_trades: 10,
      max_drawdown_limit: 0.3,
      min_sharpe_ratio: 0.5,
    },
    enable_multi_objective: true,
    objectives: ["win_rate", "max_drawdown"],
    objective_weights: [1.0, -1.0],
    regime_adaptation_enabled: false,
  },
};

// TooltipProviderでラップしたrenderヘルパー
const renderWithTooltipProvider = (component: React.ReactElement) => {
  return render(
    <TooltipProvider>
      {component}
    </TooltipProvider>
  );
};

// テストスイート
describe("GAConfigForm", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test("デフォルトでindicator_modeがtechnical_onlyであること", () => {
    renderWithTooltipProvider(
      <GAConfigForm
        onSubmit={mockOnSubmit}
        onClose={mockOnClose}
        initialConfig={initialConfig}
      />
    );

    // 指標モードのセレクト要素を取得（TAラベルで特定）
    const indicatorModeSelect = screen.getByText("TA");

    // デフォルト値が"technical_only"であることを確認
    expect(indicatorModeSelect).toBeInTheDocument();
  });

  test("指標モードの選択肢が正しく表示されること", () => {
    renderWithTooltipProvider(
      <GAConfigForm
        onSubmit={mockOnSubmit}
        onClose={mockOnClose}
        initialConfig={initialConfig}
      />
    );

    // 指標モードのセレクト要素を取得
    const indicatorModeSelect = screen.getByText("TA");

    // 要素が存在することを確認
    expect(indicatorModeSelect).toBeInTheDocument();

    // ラベルが存在することを確認
    expect(screen.getByText("指標モード (indicator_mode)")).toBeInTheDocument();
  });

  test("自動最適化説明セクションがデフォルトで折りたたまれていること", () => {
    renderWithTooltipProvider(
      <GAConfigForm
        onSubmit={mockOnSubmit}
        onClose={mockOnClose}
        initialConfig={initialConfig}
      />
    );

    // Collapsibleのトリガー要素を取得
    const collapsibleTrigger = screen.getByRole("button", {
      name: /自動最適化設定説明/i,
    });

    // aria-expanded属性がfalseであることを確認（デフォルトで閉じている）
    expect(collapsibleTrigger).toHaveAttribute("aria-expanded", "false");

    // 説明コンテンツが表示されていないことを確認
    expect(screen.queryByText(/TA: 従来のテクニカル指標のみを使用/)).not.toBeInTheDocument();
    expect(screen.queryByText("TP/SLとポジションサイズはGAが自動最適化します。")).not.toBeInTheDocument();
  });

  test("Collapsibleをクリックすると説明が表示されること", () => {
    renderWithTooltipProvider(
      <GAConfigForm
        onSubmit={mockOnSubmit}
        onClose={mockOnClose}
        initialConfig={initialConfig}
      />
    );

    // Collapsibleのトリガーをクリック
    const collapsibleTrigger = screen.getByRole("button", {
      name: /自動最適化設定説明/i,
    });
    fireEvent.click(collapsibleTrigger);

    // aria-expanded属性がtrueに変わったことを確認
    expect(collapsibleTrigger).toHaveAttribute("aria-expanded", "true");

    // 説明コンテンツが表示されることを確認
    expect(screen.getByText(/指標モード選択/)).toBeInTheDocument();
    expect(screen.getByText("TP/SLとポジションサイズはGAが自動最適化します。")).toBeInTheDocument();
  });

  test("onSubmitが正しい設定で呼び出されること", () => {
    renderWithTooltipProvider(
      <GAConfigForm
        onSubmit={mockOnSubmit}
        onClose={mockOnClose}
        initialConfig={initialConfig}
      />
    );

    // 送信ボタンをクリック
    const submitButton = screen.getByRole("button", { name: /GA戦略を生成/i });
    fireEvent.click(submitButton);

    // onSubmitが呼び出されたことを確認
    expect(mockOnSubmit).toHaveBeenCalledTimes(1);
    const submittedConfig = mockOnSubmit.mock.calls[0][0];

    // indicator_modeがtechnical_onlyであることを確認
    expect(submittedConfig.ga_config.indicator_mode).toBe("technical_only");
  });

  test("indicator_modeを変更できること", () => {
    renderWithTooltipProvider(
      <GAConfigForm
        onSubmit={mockOnSubmit}
        onClose={mockOnClose}
        initialConfig={initialConfig}
      />
    );

    // 現在の値がTAであることを確認
    expect(screen.getByText("TA")).toBeInTheDocument();

    // 指標モードのセレクトトリガーをクリック（ドロップダウンを開く）
    const selectTrigger = screen
      .getAllByRole("combobox")
      .find((element) => element.textContent?.includes("TA"));
    expect(selectTrigger).toBeTruthy();
    fireEvent.click(selectTrigger!);

    // MLオプションをクリック
    const mlOption = screen.getByText("ML");
    fireEvent.click(mlOption);

    // 値が変更されたことを確認
    expect(screen.getByText("ML")).toBeInTheDocument();

    // フォームを送信
    const submitButton = screen.getByRole("button", { name: /GA戦略を生成/i });
    fireEvent.click(submitButton);

    // onSubmitが正しい値で呼び出されたことを確認
    const submittedConfig = mockOnSubmit.mock.calls[0][0];
    expect(submittedConfig.ga_config.indicator_mode).toBe("ml_only");
  });

  test("レジーム適応チェックボックスがデフォルトで未チェックであること", () => {
    renderWithTooltipProvider(
      <GAConfigForm
        onSubmit={mockOnSubmit}
        onClose={mockOnClose}
        initialConfig={initialConfig}
      />
    );

    // レジーム適応チェックボックスを取得
    const regimeCheckbox = screen.getByLabelText("レジーム適応を有効化");

    // デフォルトで未チェックであることを確認
    expect(regimeCheckbox).not.toBeChecked();
  });

  test("レジーム適応チェックボックスをチェックできること", () => {
    renderWithTooltipProvider(
      <GAConfigForm
        onSubmit={mockOnSubmit}
        onClose={mockOnClose}
        initialConfig={initialConfig}
      />
    );

    // レジーム適応チェックボックスを取得
    const regimeCheckbox = screen.getByLabelText("レジーム適応を有効化");

    // チェックボックスをクリック
    fireEvent.click(regimeCheckbox);

    // チェックされたことを確認
    expect(regimeCheckbox).toBeChecked();

    // フォームを送信
    const submitButton = screen.getByRole("button", { name: /GA戦略を生成/i });
    fireEvent.click(submitButton);

    // onSubmitが正しい値で呼び出されたことを確認
    const submittedConfig = mockOnSubmit.mock.calls[0][0];
    expect(submittedConfig.ga_config.regime_adaptation_enabled).toBe(true);
  });

  test("レジーム適応チェックボックスがtrueで初期化されるとチェックされること", () => {
    const configWithRegimeEnabled = {
      ...initialConfig,
      ga_config: {
        ...initialConfig.ga_config,
        regime_adaptation_enabled: true,
      },
    };

    renderWithTooltipProvider(
      <GAConfigForm
        onSubmit={mockOnSubmit}
        onClose={mockOnClose}
        initialConfig={configWithRegimeEnabled}
      />
    );

    // レジーム適応チェックボックスを取得
    const regimeCheckbox = screen.getByLabelText("レジーム適応を有効化");

    // チェックされていることを確認
    expect(regimeCheckbox).toBeChecked();
  });

  test("ハイブリッドモードでDRL設定を有効化できること", () => {
    renderWithTooltipProvider(
      <GAConfigForm
        onSubmit={mockOnSubmit}
        onClose={mockOnClose}
        initialConfig={initialConfig}
      />
    );

    const hybridToggle = screen.getByLabelText("ハイブリッドGA+MLモードを有効化");
    fireEvent.click(hybridToggle);

    const drlToggle = screen.getByLabelText("DRLポリシーブレンドを有効化");
    fireEvent.click(drlToggle);

    const submitButton = screen.getByRole("button", { name: /GA戦略を生成/i });
    fireEvent.click(submitButton);

    expect(mockOnSubmit).toHaveBeenCalledTimes(1);
    const submittedConfig = mockOnSubmit.mock.calls[0][0];

    expect(submittedConfig.ga_config.hybrid_mode).toBe(true);
    expect(
      submittedConfig.ga_config.hybrid_automl_config?.drl?.enabled
    ).toBe(true);
    expect(
      submittedConfig.ga_config.hybrid_automl_config?.drl?.policy_type
    ).toBe("ppo");
    expect(
      submittedConfig.ga_config.hybrid_automl_config?.drl?.policy_weight
    ).toBeCloseTo(0.5);
  });
});