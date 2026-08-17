/**
 * GAConfigForm コンポーネントのテスト
 *
 * 指標モード選択と自動最適化設定のUIテストを行います。
 */

import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import "@testing-library/jest-dom";
import { TooltipProvider } from "@/components/ui/tooltip";
import GAConfigForm from "@/components/backtest/GAConfigForm";

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

/**
 * InputField のラベルから対応する input 要素を特定するヘルパー。
 * InputField は <label> と <Input> が別々の div に存在し htmlFor を持たないため、
 * getByLabelText では特定できない。DOM構造（label → gap div → header div → root div → input）
 * を辿って input を取得する。
 */
const getInputFieldByLabel = (labelText: string): HTMLInputElement => {
  const label = screen.getByText(labelText);
  const root = label.closest("div")?.parentElement?.parentElement;
  const input = root?.querySelector("input");
  if (!input) {
    throw new Error(`Input not found for label: ${labelText}`);
  }
  return input as HTMLInputElement;
};

// テストスイート
describe("GAConfigForm", () => {
  beforeEach(() => {
    jest.clearAllMocks();
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

    // GA設定が正しく送信されることを確認（indicator_modeなし）
    expect(submittedConfig.ga_config).toBeDefined();
    expect(submittedConfig.ga_config.population_size).toBe(20);
    expect(submittedConfig.ga_config.generations).toBe(15);
    expect(submittedConfig.ga_config.max_indicators).toBe(5);
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
    const regimeCheckbox = screen.getByLabelText("動的重み付け (レジーム適応)");

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
    const regimeCheckbox = screen.getByLabelText("動的重み付け (レジーム適応)");

    // チェックボックスをクリック
    fireEvent.click(regimeCheckbox);

    // チェックされたことを確認
    expect(regimeCheckbox).toBeChecked();

    // フォームを送信
    const submitButton = screen.getByRole("button", { name: /GA戦略を生成/i });
    fireEvent.click(submitButton);

    // onSubmitが正しい値で呼び出されたことを確認
    const submittedConfig = mockOnSubmit.mock.calls[0][0];
    expect(submittedConfig.ga_config.dynamic_objective_reweighting).toBe(true);
  });

  test("レジーム適応チェックボックスがtrueで初期化されるとチェックされること", () => {
    const configWithRegimeEnabled = {
      ...initialConfig,
      ga_config: {
        ...initialConfig.ga_config,
        dynamic_objective_reweighting: true,
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
    const regimeCheckbox = screen.getByLabelText("動的重み付け (レジーム適応)");

    // チェックされていることを確認
    expect(regimeCheckbox).toBeChecked();
  });

  // ------------------------------------------------------------------
  // 自動検証パイプライン
  // ------------------------------------------------------------------

  test("自動検証パイプラインがデフォルトで有効であること", () => {
    renderWithTooltipProvider(
      <GAConfigForm
        onSubmit={mockOnSubmit}
        onClose={mockOnClose}
        initialConfig={initialConfig}
      />
    );

    const validationToggle = screen.getByLabelText("自動検証パイプラインを有効化");
    expect(validationToggle).toBeChecked();
    // 有効時は詳細設定が表示される
    expect(screen.getByText("合格率下限 (min_pass_rate)")).toBeInTheDocument();
  });

  test("自動検証パイプラインを無効化できること", () => {
    renderWithTooltipProvider(
      <GAConfigForm
        onSubmit={mockOnSubmit}
        onClose={mockOnClose}
        initialConfig={initialConfig}
      />
    );

    // デフォルト有効 → 無効化
    const validationToggle = screen.getByLabelText("自動検証パイプラインを有効化");
    fireEvent.click(validationToggle);

    // 無効化すると詳細設定が非表示になる
    expect(validationToggle).not.toBeChecked();
    expect(screen.queryByText("合格率下限 (min_pass_rate)")).not.toBeInTheDocument();

    // フォームを送信
    const submitButton = screen.getByRole("button", { name: /GA戦略を生成/i });
    fireEvent.click(submitButton);

    const submittedConfig = mockOnSubmit.mock.calls[0][0];
    expect(submittedConfig.ga_config.validation_config.enabled).toBe(false);
  });

  test("自動検証パイプラインが有効なら詳細設定が表示され設定が送信されること", () => {
    renderWithTooltipProvider(
      <GAConfigForm
        onSubmit={mockOnSubmit}
        onClose={mockOnClose}
        initialConfig={initialConfig}
      />
    );

    // デフォルト有効のため詳細設定が表示されている
    expect(screen.getByText("合格率下限 (min_pass_rate)")).toBeInTheDocument();
    expect(screen.getByText("検証用WFAフォールド数")).toBeInTheDocument();

    // フォームを送信
    const submitButton = screen.getByRole("button", { name: /GA戦略を生成/i });
    fireEvent.click(submitButton);

    const submittedConfig = mockOnSubmit.mock.calls[0][0];
    expect(submittedConfig.ga_config.validation_config.enabled).toBe(true);
    expect(submittedConfig.ga_config.validation_config.min_pass_rate).toBe(0.5);
    expect(submittedConfig.ga_config.validation_config.wfa_n_folds).toBe(5);
    expect(submittedConfig.ga_config.validation_config.validate_candidates).toBe(true);
    expect(submittedConfig.ga_config.validation_config.max_candidates).toBe(5);
  });

  test("自動検証パイプラインの設定値を変更して送信できること", () => {
    renderWithTooltipProvider(
      <GAConfigForm
        onSubmit={mockOnSubmit}
        onClose={mockOnClose}
        initialConfig={initialConfig}
      />
    );

    // 合格率下限を変更（InputFieldのinputはラベルから特定）
    const minPassRateInput = getInputFieldByLabel("合格率下限 (min_pass_rate)");
    fireEvent.change(minPassRateInput, { target: { value: "0.8" } });

    // 候補検証をオフにする
    const candidatesCheckbox = screen.getByLabelText(/候補戦略も検証/);
    fireEvent.click(candidatesCheckbox);

    const submitButton = screen.getByRole("button", { name: /GA戦略を生成/i });
    fireEvent.click(submitButton);

    const submittedConfig = mockOnSubmit.mock.calls[0][0];
    expect(submittedConfig.ga_config.validation_config.min_pass_rate).toBe(0.8);
    expect(submittedConfig.ga_config.validation_config.validate_candidates).toBe(false);
  });

  test("自動検証パイプラインの最少取引回数を変更して送信できること", () => {
    renderWithTooltipProvider(
      <GAConfigForm
        onSubmit={mockOnSubmit}
        onClose={mockOnClose}
        initialConfig={initialConfig}
      />
    );

    // 最少取引回数（空欄の allowEmptyNumber input）をラベルから特定して変更
    const minTradesInput = getInputFieldByLabel("最少取引回数");
    fireEvent.change(minTradesInput, { target: { value: "30" } });

    const submitButton = screen.getByRole("button", { name: /GA戦略を生成/i });
    fireEvent.click(submitButton);

    const submittedConfig = mockOnSubmit.mock.calls[0][0];
    expect(submittedConfig.ga_config.validation_config.min_trades).toBe(30);
  });

  test("PBO/DSR ゲート設定を変更して送信できること", () => {
    renderWithTooltipProvider(
      <GAConfigForm
        onSubmit={mockOnSubmit}
        onClose={mockOnClose}
        initialConfig={initialConfig}
      />
    );

    // PBO 閾値を変更
    const pboThresholdInput = getInputFieldByLabel("PBO閾値 (pbo_threshold)");
    fireEvent.change(pboThresholdInput, { target: { value: "0.4" } });

    // DSR ゲートを有効化
    const dsrCheckbox = screen.getByLabelText(/DSR ゲート/);
    fireEvent.click(dsrCheckbox);
    const dsrMinInput = getInputFieldByLabel("DSR下限 (min_dsr)");
    fireEvent.change(dsrMinInput, { target: { value: "0.9" } });

    const submitButton = screen.getByRole("button", { name: /GA戦略を生成/i });
    fireEvent.click(submitButton);

    const submittedConfig = mockOnSubmit.mock.calls[0][0];
    const validationConfig = submittedConfig.ga_config.validation_config;
    expect(validationConfig.enable_pbo_gate).toBe(true);
    expect(validationConfig.pbo_threshold).toBe(0.4);
    expect(validationConfig.enable_dsr_gate).toBe(true);
    expect(validationConfig.min_dsr).toBe(0.9);
  });

  test("validation_configが初期設定で有効ならチェックされていること", () => {
    const configWithValidation = {
      ...initialConfig,
      ga_config: {
        ...initialConfig.ga_config,
        validation_config: {
          enabled: true,
          min_pass_rate: 0.7,
          max_candidates: 3,
        },
      },
    };

    renderWithTooltipProvider(
      <GAConfigForm
        onSubmit={mockOnSubmit}
        onClose={mockOnClose}
        initialConfig={configWithValidation}
      />
    );

    const validationToggle = screen.getByLabelText("自動検証パイプラインを有効化");
    expect(validationToggle).toBeChecked();
    // InputFieldはlabelがinputと関連付かないためヘルパーで対象inputを特定して値を検証
    // number input の toHaveValue は数値比較が正しい
    expect(getInputFieldByLabel("合格率下限 (min_pass_rate)")).toHaveValue(0.7);
    expect(getInputFieldByLabel("候補検証数 (max_candidates)")).toHaveValue(3);
  });

  // ------------------------------------------------------------------
  // 反復改善ループ
  // ------------------------------------------------------------------

  test("反復改善ループがデフォルトで無効であること", () => {
    renderWithTooltipProvider(
      <GAConfigForm
        onSubmit={mockOnSubmit}
        onClose={mockOnClose}
        initialConfig={initialConfig}
      />
    );

    const iterativeToggle = screen.getByLabelText("反復改善ループを有効化");
    expect(iterativeToggle).not.toBeChecked();
    expect(screen.queryByText("シード戦略数 (max_seed_strategies)")).not.toBeInTheDocument();
  });

  test("反復改善ループを有効化すると詳細設定が表示され設定が送信されること", () => {
    renderWithTooltipProvider(
      <GAConfigForm
        onSubmit={mockOnSubmit}
        onClose={mockOnClose}
        initialConfig={initialConfig}
      />
    );

    const iterativeToggle = screen.getByLabelText("反復改善ループを有効化");
    fireEvent.click(iterativeToggle);

    expect(screen.getByText("シード戦略数 (max_seed_strategies)")).toBeInTheDocument();
    expect(screen.getByLabelText(/自動検証に合格した戦略のみ使用/)).toBeInTheDocument();

    const submitButton = screen.getByRole("button", { name: /GA戦略を生成/i });
    fireEvent.click(submitButton);

    const submittedConfig = mockOnSubmit.mock.calls[0][0];
    expect(submittedConfig.ga_config.iterative_improvement_config.enabled).toBe(true);
    expect(submittedConfig.ga_config.iterative_improvement_config.max_seed_strategies).toBe(5);
    expect(submittedConfig.ga_config.iterative_improvement_config.validation_passed_only).toBe(true);
  });

  test("反復改善ループのシード戦略数と最低フィットネスを変更して送信できること", () => {
    renderWithTooltipProvider(
      <GAConfigForm
        onSubmit={mockOnSubmit}
        onClose={mockOnClose}
        initialConfig={initialConfig}
      />
    );

    const iterativeToggle = screen.getByLabelText("反復改善ループを有効化");
    fireEvent.click(iterativeToggle);

    // InputFieldのinputはラベルから特定する
    const seedCountInput = getInputFieldByLabel("シード戦略数 (max_seed_strategies)");
    fireEvent.change(seedCountInput, { target: { value: "8" } });

    const minFitnessInput = getInputFieldByLabel("最低フィットネス");
    fireEvent.change(minFitnessInput, { target: { value: "0.4" } });

    // 合格戦略のみ使用をオフにする
    const passedOnlyCheckbox = screen.getByLabelText(/自動検証に合格した戦略のみ使用/);
    fireEvent.click(passedOnlyCheckbox);

    const submitButton = screen.getByRole("button", { name: /GA戦略を生成/i });
    fireEvent.click(submitButton);

    const submittedConfig = mockOnSubmit.mock.calls[0][0];
    expect(submittedConfig.ga_config.iterative_improvement_config.max_seed_strategies).toBe(8);
    expect(submittedConfig.ga_config.iterative_improvement_config.min_fitness).toBe(0.4);
    expect(submittedConfig.ga_config.iterative_improvement_config.validation_passed_only).toBe(false);
  });
});
