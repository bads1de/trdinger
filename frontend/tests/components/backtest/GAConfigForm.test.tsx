import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import "@testing-library/jest-dom";
import GAConfigForm from "@/components/backtest/GAConfigForm";

// モックの定義
// 複雑な依存関係を持つコンポーネントを単純化する

jest.mock("@/components/backtest/BaseBacktestConfigForm", () => ({
  BaseBacktestConfigForm: () => (
    <div data-testid="base-backtest-config-form">Base Config Form</div>
  ),
}));

jest.mock("@/components/common/InputField", () => ({
  InputField: ({ label, value, onChange, allowEmptyNumber, type }: any) => (
    <div data-testid="input-field">
      <label>{label}</label>
      <input
        data-testid={`input-${label}`}
        value={value ?? ""}
        onChange={(e) => {
          if (allowEmptyNumber && type === "number" && e.target.value === "") {
            onChange(null);
            return;
          }
          onChange(Number(e.target.value) || e.target.value);
        }}
      />
    </div>
  ),
}));

jest.mock("@/components/common/SelectField", () => ({
  SelectField: ({ label, value, onChange }: any) => (
    <div data-testid="select-field">
      <label>{label}</label>
      <select
        data-testid={`select-${label}`}
        value={value}
        onChange={(e) => onChange(e.target.value)}
      >
        <option value="lightgbm">LightGBM</option>
        <option value="xgboost">XGBoost</option>
      </select>
    </div>
  ),
}));

jest.mock("@/components/backtest/optimization/ObjectiveSelection", () => ({
  ObjectiveSelection: () => (
    <div data-testid="objective-selection">Objective Selection</div>
  ),
}));

// APIボタンのモック (クリックイベントを伝播させる)
jest.mock("@/components/button/ApiButton", () => {
  return function MockApiButton({ children, onClick, loading }: any) {
    return (
      <button onClick={onClick} disabled={loading} data-testid="submit-button">
        {children}
      </button>
    );
  };
});

jest.mock("@/components/common/ActionButton", () => {
  return function MockActionButton({ children, onClick }: any) {
    return (
      <button onClick={onClick} data-testid="cancel-button">
        {children}
      </button>
    );
  };
});

// Collapsibleのモック (常に開いた状態にするか、制御可能にする)
jest.mock("@/components/ui/collapsible", () => ({
  Collapsible: ({ children }: any) => <div>{children}</div>,
  CollapsibleTrigger: ({ children }: any) => <button>{children}</button>,
  CollapsibleContent: ({ children }: any) => <div>{children}</div>,
}));

describe("GAConfigForm", () => {
  const mockOnSubmit = jest.fn();
  const mockOnClose = jest.fn();

  beforeEach(() => {
    mockOnSubmit.mockClear();
    mockOnClose.mockClear();
  });

  it("正常にレンダリングされること", () => {
    render(<GAConfigForm onSubmit={mockOnSubmit} onClose={mockOnClose} />);

    expect(screen.getByTestId("base-backtest-config-form")).toBeInTheDocument();
    expect(screen.getByText(/GA戦略を生成/)).toBeInTheDocument();

    // デフォルト値の確認
    const popInput = screen.getByTestId(
      "input-個体数 (population_size)",
    ) as HTMLInputElement;
    expect(popInput.value).toBe("10"); // デフォルト値

    // max_workersのデフォルト値確認
    const workerInput = screen.getByTestId(
      "input-最大ワーカー数",
    ) as HTMLInputElement;
    expect(workerInput.value).toBe("4");
  });

  it("多目的最適化の初期値が backend と一致すること", () => {
    render(<GAConfigForm onSubmit={mockOnSubmit} onClose={mockOnClose} />);

    fireEvent.click(screen.getByTestId("submit-button"));

    expect(mockOnSubmit).toHaveBeenCalledWith(
      expect.objectContaining({
        ga_config: expect.objectContaining({
          enable_multi_objective: false,
          objectives: ["total_return"],
          objective_weights: [1.0],
        }),
      }),
    );
  });

  it("基本設定の変更が反映されること（モック経由のため間接的検証）", () => {
    // BaseBacktestConfigFormはモックされているため、
    // ここではGAConfigForm独自のフィールド（ga_config）の変更をテストする
    render(<GAConfigForm onSubmit={mockOnSubmit} onClose={mockOnClose} />);

    const genInput = screen.getByTestId("input-世代数 (generations)");
    fireEvent.change(genInput, { target: { value: "50" } });

    // Submitして変更が反映されているか確認
    const submitBtn = screen.getByTestId("submit-button");
    fireEvent.click(submitBtn);

    expect(mockOnSubmit).toHaveBeenCalledWith(
      expect.objectContaining({
        ga_config: expect.objectContaining({
          generations: 50,
        }),
      }),
    );
  });

  it("ハイブリッドモードを有効にすると、ML設定が表示されること", () => {
    render(<GAConfigForm onSubmit={mockOnSubmit} onClose={mockOnClose} />);

    // ハイブリッドモードのチェックボックスを探す（ラベルまたはaria-labelで）
    const hybridCheckbox =
      screen.getByLabelText("ハイブリッドGA+MLモードを有効化");

    // 初期状態はオフ
    expect(hybridCheckbox).not.toBeChecked();

    // オンにする
    fireEvent.click(hybridCheckbox);
    expect(hybridCheckbox).toBeChecked();

    // MLモデル選択などが表示されることを確認
    expect(screen.getByTestId("select-MLモデル")).toBeInTheDocument();
  });

  it("並列評価設定の切り替えが機能すること", () => {
    render(<GAConfigForm onSubmit={mockOnSubmit} onClose={mockOnClose} />);

    const parallelCheckbox = screen.getByLabelText("並列評価を有効化");

    // 初期状態はオン（デフォルト）
    expect(parallelCheckbox).toBeChecked();

    // オフにする
    fireEvent.click(parallelCheckbox);
    expect(parallelCheckbox).not.toBeChecked();

    // 設定項目が消えることを確認 (max_workersの入力欄など)
    // 存在しないことを確認するにはqueryByを使用
    expect(
      screen.queryByTestId("input-最大ワーカー数"),
    ).not.toBeInTheDocument();
  });

  it("キャンセルボタンがonCloseを呼ぶこと", () => {
    render(<GAConfigForm onSubmit={mockOnSubmit} onClose={mockOnClose} />);

    const cancelBtn = screen.getByTestId("cancel-button");
    fireEvent.click(cancelBtn);

    expect(mockOnClose).toHaveBeenCalled();
  });

  it("fitness sharing の設定を nested のまま送信すること", () => {
    render(
      <GAConfigForm
        onSubmit={mockOnSubmit}
        onClose={mockOnClose}
        initialConfig={{
          ga_config: {
            fitness_sharing: {
              enable_fitness_sharing: false,
              sharing_radius: 0.25,
              sharing_alpha: 1.3,
              sampling_threshold: 80,
              sampling_ratio: 0.4,
            },
          },
        }}
      />,
    );

    fireEvent.click(screen.getByTestId("submit-button"));

    expect(mockOnSubmit).toHaveBeenCalledWith(
      expect.objectContaining({
        ga_config: expect.objectContaining({
          fitness_sharing: expect.objectContaining({
            enable_fitness_sharing: false,
            sharing_radius: 0.25,
            sharing_alpha: 1.3,
            sampling_threshold: 80,
            sampling_ratio: 0.4,
          }),
        }),
      }),
    );
  });

  it("nullable な早期打ち切り閾値を既存設定のまま送信できること", () => {
    render(
      <GAConfigForm
        onSubmit={mockOnSubmit}
        onClose={mockOnClose}
        initialConfig={{
          ga_config: {
            evaluation_config: {
              early_termination_settings: {
                enabled: true,
                max_drawdown: null,
                min_trades: null,
                min_expectancy: null,
              },
            },
          },
        }}
      />,
    );

    fireEvent.click(screen.getByTestId("submit-button"));

    expect(mockOnSubmit).toHaveBeenCalledWith(
      expect.objectContaining({
        ga_config: expect.objectContaining({
          evaluation_config: expect.objectContaining({
            early_termination_settings: expect.objectContaining({
              max_drawdown: null,
              min_trades: null,
              min_expectancy: null,
            }),
          }),
        }),
      }),
    );
  });

  it("nullable な早期打ち切り閾値を空入力で null に戻せること", () => {
    render(
      <GAConfigForm
        onSubmit={mockOnSubmit}
        onClose={mockOnClose}
        initialConfig={{
          ga_config: {
            evaluation_config: {
              early_termination_settings: {
                enabled: true,
                max_drawdown: 0.2,
              },
            },
          },
        }}
      />,
    );

    fireEvent.change(screen.getByTestId("input-最大DD打ち切り"), {
      target: { value: "" },
    });
    fireEvent.click(screen.getByTestId("submit-button"));

    expect(mockOnSubmit).toHaveBeenCalledWith(
      expect.objectContaining({
        ga_config: expect.objectContaining({
          evaluation_config: expect.objectContaining({
            early_termination_settings: expect.objectContaining({
              max_drawdown: null,
            }),
          }),
        }),
      }),
    );
  });
});
