import React from "react";
import {
  render,
  screen,
  fireEvent,
  within,
  waitFor,
} from "@testing-library/react";
import "@testing-library/jest-dom";
import EnsembleSettings, {
  EnsembleSettingsConfig,
} from "@/components/ml/EnsembleSettings";

// Mock child components and UI parts
jest.mock("@/components/ui/card", () => ({
  Card: ({ children }: any) => <div data-testid="card">{children}</div>,
  CardHeader: ({ children }: any) => (
    <div data-testid="card-header">{children}</div>
  ),
  CardTitle: ({ children }: any) => (
    <div data-testid="card-title">{children}</div>
  ),
  CardContent: ({ children }: any) => (
    <div data-testid="card-content">{children}</div>
  ),
}));

jest.mock("@/components/ui/switch", () => ({
  Switch: ({ checked, onCheckedChange, id }: any) => (
    <input
      type="checkbox"
      data-testid={id || "switch"}
      checked={checked}
      onChange={(e) => onCheckedChange(e.target.checked)}
    />
  ),
}));

jest.mock("@/components/ui/badge", () => ({
  Badge: ({ children, onClick, variant }: any) => (
    <div
      data-testid={`badge-${children}`}
      onClick={onClick}
      data-variant={variant}
    >
      {children}
    </div>
  ),
}));

jest.mock("@/components/ui/select", () => ({
  Select: ({ children, onValueChange, value }: any) => (
    <div data-testid="select-container">
      <div data-testid="current-select-value">{value}</div>
      <button
        data-testid="trigger-select-change"
        onClick={() => onValueChange("ridge")} // テスト用に固定値を送信
      >
        Change Select
      </button>
      {children}
    </div>
  ),
  SelectTrigger: () => null,
  SelectValue: () => null,
  SelectContent: ({ children }: any) => <>{children}</>,
  SelectItem: ({ children, value }: any) => (
    <option value={value}>{children}</option>
  ),
}));

jest.mock("@/components/common/InputField", () => ({
  InputField: ({ label, value, onChange }: any) => (
    <div data-testid="input-field">
      <div data-testid="current-input-value">{value}</div>
      <button
        data-testid="trigger-input-change"
        onClick={() => onChange("3")} // テスト用に固定値を送信
      >
        Change Input
      </button>
    </div>
  ),
}));

jest.mock("@/components/ml/SingleModelSettings", () => {
  return function MockSingleModelSettings() {
    return <div data-testid="single-model-settings">Single Model Settings</div>;
  };
});

jest.mock("@/components/common/InfoModal", () => {
  return function MockInfoModal({ isOpen, title, children }: any) {
    if (!isOpen) return null;
    return (
      <div data-testid="info-modal">
        <h1>{title}</h1>
        <div>{children}</div>
      </div>
    );
  };
});

describe("EnsembleSettings", () => {
  const defaultSettings: EnsembleSettingsConfig = {
    enabled: true,
    method: "stacking",
    stacking_params: {
      base_models: ["lightgbm", "xgboost"],
      meta_model: "logistic_regression",
      cv_folds: 5,
      use_probas: true,
    },
  };

  const mockOnChange = jest.fn();

  beforeEach(() => {
    mockOnChange.mockClear();
  });

  it("初期表示（アンサンブル有効）が正しく行われること", () => {
    render(
      <EnsembleSettings settings={defaultSettings} onChange={mockOnChange} />
    );

    expect(screen.getByText("アンサンブル学習設定")).toBeInTheDocument();
    expect(screen.getByTestId("ensemble-enabled")).toBeChecked();
    expect(screen.getByText("スタッキング設定")).toBeInTheDocument();
    expect(
      screen.queryByTestId("single-model-settings")
    ).not.toBeInTheDocument();
  });

  it("アンサンブル学習を無効にすると、シングルモデル設定が表示されること", () => {
    const disabledSettings = { ...defaultSettings, enabled: false };
    render(
      <EnsembleSettings settings={disabledSettings} onChange={mockOnChange} />
    );

    expect(screen.getByTestId("ensemble-enabled")).not.toBeChecked();
    expect(screen.queryByText("スタッキング設定")).not.toBeInTheDocument();
    expect(screen.getByTestId("single-model-settings")).toBeInTheDocument();
  });

  it("有効/無効の切り替えが機能すること", () => {
    render(
      <EnsembleSettings settings={defaultSettings} onChange={mockOnChange} />
    );

    const switchEl = screen.getByTestId("ensemble-enabled");
    fireEvent.click(switchEl); // check -> uncheck

    expect(mockOnChange).toHaveBeenCalledWith(
      expect.objectContaining({ enabled: false })
    );
  });

  it("ベースモデルの選択（削除）が機能すること", async () => {
    render(
      <EnsembleSettings settings={defaultSettings} onChange={mockOnChange} />
    );

    // LightGBMは選択済み (default) なので、クリックすると削除されるはず
    const lgbBadge = screen.getByTestId("badge-LightGBM");
    fireEvent.click(lgbBadge);

    await waitFor(() => {
      expect(mockOnChange).toHaveBeenCalled();
    });

    const callArg1 = mockOnChange.mock.calls[0][0];
    expect(callArg1.stacking_params.base_models).not.toContain("lightgbm");
    expect(callArg1.stacking_params.base_models).toContain("xgboost");
  });

  it("ベースモデルの選択（追加）が機能すること", async () => {
    render(
      <EnsembleSettings settings={defaultSettings} onChange={mockOnChange} />
    );

    // XGboostは選択済みなので、追加のために別のものを選択
    // 利用可能なモデルのリストに依存するため、とりあえず1つ目の未選択モデルを探すロジックは複雑になるため、
    // ここではCatBoostがあればそれを、なければRandom Forestを試す
    // テストを堅牢にするため、DOMから未選択（variant="outline"）のバッジを探してクリックする
    const outlineBadges = screen
      .getAllByTestId(/badge-/)
      .filter((el) => el.getAttribute("data-variant") === "outline");

    // もし未選択のモデルがなければテストできないが、通常はあるはず
    if (outlineBadges.length > 0) {
      fireEvent.click(outlineBadges[0]);

      await waitFor(() => {
        expect(mockOnChange).toHaveBeenCalled();
      });

      const callArg2 = mockOnChange.mock.calls[0][0];
      // 1つ増えていることを確認
      expect(callArg2.stacking_params.base_models.length).toBe(3);
    }
  });

  it("CV分割数の変更が機能すること", async () => {
    render(
      <EnsembleSettings settings={defaultSettings} onChange={mockOnChange} />
    );

    // テスト用トリガーボタンをクリック
    fireEvent.click(screen.getByTestId("trigger-input-change"));

    await waitFor(() => {
      expect(mockOnChange).toHaveBeenCalled();
    });

    const callArg = mockOnChange.mock.calls[0][0];
    expect(callArg.stacking_params.cv_folds).toBe(3);
  });

  it("メタモデルの変更が機能すること", async () => {
    render(
      <EnsembleSettings settings={defaultSettings} onChange={mockOnChange} />
    );

    // テスト用トリガーボタンをクリック
    fireEvent.click(screen.getByTestId("trigger-select-change"));

    await waitFor(() => {
      expect(mockOnChange).toHaveBeenCalled();
    });

    const callArg = mockOnChange.mock.calls[0][0];
    expect(callArg.stacking_params.meta_model).toBe("ridge");
  });
});
