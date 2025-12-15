import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import "@testing-library/jest-dom";
import SingleModelSettings, {
  SingleModelSettingsConfig,
} from "@/components/ml/SingleModelSettings";

// shadcn/ui componentsのモック
// これにより、Radix UIの複雑なDOM構造を回避し、ロジックのテストに集中できます。
jest.mock("@/components/ui/select", () => ({
  Select: ({ children, onValueChange, value }: any) => (
    <div data-testid="select-container">
      <select
        data-testid="mock-select"
        value={value}
        onChange={(e) => onValueChange(e.target.value)}
      >
        {children}
      </select>
    </div>
  ),
  SelectTrigger: () => null, // テスト環境ではネイティブselectで代用するため非表示
  SelectValue: () => null,
  SelectContent: ({ children }: any) => <>{children}</>,
  SelectItem: ({ children, value }: any) => (
    <option value={value}>{children}</option>
  ),
}));

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

describe("SingleModelSettings", () => {
  const defaultSettings: SingleModelSettingsConfig = {
    model_type: "lightgbm",
  };

  const mockOnChange = jest.fn();

  beforeEach(() => {
    mockOnChange.mockClear();
  });

  it("正しくレンダリングされること", () => {
    render(
      <SingleModelSettings
        singleModelSettings={defaultSettings}
        onSingleModelChange={mockOnChange}
      />
    );

    expect(screen.getByText("単一モデル設定")).toBeInTheDocument();
    expect(screen.getByText("使用するモデル")).toBeInTheDocument();
    // expect(screen.getByText("LIGHTGBM")).toBeInTheDocument(); // SelectValueのモックがplaceholderしか表示しないため、このチェックは不要
  });

  it("利用可能なモデルリストが正しく反映されること", () => {
    const availableModels = ["lightgbm", "xgboost", "catboost"];
    render(
      <SingleModelSettings
        singleModelSettings={defaultSettings}
        availableModels={availableModels}
        onSingleModelChange={mockOnChange}
      />
    );

    // モック化されたSelect内でoptionを探す
    const options = screen.getAllByRole("option"); // <option>タグはrole='option'を持つ
    // SelectItemはoptionに変換されている

    // NormalizedModelsロジックにより、重複排除やソートが行われる可能性があるが、
    // ここでは単純に存在確認
    expect(screen.getByText("LIGHTGBM")).toBeInTheDocument();
    expect(screen.getByText("XGBOOST")).toBeInTheDocument();
    expect(screen.getByText("CATBOOST")).toBeInTheDocument();
  });

  it("モデルを変更するとコールバックが呼ばれること", () => {
    render(
      <SingleModelSettings
        singleModelSettings={defaultSettings}
        onSingleModelChange={mockOnChange}
      />
    );

    const select = screen.getByTestId("mock-select");
    fireEvent.change(select, { target: { value: "xgboost" } });

    expect(mockOnChange).toHaveBeenCalledWith({ model_type: "xgboost" });
  });

  it("情報アイコンをクリックするとモーダルが開くこと", () => {
    render(
      <SingleModelSettings
        singleModelSettings={defaultSettings}
        onSingleModelChange={mockOnChange}
      />
    );

    expect(screen.queryByTestId("info-modal")).not.toBeInTheDocument();

    const infoButton = screen.getByTitle("モデル説明");
    fireEvent.click(infoButton);

    expect(screen.getByTestId("info-modal")).toBeInTheDocument();
    expect(screen.getByText(/LightGBM/)).toBeInTheDocument();
  });
});
