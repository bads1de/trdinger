import { render, screen } from "@testing-library/react";
import { SelectField } from "@/components/common/SelectField";

jest.mock("@/components/ui/select", () => ({
  Select: ({ children, onValueChange, value }: any) => (
    <div data-testid="select" data-value={value} data-onchange={typeof onValueChange}>
      {children}
    </div>
  ),
  SelectTrigger: ({ children, className }: any) => (
    <div data-testid="select-trigger" className={className}>{children}</div>
  ),
  SelectValue: ({ placeholder }: any) => <span>{placeholder}</span>,
  SelectContent: ({ children }: any) => <div>{children}</div>,
  SelectItem: ({ children, value }: any) => (
    <div data-testid="select-item" data-value={value}>{children}</div>
  ),
}));

describe("SelectField", () => {
  const options = [
    { value: "option1", label: "オプション1" },
    { value: "option2", label: "オプション2" },
    { value: "option3", label: "オプション3" },
  ];

  it("ラベルを表示すること", () => {
    render(
      <SelectField label="テストラベル" value="option1" onChange={jest.fn()} options={options} />
    );
    expect(screen.getAllByText("テストラベル").length).toBeGreaterThan(0);
  });

  it("オプションを表示すること", () => {
    render(
      <SelectField label="テスト" value="option1" onChange={jest.fn()} options={options} />
    );
    expect(screen.getByText("オプション1")).toBeInTheDocument();
    expect(screen.getByText("オプション2")).toBeInTheDocument();
    expect(screen.getByText("オプション3")).toBeInTheDocument();
  });

  it("選択された値を表示すること", () => {
    render(
      <SelectField label="テスト" value="option2" onChange={jest.fn()} options={options} />
    );
    expect(screen.getByTestId("select")).toHaveAttribute("data-value", "option2");
  });

  it("disabled状態を設定すること", () => {
    render(
      <SelectField label="テスト" value="option1" onChange={jest.fn()} options={options} disabled />
    );
    expect(screen.getByTestId("select")).toBeInTheDocument();
  });

  it("空のオプションリストを処理すること", () => {
    render(
      <SelectField label="ユニークラベル" value="" onChange={jest.fn()} options={[]} />
    );
    expect(screen.getAllByText("ユニークラベル").length).toBeGreaterThan(0);
  });
});
