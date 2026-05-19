import { render, screen, fireEvent } from "@testing-library/react";
import { InputField } from "@/components/common/InputField";

jest.mock("@/components/ui/tooltip", () => ({
  Tooltip: ({ children }: any) => <div>{children}</div>,
  TooltipTrigger: ({ children }: any) => <div>{children}</div>,
  TooltipContent: ({ children }: any) => <div>{children}</div>,
}));

jest.mock("@/components/ui/input", () => ({
  Input: ({ ...props }: any) => <input data-testid="input" {...props} />,
}));

describe("InputField", () => {
  it("ラベルを表示すること", () => {
    render(<InputField label="テストラベル" value="" onChange={jest.fn()} />);
    expect(screen.getByText("テストラベル")).toBeInTheDocument();
  });

  it("入力値を表示すること", () => {
    render(<InputField label="テスト" value="hello" onChange={jest.fn()} />);
    expect(screen.getByTestId("input")).toHaveValue("hello");
  });

  it("onChangeが呼ばれること", () => {
    const onChange = jest.fn();
    render(<InputField label="テスト" value="" onChange={onChange} />);

    fireEvent.change(screen.getByTestId("input"), {
      target: { value: "new value" },
    });

    expect(onChange).toHaveBeenCalledWith("new value");
  });

  it("数値型の場合、数値に変換してonChangeを呼ぶこと", () => {
    const onChange = jest.fn();
    render(<InputField label="テスト" value={0} onChange={onChange} type="number" />);

    fireEvent.change(screen.getByTestId("input"), {
      target: { value: "42" },
    });

    expect(onChange).toHaveBeenCalledWith(42);
  });

  it("disabledの場合、入力が無効になること", () => {
    render(<InputField label="テスト" value="" onChange={jest.fn()} disabled />);
    expect(screen.getByTestId("input")).toBeDisabled();
  });

  it("placeholderを表示すること", () => {
    render(
      <InputField label="テスト" value="" onChange={jest.fn()} placeholder="入力してください" />
    );
    expect(screen.getByTestId("input")).toHaveAttribute("placeholder", "入力してください");
  });

  it("descriptionが指定された場合、Infoアイコンを表示すること", () => {
    render(
      <InputField label="テスト" value="" onChange={jest.fn()} description="説明文" />
    );
    expect(screen.getByText("説明文")).toBeInTheDocument();
  });

  it("min/max属性を設定すること", () => {
    render(
      <InputField label="テスト" value={5} onChange={jest.fn()} type="number" min={0} max={10} />
    );
    const input = screen.getByTestId("input");
    expect(input).toHaveAttribute("min", "0");
    expect(input).toHaveAttribute("max", "10");
  });
});
