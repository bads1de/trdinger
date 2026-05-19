import { render, screen, fireEvent } from "@testing-library/react";
import ErrorDisplay from "@/components/common/ErrorDisplay";

jest.mock("@/components/ui/alert", () => ({
  Alert: ({ children, ...props }: any) => <div data-testid="alert" {...props}>{children}</div>,
  AlertDescription: ({ children }: any) => <span>{children}</span>,
}));

jest.mock("@/components/ui/button", () => ({
  Button: ({ children, onClick, ...props }: any) => (
    <button onClick={onClick} {...props}>{children}</button>
  ),
}));

describe("ErrorDisplay", () => {
  it("エラーメッセージを表示すること", () => {
    render(<ErrorDisplay message="エラーが発生しました" />);
    expect(screen.getByText("エラーが発生しました")).toBeInTheDocument();
  });

  it("メッセージが空の場合、何も表示しないこと", () => {
    const { container } = render(<ErrorDisplay message="" />);
    expect(container.firstChild).toBeNull();
  });

  it("onRetryが指定された場合、再試行ボタンを表示すること", () => {
    const onRetry = jest.fn();
    render(<ErrorDisplay message="エラー" onRetry={onRetry} />);
    expect(screen.getByText("再試行")).toBeInTheDocument();
  });

  it("onRetryが未指定の場合、再試行ボタンを表示しないこと", () => {
    render(<ErrorDisplay message="エラー" />);
    expect(screen.queryByText("再試行")).not.toBeInTheDocument();
  });

  it("再試行ボタンをクリックするとonRetryが呼ばれること", () => {
    const onRetry = jest.fn();
    render(<ErrorDisplay message="エラー" onRetry={onRetry} />);

    fireEvent.click(screen.getByText("再試行"));

    expect(onRetry).toHaveBeenCalledTimes(1);
  });
});
