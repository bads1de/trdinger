import { render, screen } from "@testing-library/react";
import LoadingSpinner from "@/components/common/LoadingSpinner";

describe("LoadingSpinner", () => {
  it("デフォルトテキストを表示すること", () => {
    render(<LoadingSpinner />);
    expect(screen.getByText("読み込み中...")).toBeInTheDocument();
  });

  it("カスタムテキストを表示すること", () => {
    render(<LoadingSpinner text="データ取得中..." />);
    expect(screen.getByText("データ取得中...")).toBeInTheDocument();
  });

  it("スピナー要素を表示すること", () => {
    const { container } = render(<LoadingSpinner />);
    const spinner = container.querySelector(".animate-spin");
    expect(spinner).toBeInTheDocument();
  });

  it("smサイズのスピナーを表示すること", () => {
    const { container } = render(<LoadingSpinner size="sm" />);
    const spinner = container.querySelector(".h-4.w-4");
    expect(spinner).toBeInTheDocument();
  });

  it("lgサイズのスピナーを表示すること", () => {
    const { container } = render(<LoadingSpinner size="lg" />);
    const spinner = container.querySelector(".h-8.w-8");
    expect(spinner).toBeInTheDocument();
  });
});
