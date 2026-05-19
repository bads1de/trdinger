import { render, screen, fireEvent } from "@testing-library/react";
import DataTable from "@/components/table/DataTable";
import { TableColumn } from "@/types/common";

jest.mock("@/components/common/Icons", () => ({
  SortNeutralIcon: ({ className }: any) => <span data-testid="sort-neutral" className={className} />,
  SortAscIcon: ({ className }: any) => <span data-testid="sort-asc" className={className} />,
  SortDescIcon: ({ className }: any) => <span data-testid="sort-desc" className={className} />,
  ErrorIcon: () => <span />,
  ExportIcon: () => <span />,
  SearchIcon: () => <span />,
}));

jest.mock("@/components/common/LoadingSpinner", () => {
  return function MockLoadingSpinner({ text }: { text?: string }) {
    return <div data-testid="loading-spinner">{text}</div>;
  };
});

jest.mock("@/components/common/ErrorDisplay", () => {
  return function MockErrorDisplay({ message }: { message: string }) {
    return <div data-testid="error-display">{message}</div>;
  };
});

interface TestData {
  id: number;
  name: string;
  value: number;
}

const testColumns: TableColumn<TestData>[] = [
  { key: "id", header: "ID", sortable: true },
  { key: "name", header: "名前", sortable: true },
  { key: "value", header: "値", sortable: false },
];

const testData: TestData[] = [
  { id: 1, name: "Alice", value: 100 },
  { id: 2, name: "Bob", value: 200 },
  { id: 3, name: "Charlie", value: 150 },
];

describe("DataTable", () => {
  it("タイトルを表示すること", () => {
    render(
      <DataTable data={testData} columns={testColumns} title="テストテーブル" />
    );
    expect(screen.getByText("テストテーブル")).toBeInTheDocument();
  });

  it("データ件数を表示すること", () => {
    render(<DataTable data={testData} columns={testColumns} />);
    expect(screen.getByText("3件のデータを表示中")).toBeInTheDocument();
  });

  it("カラムヘッダーを表示すること", () => {
    render(<DataTable data={testData} columns={testColumns} />);
    expect(screen.getByText("ID")).toBeInTheDocument();
    expect(screen.getByText("名前")).toBeInTheDocument();
    expect(screen.getByText("値")).toBeInTheDocument();
  });

  it("データ行を表示すること", () => {
    render(<DataTable data={testData} columns={testColumns} />);
    expect(screen.getByText("Alice")).toBeInTheDocument();
    expect(screen.getByText("Bob")).toBeInTheDocument();
    expect(screen.getByText("Charlie")).toBeInTheDocument();
  });

  it("ローディング中にスピナーを表示すること", () => {
    render(<DataTable data={[]} columns={testColumns} loading={true} />);
    expect(screen.getByTestId("loading-spinner")).toBeInTheDocument();
  });

  it("エラー時にエラー表示をすること", () => {
    render(<DataTable data={[]} columns={testColumns} error="エラーが発生しました" />);
    expect(screen.getByTestId("error-display")).toBeInTheDocument();
    expect(screen.getByText("エラーが発生しました")).toBeInTheDocument();
  });

  it("データが空の場合にメッセージを表示すること", () => {
    render(<DataTable data={[]} columns={testColumns} />);
    expect(screen.getByText("データがありません")).toBeInTheDocument();
  });

  it("ソート可能カラムのヘッダーをクリックするとソートすること", () => {
    render(<DataTable data={testData} columns={testColumns} />);

    fireEvent.click(screen.getByText("ID"));

    expect(screen.getByTestId("sort-asc")).toBeInTheDocument();
  });

  it("CSV出力ボタンを表示すること", () => {
    render(<DataTable data={testData} columns={testColumns} />);
    expect(screen.getByText("CSV出力")).toBeInTheDocument();
  });

  it("enableExport=falseの場合、CSV出力ボタンを表示しないこと", () => {
    render(<DataTable data={testData} columns={testColumns} enableExport={false} />);
    expect(screen.queryByText("CSV出力")).not.toBeInTheDocument();
  });

  it("ページネーションを表示すること（大量データ）", () => {
    const largeData = Array.from({ length: 100 }, (_, i) => ({
      id: i + 1,
      name: `User ${i + 1}`,
      value: i * 10,
    }));

    render(
      <DataTable data={largeData} columns={testColumns} pageSize={10} />
    );

    expect(screen.getByText("1 / 10")).toBeInTheDocument();
    expect(screen.getByText("前へ")).toBeInTheDocument();
    expect(screen.getByText("次へ")).toBeInTheDocument();
  });

  it("検索入力欄を表示すること（searchKeys指定時）", () => {
    render(
      <DataTable
        data={testData}
        columns={testColumns}
        enableSearch={true}
        searchKeys={["name"]}
      />
    );

    expect(screen.getByPlaceholderText("検索...")).toBeInTheDocument();
  });
});
