import { render, screen } from "@testing-library/react";
import Navbar from "@/components/navigation/Navbar";

jest.mock("next/navigation", () => ({
  usePathname: jest.fn(() => "/"),
}));

jest.mock("next/link", () => {
  return ({ children, href, ...props }: any) => (
    <a href={href} {...props}>{children}</a>
  );
});

jest.mock("@/components/ui/sidebar", () => ({
  Sidebar: ({ children }: any) => <div data-testid="sidebar">{children}</div>,
  SidebarContent: ({ children }: any) => <div>{children}</div>,
  SidebarFooter: ({ children }: any) => <div>{children}</div>,
  SidebarGroup: ({ children }: any) => <div>{children}</div>,
  SidebarGroupContent: ({ children }: any) => <div>{children}</div>,
  SidebarGroupLabel: ({ children }: any) => <div>{children}</div>,
  SidebarHeader: ({ children }: any) => <div>{children}</div>,
  SidebarMenu: ({ children }: any) => <div>{children}</div>,
  SidebarMenuButton: ({ children, asChild, ...rest }: any) => <div>{children}</div>,
  SidebarMenuItem: ({ children }: any) => <div>{children}</div>,
  SidebarProvider: ({ children }: any) => <div>{children}</div>,
  SidebarRail: () => <div />,
}));

describe("Navbar", () => {
  it("Trdingerブランド名を表示すること", () => {
    render(<Navbar><div>content</div></Navbar>);
    expect(screen.getByText("Trdinger")).toBeInTheDocument();
  });

  it("ナビゲーション項目を表示すること", () => {
    render(<Navbar><div>content</div></Navbar>);
    expect(screen.getByText("Dashboard")).toBeInTheDocument();
    expect(screen.getByText("Backtest")).toBeInTheDocument();
    expect(screen.getByText("Data Management")).toBeInTheDocument();
  });

  it("子要素を表示すること", () => {
    render(<Navbar><div>テストコンテンツ</div></Navbar>);
    expect(screen.getByText("テストコンテンツ")).toBeInTheDocument();
  });

  it("バージョン情報を表示すること", () => {
    render(<Navbar><div>content</div></Navbar>);
    expect(screen.getByText("v1.0.0")).toBeInTheDocument();
  });

  it("サイドバーが存在すること", () => {
    render(<Navbar><div>content</div></Navbar>);
    expect(screen.getByTestId("sidebar")).toBeInTheDocument();
  });
});
