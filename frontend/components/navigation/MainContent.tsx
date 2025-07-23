/**
 * メインコンテンツラッパーコンポーネント
 *
 * shadcn/uiのSidebarInsetコンポーネントを使用して
 * サイドバーの状態に応じて自動的にレイアウトを調整します。
 *
 */

"use client";

import React from "react";
import { usePathname } from "next/navigation";
import { SidebarInset, SidebarTrigger } from "@/components/ui/sidebar";
import { Separator } from "@/components/ui/separator";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb";

interface MainContentProps {
  children: React.ReactNode;
}

/**
 * パスに基づいてページタイトルを取得
 */
const getPageTitle = (pathname: string): string => {
  if (pathname === "/") return "Dashboard";

  if (pathname.startsWith("/backtest")) return "Backtest";

  if (pathname.startsWith("/ml")) return "ML Management";

  if (pathname.startsWith("/data")) return "Data Management";

  return "Dashboard";
};

/**
 * メインコンテンツラッパーコンポーネント
 */
const MainContent: React.FC<MainContentProps> = ({ children }) => {
  const pathname = usePathname();
  const pageTitle = getPageTitle(pathname);
  return (
    <SidebarInset>
      <header className="flex h-16 shrink-0 items-center gap-2 transition-[width,height] ease-linear group-has-[[data-collapsible=icon]]/sidebar-wrapper:h-12">
        <div className="flex items-center gap-2 px-4">
          <SidebarTrigger className="-ml-1" />
          <Separator orientation="vertical" className="mr-2 h-4" />
          <Breadcrumb>
            <BreadcrumbList>
              <BreadcrumbItem className="hidden md:block">
                <BreadcrumbLink href="/">Trdinger</BreadcrumbLink>
              </BreadcrumbItem>
              <BreadcrumbSeparator className="hidden md:block" />
              <BreadcrumbItem>
                <BreadcrumbPage>{pageTitle}</BreadcrumbPage>
              </BreadcrumbItem>
            </BreadcrumbList>
          </Breadcrumb>
        </div>
      </header>
      <div className="flex flex-1 flex-col gap-4 p-4 pt-0">{children}</div>
    </SidebarInset>
  );
};

export default MainContent;
