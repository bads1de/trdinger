/**
 * 改良されたメインコンテンツラッパーコンポーネント
 *
 * トレーディングプラットフォーム向けに最適化された
 * モダンなヘッダーとレイアウトを提供します。
 * リアルタイム情報とより良いユーザーエクスペリエンスを実現します。
 *
 */

"use client";

import React, { useState, useEffect } from "react";
import { usePathname } from "next/navigation";
import { SidebarInset, SidebarTrigger } from "@/components/ui/sidebar";
import { Separator } from "@/components/ui/separator";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb";
import {
  Clock,
  TrendingUp,
  TrendingDown,
  Activity,
  RefreshCw,
  Settings,
} from "lucide-react";

interface MainContentProps {
  children: React.ReactNode;
}

/**
 * パスに基づいてページ情報を取得
 */
const getPageInfo = (pathname: string) => {
  if (pathname === "/") {
    return {
      title: "Dashboard",
      description: "トレーディング概要とリアルタイム情報",
      icon: Activity,
    };
  }

  if (pathname.startsWith("/backtest")) {
    return {
      title: "Backtest",
      description: "戦略の有効性を検証",
      icon: TrendingUp,
    };
  }

  if (pathname.startsWith("/ml")) {
    return {
      title: "ML Management",
      description: "機械学習モデル管理",
      icon: Activity,
    };
  }

  if (pathname.startsWith("/data")) {
    return {
      title: "Data Management",
      description: "価格データを管理",
      icon: Activity,
    };
  }

  return {
    title: "Dashboard",
    description: "トレーディング概要とリアルタイム情報",
    icon: Activity,
  };
};

/**
 * リアルタイム時計コンポーネント
 */
const RealTimeClock: React.FC = () => {
  const [time, setTime] = useState<Date | null>(null);

  useEffect(() => {
    setTime(new Date());
    const timer = setInterval(() => {
      setTime(new Date());
    }, 1000);

    return () => clearInterval(timer);
  }, []);

  return (
    <div className="flex items-center gap-2 px-3 py-1.5 bg-sidebar-accent/50 rounded-lg">
      <Clock className="size-4 text-muted-foreground" />
      <span className="text-sm font-mono text-foreground">
        {time
          ? time.toLocaleTimeString("ja-JP", {
              hour12: false,
              hour: "2-digit",
              minute: "2-digit",
              second: "2-digit",
            })
          : "--:--:--"}
      </span>
    </div>
  );
};

/**
 * 市場ステータスコンポーネント
 */
const MarketStatus: React.FC = () => {
  const [isMarketOpen, setIsMarketOpen] = useState(true);

  return (
    <div className="flex items-center gap-2">
      <Badge
        variant={isMarketOpen ? "default" : "secondary"}
        className={`gap-1 ${
          isMarketOpen ? "bg-green-500 hover:bg-green-600" : "bg-gray-500"
        }`}
      >
        <div
          className={`w-2 h-2 rounded-full ${
            isMarketOpen ? "bg-white animate-pulse" : "bg-gray-300"
          }`}
        />
        {isMarketOpen ? "Market Open" : "Market Closed"}
      </Badge>
    </div>
  );
};

/**
 * メインコンテンツラッパーコンポーネント
 */
const MainContent: React.FC<MainContentProps> = ({ children }) => {
  const pathname = usePathname();
  const pageInfo = getPageInfo(pathname);
  const PageIcon = pageInfo.icon;

  const reloadPage = () => {
    window.location.reload();
  };

  return (
    <SidebarInset>
      <header className="sticky top-0 z-40 flex h-16 shrink-0 items-center gap-2 border-b border-sidebar-border/50 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 transition-[width,height] ease-linear group-has-[[data-collapsible=icon]]/sidebar-wrapper:h-12">
        <div className="flex items-center justify-between w-full px-4">
          {/* 左側: ナビゲーション */}
          <div className="flex items-center gap-2">
            <SidebarTrigger className="-ml-1 hover:bg-sidebar-accent rounded-md transition-colors" />
            <Separator orientation="vertical" className="mr-2 h-4" />
            <Breadcrumb>
              <BreadcrumbList>
                <BreadcrumbItem className="hidden md:block">
                  <BreadcrumbLink
                    href="/"
                    className="flex items-center gap-2 hover:text-foreground transition-colors"
                  >
                    <TrendingUp className="size-4" />
                    Trdinger
                  </BreadcrumbLink>
                </BreadcrumbItem>
                <BreadcrumbSeparator className="hidden md:block" />
                <BreadcrumbItem>
                  <BreadcrumbPage className="flex items-center gap-2 font-semibold">
                    <PageIcon className="size-4" />
                    {pageInfo.title}
                  </BreadcrumbPage>
                </BreadcrumbItem>
              </BreadcrumbList>
            </Breadcrumb>
          </div>

          {/* 右側: ステータス情報 */}
          <div className="flex items-center gap-3">
            <MarketStatus />
            <RealTimeClock />
            <Separator orientation="vertical" className="h-4" />
            <Button
              variant="ghost"
              size="sm"
              className="gap-2 hover:bg-sidebar-accent transition-colors"
              onClick={reloadPage}
            >
              <RefreshCw className="size-4" />
              <span className="hidden sm:inline">Refresh</span>
            </Button>
          </div>
        </div>
      </header>

      {/* ページ説明 */}
      <div className="px-4 py-3 border-b border-sidebar-border/30 bg-sidebar-accent/20">
        <p className="text-sm text-muted-foreground">{pageInfo.description}</p>
      </div>

      {/* メインコンテンツ */}
      <div className="flex flex-1 flex-col gap-6 p-6 bg-gradient-to-br from-background to-sidebar-accent/10">
        {children}
      </div>
    </SidebarInset>
  );
};

export default MainContent;
