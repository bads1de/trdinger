/**
 * 改良されたサイドバーナビゲーションコンポーネント
 *
 * トレーディングプラットフォーム向けに最適化された
 * モダンで視覚的に魅力的なサイドバーナビゲーション。
 * リアルタイム情報、アニメーション、改善されたUXを提供します。
 *
 */

"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Home,
  BarChart3,
  Brain,
  Database,
  TrendingUp,
  Settings,
  Bell,
  Wifi,
  WifiOff,
  User,
  ChevronDown,
  Activity,
} from "lucide-react";

import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarProvider,
  SidebarRail,
} from "@/components/ui/sidebar";
import { Badge } from "@/components/ui/badge";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

interface NavItem {
  href: string;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
  description: string;
  badge?: string;
  color?: string;
}

const navItems: NavItem[] = [
  {
    href: "/",
    label: "Dashboard",
    icon: Home,
    description: "メインダッシュボード",
    color: "text-blue-500",
  },
  {
    href: "/backtest",
    label: "Backtest",
    icon: BarChart3,
    description: "戦略の有効性を検証",
    color: "text-purple-500",
  },
  {
    href: "/ml",
    label: "ML Management",
    icon: Brain,
    description: "機械学習モデル管理",
    color: "text-green-500",
    badge: "AI",
  },
  {
    href: "/data",
    label: "Data Management",
    icon: Database,
    description: "価格データを管理",
    color: "text-orange-500",
  },
];

// モックデータ - 実際の実装では適切なデータソースから取得
const mockPortfolioValue = "¥1,234,567";
const mockPnL = "+2.34%";
const mockNotificationCount = 3;

/**
 * アクティブページかどうかを判定
 */
const isActivePage = (href: string, pathname: string): boolean => {
  if (href === "/") {
    return pathname === "/";
  }
  return pathname.startsWith(href);
};

/**
 * 接続ステータスコンポーネント
 */
const ConnectionStatus: React.FC = () => {
  const [isConnected, setIsConnected] = useState(true);

  useEffect(() => {
    // 実際の実装では、WebSocketやAPIの接続状態を監視
    const interval = setInterval(() => {
      setIsConnected(Math.random() > 0.1); // 90%の確率で接続状態
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex items-center gap-2 px-2 py-1 rounded-md bg-sidebar-accent/50">
      {isConnected ? (
        <>
          <Wifi className="size-3 text-green-500" />
          <span className="text-xs text-green-500 font-medium">Connected</span>
        </>
      ) : (
        <>
          <WifiOff className="size-3 text-red-500" />
          <span className="text-xs text-red-500 font-medium">Disconnected</span>
        </>
      )}
    </div>
  );
};

/**
 * サイドバーナビゲーションコンポーネント
 */
const SidebarNavigation: React.FC = () => {
  const pathname = usePathname();

  return (
    <Sidebar
      variant="inset"
      collapsible="icon"
      className="border-r border-sidebar-border/50"
    >
      <SidebarHeader className="gap-4 p-4">
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton
              size="lg"
              asChild
              className="gap-3 hover:bg-sidebar-accent/80 transition-colors"
            >
              <Link href="/" className="flex items-center gap-3">
                <div className="flex aspect-square size-10 items-center justify-center rounded-xl bg-gradient-to-br from-blue-500 to-purple-600 text-white shadow-lg">
                  <TrendingUp className="size-5" />
                </div>
                <div className="flex flex-col gap-0.5">
                  <span className="truncate font-bold text-lg bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                    Trdinger
                  </span>
                  <span className="truncate text-xs text-muted-foreground font-medium">
                    Trading Platform
                  </span>
                </div>
              </Link>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>

        {/* 接続ステータス */}
        <div className="group-data-[collapsible=icon]:hidden">
          <ConnectionStatus />
        </div>
      </SidebarHeader>

      <SidebarContent className="px-2">
        {/* ポートフォリオ情報 */}
        <SidebarGroup className="group-data-[collapsible=icon]:hidden">
          <SidebarGroupLabel className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
            Portfolio
          </SidebarGroupLabel>
          <SidebarGroupContent>
            <div className="p-3 rounded-lg bg-gradient-to-r from-green-50 to-blue-50 dark:from-green-950/20 dark:to-blue-950/20 border border-green-200/50 dark:border-green-800/50">
              <div className="flex items-center justify-between mb-1">
                <span className="text-sm font-medium text-muted-foreground">
                  Total Value
                </span>
                <Activity className="size-4 text-green-500" />
              </div>
              <div className="text-lg font-bold text-foreground">
                {mockPortfolioValue}
              </div>
              <div className="text-sm font-medium text-green-600 dark:text-green-400">
                {mockPnL}
              </div>
            </div>
          </SidebarGroupContent>
        </SidebarGroup>

        {/* ナビゲーション */}
        <SidebarGroup>
          <SidebarGroupLabel className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
            Navigation
          </SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu className="gap-1">
              {navItems.map((item) => {
                const Icon = item.icon;
                const isActive = isActivePage(item.href, pathname);

                return (
                  <SidebarMenuItem key={item.href}>
                    <SidebarMenuButton
                      asChild
                      isActive={isActive}
                      tooltip={{
                        children: (
                          <div className="flex flex-col gap-1">
                            <div className="font-medium">{item.label}</div>
                            <div className="text-xs text-muted-foreground">
                              {item.description}
                            </div>
                          </div>
                        ),
                      }}
                      className={`
                        gap-3 h-11 px-3 rounded-lg transition-all duration-200 
                        hover:bg-sidebar-accent hover:shadow-sm
                        group-data-[collapsible=icon]:gap-0 group-data-[collapsible=icon]:justify-center
                        ${
                          isActive
                            ? "bg-sidebar-accent shadow-sm border border-sidebar-border/50"
                            : ""
                        }
                      `}
                    >
                      <Link
                        href={item.href}
                        className="flex items-center gap-3 w-full group-data-[collapsible=icon]:gap-0"
                      >
                        <Icon
                          className={`size-5 transition-colors group-data-[collapsible=icon]:size-6 ${
                            item.color || "text-muted-foreground"
                          }`}
                        />
                        <div className="flex items-center justify-between w-full group-data-[collapsible=icon]:hidden">
                          <span className="text-sm font-medium">
                            {item.label}
                          </span>
                          {item.badge && (
                            <Badge
                              variant="secondary"
                              className="text-xs px-1.5 py-0.5 h-5"
                            >
                              {item.badge}
                            </Badge>
                          )}
                        </div>
                      </Link>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                );
              })}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>

        {/* 通知セクション */}
        <SidebarGroup className="group-data-[collapsible=icon]:hidden">
          <SidebarGroupLabel className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
            Notifications
          </SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              <SidebarMenuItem>
                <SidebarMenuButton className="gap-3 h-10 px-3 rounded-lg hover:bg-sidebar-accent transition-colors">
                  <div className="flex items-center gap-3 w-full">
                    <div className="relative">
                      <Bell className="size-4 text-muted-foreground" />
                      {mockNotificationCount > 0 && (
                        <Badge className="absolute -top-1 -right-1 h-4 w-4 p-0 text-xs flex items-center justify-center bg-red-500">
                          {mockNotificationCount}
                        </Badge>
                      )}
                    </div>
                    <span className="text-sm font-medium">Alerts</span>
                  </div>
                </SidebarMenuButton>
              </SidebarMenuItem>
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>

      <SidebarFooter className="gap-2 p-2">
        {/* ユーザープロフィール */}
        <SidebarMenu>
          <SidebarMenuItem>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <SidebarMenuButton
                  size="lg"
                  className="gap-3 h-12 px-3 rounded-lg hover:bg-sidebar-accent transition-colors group-data-[collapsible=icon]:justify-center"
                >
                  <div className="flex items-center gap-3 w-full group-data-[collapsible=icon]:gap-0">
                    <Avatar className="h-8 w-8">
                      <AvatarImage src="/placeholder-avatar.jpg" alt="User" />
                      <AvatarFallback className="bg-gradient-to-br from-blue-500 to-purple-600 text-white text-sm font-semibold">
                        TR
                      </AvatarFallback>
                    </Avatar>
                    <div className="flex items-center justify-between w-full group-data-[collapsible=icon]:hidden">
                      <div className="flex flex-col gap-0.5">
                        <span className="text-sm font-medium">Trader</span>
                        <span className="text-xs text-muted-foreground">
                          trader@example.com
                        </span>
                      </div>
                      <ChevronDown className="size-4 text-muted-foreground" />
                    </div>
                  </div>
                </SidebarMenuButton>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-56">
                <DropdownMenuLabel>My Account</DropdownMenuLabel>
                <DropdownMenuSeparator />
                <DropdownMenuItem>
                  <User className="mr-2 h-4 w-4" />
                  <span>Profile</span>
                </DropdownMenuItem>
                <DropdownMenuItem>
                  <Settings className="mr-2 h-4 w-4" />
                  <span>Settings</span>
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuItem className="text-red-600">
                  <span>Log out</span>
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </SidebarMenuItem>
        </SidebarMenu>

        {/* バージョン情報 */}
        <div className="px-3 py-2 group-data-[collapsible=icon]:hidden">
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span>Version</span>
            <span className="font-mono bg-sidebar-accent px-1.5 py-0.5 rounded">
              v1.0.0
            </span>
          </div>
        </div>
      </SidebarFooter>

      <SidebarRail />
    </Sidebar>
  );
};

/**
 * メインのNavbarコンポーネント（SidebarProviderでラップ）
 */
const Navbar: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  return (
    <SidebarProvider>
      <SidebarNavigation />
      {children}
    </SidebarProvider>
  );
};

export default Navbar;
