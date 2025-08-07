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
  },
  {
    href: "/data",
    label: "Data Management",
    icon: Database,
    description: "価格データを管理",
    color: "text-orange-500",
  },
];



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
      </SidebarHeader>

      <SidebarContent className="px-1">
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
      </SidebarContent>

      <SidebarFooter className="gap-2 p-2">
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
