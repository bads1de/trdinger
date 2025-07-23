/**
 * サイドバーナビゲーションコンポーネント
 *
 * shadcn/uiのSidebarコンポーネントを使用した
 * 左側に配置される折り畳み可能なサイドバーナビゲーションです。
 * レスポンシブデザインとアクティブページのハイライト機能を提供します。
 *
 */

"use client";

import React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Home, BarChart3, Brain, Database, Zap } from "lucide-react";

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

interface NavItem {
  href: string;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
  description: string;
}

const navItems: NavItem[] = [
  {
    href: "/",
    label: "Home",
    icon: Home,
    description: "メインダッシュボード",
  },
  {
    href: "/backtest",
    label: "Backtest",
    icon: BarChart3,
    description: "戦略の有効性を検証",
  },
  {
    href: "/ml",
    label: "ML Management",
    icon: Brain,
    description: "機械学習モデル管理",
  },
  {
    href: "/data",
    label: "Data Management",
    icon: Database,
    description: "価格データを管理",
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
    <Sidebar variant="inset" collapsible="icon">
      <SidebarHeader>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton size="lg" asChild>
              <Link href="/">
                <div className="flex aspect-square size-8 items-center justify-center rounded-lg bg-sidebar-primary text-sidebar-primary-foreground">
                  <Zap className="size-4" />
                </div>
                <div className="grid flex-1 text-left text-sm leading-tight">
                  <span className="truncate font-semibold">Trdinger</span>
                  <span className="truncate text-xs">Trading Platform</span>
                </div>
              </Link>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarHeader>

      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel>Navigation</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
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
                          <div>
                            <div className="font-medium">{item.label}</div>
                            <div className="text-xs text-muted-foreground">
                              {item.description}
                            </div>
                          </div>
                        ),
                      }}
                    >
                      <Link href={item.href}>
                        <Icon className="size-4" />
                        <span>{item.label}</span>
                      </Link>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                );
              })}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>

      <SidebarFooter>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton size="sm" asChild>
              <div className="flex items-center gap-2 text-xs text-sidebar-foreground/70">
                <span>v1.0.0</span>
              </div>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
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
