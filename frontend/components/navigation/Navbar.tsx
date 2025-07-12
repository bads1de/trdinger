/**
 * サイドバーナビゲーションコンポーネント
 *
 * 左側に配置される折り畳み可能なサイドバーナビゲーションです。
 * レスポンシブデザインとアクティブページのハイライト機能を提供します。
 *
 */

"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";

interface NavItem {
  href: string;
  label: string;
  icon: string;
  description: string;
}

const navItems: NavItem[] = [
  {
    href: "/",
    label: "Home",
    icon: "🏠",
    description: "メインダッシュボード",
  },
  {
    href: "/backtest",
    label: "Backtest",
    icon: "📊",
    description: "戦略の有効性を検証",
  },
  {
    href: "/ml",
    label: "ML Management",
    icon: "🧠",
    description: "機械学習モデル管理",
  },
  {
    href: "/data",
    label: "Data Management",
    icon: "📈",
    description: "価格データを管理",
  },
];

/**
 * サイドバーナビゲーションコンポーネント
 */
const Navbar: React.FC = () => {
  const pathname = usePathname();
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  /**
   * アクティブページかどうかを判定
   */
  const isActivePage = (href: string): boolean => {
    if (href === "/") {
      return pathname === "/";
    }
    return pathname.startsWith(href);
  };

  /**
   * サイドバーの折り畳み切り替え
   */
  const toggleCollapse = () => {
    setIsCollapsed(!isCollapsed);
  };

  /**
   * モバイルメニューの開閉
   */
  const toggleMobileMenu = () => {
    setIsMobileMenuOpen(!isMobileMenuOpen);
  };

  /**
   * ローカルストレージから折り畳み状態を復元
   */
  useEffect(() => {
    const savedCollapsed = localStorage.getItem("sidebar-collapsed");
    if (savedCollapsed !== null) {
      setIsCollapsed(JSON.parse(savedCollapsed));
    }
  }, []);

  /**
   * 折り畳み状態をローカルストレージに保存
   */
  useEffect(() => {
    localStorage.setItem("sidebar-collapsed", JSON.stringify(isCollapsed));
    // カスタムイベントを発火してメインコンテンツに通知
    window.dispatchEvent(new Event("sidebar-toggle"));
  }, [isCollapsed]);

  return (
    <>
      {/* デスクトップサイドバー */}
      <aside
        className={`
          fixed left-0 top-0 h-full bg-black border-r border-gray-700 z-50
          transition-all duration-300 ease-in-out
          ${isCollapsed ? "w-16" : "w-64"}
          hidden md:block
        `}
      >
        {/* ヘッダー部分 */}
        <div className="flex items-center justify-between p-4 border-b border-gray-700">
          {!isCollapsed && (
            <Link href="/" className="flex items-center space-x-3">
              <div className="text-2xl">⚡</div>
              <div className="text-xl font-bold text-white">Trdinger</div>
            </Link>
          )}
          {isCollapsed && (
            <Link href="/" className="flex items-center justify-center w-full">
              <div className="text-2xl">⚡</div>
            </Link>
          )}
          <button
            onClick={toggleCollapse}
            className="text-gray-300 hover:text-white p-1 rounded-lg hover:bg-gray-800 transition-colors"
            aria-label={isCollapsed ? "Expand sidebar" : "Collapse sidebar"}
          >
            <svg
              className="w-5 h-5"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              {isCollapsed ? (
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 5l7 7-7 7"
                />
              ) : (
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M15 19l-7-7 7-7"
                />
              )}
            </svg>
          </button>
        </div>

        {/* ナビゲーションメニュー */}
        <nav className="mt-4 px-2">
          <div className="space-y-1">
            {navItems.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                className={`
                  flex items-center px-3 py-3 rounded-lg text-sm font-medium
                  transition-all duration-200 group relative
                  ${
                    isActivePage(item.href)
                      ? "bg-blue-600 text-white shadow-lg"
                      : "text-gray-300 hover:text-white hover:bg-gray-800"
                  }
                `}
                title={isCollapsed ? item.label : ""}
              >
                <span className="text-lg flex-shrink-0">{item.icon}</span>
                {!isCollapsed && (
                  <div className="ml-3">
                    <div>{item.label}</div>
                    <div className="text-xs text-gray-400 mt-1">
                      {item.description}
                    </div>
                  </div>
                )}
                {/* 折り畳み時のツールチップ */}
                {isCollapsed && (
                  <div className="absolute left-full ml-2 px-2 py-1 bg-gray-800 text-white text-xs rounded-lg opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none whitespace-nowrap z-50">
                    {item.label}
                  </div>
                )}
              </Link>
            ))}
          </div>
        </nav>
      </aside>

      {/* モバイル用オーバーレイ */}
      {isMobileMenuOpen && (
        <div
          className="fixed inset-0 bg-black bg-opacity-50 z-40 md:hidden"
          onClick={() => setIsMobileMenuOpen(false)}
        />
      )}

      {/* モバイル用サイドバー */}
      <aside
        className={`
          fixed left-0 top-0 h-full w-64 bg-black border-r border-gray-700 z-50
          transform transition-transform duration-300 ease-in-out md:hidden
          ${isMobileMenuOpen ? "translate-x-0" : "-translate-x-full"}
        `}
      >
        {/* モバイルヘッダー */}
        <div className="flex items-center justify-between p-4 border-b border-gray-700">
          <Link href="/" className="flex items-center space-x-3">
            <div className="text-2xl">⚡</div>
            <div className="text-xl font-bold text-white">Trdinger</div>
          </Link>
          <button
            onClick={toggleMobileMenu}
            className="text-gray-300 hover:text-white p-1 rounded-lg hover:bg-gray-800 transition-colors"
            aria-label="Close menu"
          >
            <svg
              className="w-6 h-6"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </button>
        </div>

        {/* モバイルナビゲーション */}
        <nav className="mt-4 px-2">
          <div className="space-y-1">
            {navItems.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                onClick={() => setIsMobileMenuOpen(false)}
                className={`
                  flex items-center px-3 py-3 rounded-lg text-sm font-medium
                  transition-all duration-200
                  ${
                    isActivePage(item.href)
                      ? "bg-blue-600 text-white"
                      : "text-gray-300 hover:text-white hover:bg-gray-800"
                  }
                `}
              >
                <span className="text-lg">{item.icon}</span>
                <div className="ml-3">
                  <div>{item.label}</div>
                  <div className="text-xs text-gray-400 mt-1">
                    {item.description}
                  </div>
                </div>
              </Link>
            ))}
          </div>
        </nav>
      </aside>

      {/* モバイル用メニューボタン */}
      <button
        onClick={toggleMobileMenu}
        className="fixed top-4 left-4 z-50 md:hidden bg-black text-gray-300 hover:text-white p-2 rounded-lg hover:bg-gray-800 transition-colors border border-gray-700"
        aria-label="Open menu"
      >
        <svg
          className="w-6 h-6"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M4 6h16M4 12h16M4 18h16"
          />
        </svg>
      </button>
    </>
  );
};

export default Navbar;
