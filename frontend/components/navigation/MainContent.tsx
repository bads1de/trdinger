/**
 * メインコンテンツラッパーコンポーネント
 *
 * サイドバーの折り畳み状態に応じてレイアウトを調整します。
 *
 */

"use client";

import React, { useState, useEffect } from "react";

interface MainContentProps {
  children: React.ReactNode;
}

/**
 * メインコンテンツラッパーコンポーネント
 */
const MainContent: React.FC<MainContentProps> = ({ children }) => {
  const [isCollapsed, setIsCollapsed] = useState(false);

  /**
   * ローカルストレージから折り畳み状態を監視
   */
  useEffect(() => {
    const checkCollapsedState = () => {
      const savedCollapsed = localStorage.getItem("sidebar-collapsed");
      if (savedCollapsed !== null) {
        setIsCollapsed(JSON.parse(savedCollapsed));
      }
    };

    // 初期状態をチェック
    checkCollapsedState();

    // ストレージの変更を監視
    const handleStorageChange = () => {
      checkCollapsedState();
    };

    window.addEventListener("storage", handleStorageChange);

    // カスタムイベントも監視（同一タブ内での変更）
    const handleCustomEvent = () => {
      checkCollapsedState();
    };

    window.addEventListener("sidebar-toggle", handleCustomEvent);

    return () => {
      window.removeEventListener("storage", handleStorageChange);
      window.removeEventListener("sidebar-toggle", handleCustomEvent);
    };
  }, []);

  return (
    <div
      className={`
        transition-all duration-300 ease-in-out
        ${isCollapsed ? "md:ml-16" : "md:ml-64"}
        ml-0
      `}
    >
      {children}
    </div>
  );
};

export default MainContent;
