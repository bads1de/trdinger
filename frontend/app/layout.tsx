/**
 * ルートレイアウトコンポーネント
 *
 * アプリケーション全体の基本的なHTML構造とメタデータを定義します。
 * 全ページで共通のフォント設定とスタイルを適用します。
 *
 */

import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import Navbar from "@/components/navigation/Navbar";
import MainContent from "@/components/navigation/MainContent";
import { Toaster } from "@/components/ui/sonner";

// Google Fontsからインポートしたフォント設定
// ラテン文字セットのみを読み込んでパフォーマンスを最適化
const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Trdinger - Trading Strategy Backtest",
  description: "A trading strategy backtesting service for cryptocurrency",
};

/**
 * ルートレイアウトコンポーネント
 *
 * @param props - コンポーネントのプロパティ
 * @param props.children - 子コンポーネント（各ページのコンテンツ）
 * @returns アプリケーションの基本HTML構造
 */
export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="ja" className="dark">
      <body className={inter.className}>
        <Navbar>
          <MainContent>{children}</MainContent>
        </Navbar>
        <Toaster />
      </body>
    </html>
  );
}
