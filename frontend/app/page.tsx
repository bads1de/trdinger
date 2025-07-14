"use client";
import { useState, useEffect, useRef } from "react";

/**
 * ホームページコンポーネント
 *
 * アプリケーションのメインランディングページです。
 * 主要機能へのナビゲーションリンクと概要を表示します。
 * マウスカーソルの位置に応じて動的に変化するグラデーション背景を実装。
 *
 * 機能:
 * - 戦略定義ページへのリンク
 * - バックテストページへのリンク
 * - 結果分析ページへのリンク
 * - データ管理ページへのリンク
 * - マウス追従型動的グラデーション背景
 *
 * @returns ホームページのJSX要素
 */
export default function Home() {
  // マウス座標を正規化した値（0-1の範囲）で管理
  const [mousePosition, setMousePosition] = useState({ x: 0.5, y: 0.5 });
  const animationRef = useRef<number>();

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      // 前のアニメーションフレームをキャンセル（パフォーマンス最適化）
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }

      // requestAnimationFrameを使用してスムーズなアニメーションを実現
      animationRef.current = requestAnimationFrame(() => {
        const x = e.clientX / window.innerWidth;
        const y = e.clientY / window.innerHeight;
        setMousePosition({ x, y });
      });
    };

    // マウスイベントリスナーを追加
    window.addEventListener("mousemove", handleMouseMove);

    // クリーンアップ処理
    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, []);

  // CSS変数としてマウス座標を設定
  const dynamicStyle = {
    "--mouse-x": mousePosition.x,
    "--mouse-y": mousePosition.y,
  } as React.CSSProperties;
  
  return (
    <main
      style={dynamicStyle}
      className="flex min-h-screen flex-col items-center justify-between p-24"
    >
      {/* ヘッダー部分 - アプリケーション名を表示 */}
      <div className="z-10 max-w-5xl w-full items-center justify-between font-mono text-sm lg:flex">
        <p className="fixed left-0 top-0 flex w-full justify-center border-b border-gray-700 bg-gradient-to-b from-gray-900 pb-6 pt-8 backdrop-blur-2xl dark:border-gray-700 dark:bg-gray-900/30 dark:from-inherit lg:static lg:w-auto  lg:rounded-xl lg:border lg:bg-gray-900 lg:p-4 lg:dark:bg-gray-900/30">
          Trdinger - Trading Strategy Backtest
        </p>
      </div>

      {/* メインタイトル部分 - マウス追従型動的グラデーション背景付き */}
      <div
        className="relative flex place-items-center z-[-1]"
        style={
          {
            "--gradient-x": `${(mousePosition.x - 0.5) * 100}px`,
            "--gradient-y": `${(mousePosition.y - 0.5) * 100}px`,
          } as React.CSSProperties
        }
      >
        {/* 放射状グラデーション（マウス追従） */}
        <div
          className="absolute h-[300px] w-[480px] rounded-full bg-gradient-radial from-white to-transparent blur-2xl transition-transform duration-300 ease-out dark:bg-gradient-to-br dark:from-transparent dark:to-blue-700 dark:opacity-10 lg:h-[360px]"
          style={{
            transform: `translate(calc(-50% + var(--gradient-x)), calc(25% + var(--gradient-y)))`,
          }}
        />
        {/* 円錐状グラデーション（マウス追従） */}
        <div
          className="absolute -z-20 h-[180px] w-[240px] bg-gradient-conic from-sky-200 via-blue-200 blur-2xl transition-transform duration-300 ease-out dark:from-sky-900 dark:via-[#0141ff] dark:opacity-40"
          style={{
            transform: `translate(calc(33% + var(--gradient-x) * 0.8), calc(0% + var(--gradient-y) * 0.8))`,
          }}
        />
        <h1 className="text-4xl font-bold">
          仮想通貨トレーディング戦略
          <br />
          バックテストサービス
        </h1>
      </div>

      {/* 機能ナビゲーショングリッド - 4つの主要機能へのリンク */}
      <div className="mb-32 grid text-center lg:max-w-5xl lg:w-full lg:mb-0 lg:grid-cols-4 lg:text-left">
        {/* 戦略ショーケースページへのリンク */}
        <a
          href="/strategies"
          className="group rounded-lg border border-transparent px-5 py-4 transition-colors hover:border-gray-600 hover:bg-gray-800 hover:dark:border-gray-600 hover:dark:bg-gray-800/30"
        >
          <h2 className={`mb-3 text-2xl font-semibold`}>
            戦略ショーケース{" "}
            <span className="inline-block transition-transform group-hover:translate-x-1 motion-reduce:transform-none">
              -&gt;
            </span>
          </h2>
          <p className={`m-0 max-w-[30ch] text-sm opacity-50`}>
            AI生成による30種類の投資戦略を比較・検討
          </p>
        </a>

        {/* バックテストページへのリンク */}
        <a
          href="/backtest"
          className="group rounded-lg border border-transparent px-5 py-4 transition-colors hover:border-gray-600 hover:bg-gray-800 hover:dark:border-gray-600 hover:dark:bg-gray-800/30"
        >
          <h2 className={`mb-3 text-2xl font-semibold`}>
            バックテスト{" "}
            <span className="inline-block transition-transform group-hover:translate-x-1 motion-reduce:transform-none">
              -&gt;
            </span>
          </h2>
          <p className={`m-0 max-w-[30ch] text-sm opacity-50`}>
            過去データを使用して戦略の有効性を検証
          </p>
        </a>

        {/* データ管理ページへのリンク */}
        <a
          href="/data"
          className="group rounded-lg border border-transparent px-5 py-4 transition-colors hover:border-gray-600 hover:bg-gray-800 hover:dark:border-gray-600 hover:dark:bg-gray-800/30"
        >
          <h2 className={`mb-3 text-2xl font-semibold`}>
            データ管理{" "}
            <span className="inline-block transition-transform group-hover:translate-x-1 motion-reduce:transform-none">
              -&gt;
            </span>
          </h2>
          <p className={`m-0 max-w-[30ch] text-sm opacity-50 text-balance`}>
            仮想通貨の価格データを管理・更新
          </p>
        </a>
      </div>
    </main>
  );
}
