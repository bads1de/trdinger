/**
 * ホームページコンポーネント
 *
 * アプリケーションのメインランディングページです。
 * 主要機能へのナビゲーションリンクと概要を表示します。
 *
 * 機能:
 * - 戦略定義ページへのリンク
 * - バックテストページへのリンク
 * - 結果分析ページへのリンク
 * - データ管理ページへのリンク
 *
 * @returns ホームページのJSX要素
 */
export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-24">
      {/* ヘッダー部分 - アプリケーション名を表示 */}
      <div className="z-10 max-w-5xl w-full items-center justify-between font-mono text-sm lg:flex">
        <p className="fixed left-0 top-0 flex w-full justify-center border-b border-gray-700 bg-gradient-to-b from-gray-900 pb-6 pt-8 backdrop-blur-2xl dark:border-gray-700 dark:bg-gray-900/30 dark:from-inherit lg:static lg:w-auto  lg:rounded-xl lg:border lg:bg-gray-900 lg:p-4 lg:dark:bg-gray-900/30">
          Trdinger - Trading Strategy Backtest
        </p>
      </div>

      {/* メインタイトル部分 - グラデーション背景付き */}
      <div className="relative flex place-items-center before:absolute before:h-[300px] before:w-[480px] before:-translate-x-1/2 before:translate-y-1/4 before:rounded-full before:bg-gradient-radial before:from-white before:to-transparent before:blur-2xl before:content-[''] after:absolute after:-z-20 after:h-[180px] after:w-[240px] after:translate-x-1/3 after:bg-gradient-conic after:from-sky-200 after:via-blue-200 after:blur-2xl after:content-[''] before:dark:bg-gradient-to-br before:dark:from-transparent before:dark:to-blue-700 before:dark:opacity-10 after:dark:from-sky-900 after:dark:via-[#0141ff] after:dark:opacity-40 before:lg:h-[360px] z-[-1]">
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
