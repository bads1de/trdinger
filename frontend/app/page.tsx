"use client";

import Link from "next/link";
import {
  ArrowRight,
  BarChart3,
  Brain,
  Database,
  ShieldCheck,
  Sparkles,
  TrendingUp,
} from "lucide-react";

/**
 * 0ベース再設計 ホームページ
 *
 * 目的:
 * - プラットフォームの価値提案を明確に伝えるHero
 * - 主要機能（バックテスト/ML/データ）のわかりやすい導線
 * - 信頼性・特徴のハイライト
 * - 既存レイアウト(Navbar/MainContent)と一貫したダークUI
 */
export default function Home() {
  return (
    <main className="min-h-screen">
      {/* Hero */}
      <section className="relative overflow-hidden">
        {/* 背景の微細グラデーション */}
        <div className="pointer-events-none absolute inset-0 -z-10 bg-[radial-gradient(60%_60%_at_50%_0%,rgba(37,99,235,0.2),transparent),radial-gradient(40%_40%_at_80%_20%,rgba(147,51,234,0.12),transparent)]" />
        <div className="mx-auto max-w-7xl px-6 pt-16 pb-10 md:pt-24 md:pb-16">
          <div className="flex flex-col items-start gap-6">
            <span className="inline-flex items-center gap-2 rounded-full border border-sidebar-border/60 bg-sidebar-accent/20 px-3 py-1 text-xs font-medium text-muted-foreground">
              <Sparkles className="h-3.5 w-3.5 text-blue-400" />
              エンタープライズ向けトレーディング研究基盤
            </span>
            <h1 className="text-balance bg-gradient-to-r from-blue-500 via-cyan-400 to-indigo-400 bg-clip-text text-4xl font-extrabold leading-tight tracking-tight text-transparent md:text-5xl">
              仮想通貨トレーディング戦略を
              <br className="hidden md:block" />
              科学的に検証・最適化
            </h1>
            <p className="max-w-2xl text-pretty text-base text-muted-foreground md:text-lg">
              高度なバックテスト基盤、機械学習によるモデル管理、マーケットデータ統合をワンストップで。
              再現性と操作性を両立したワークフローで、研究から実運用までを加速します。
            </p>
            <div className="flex flex-wrap gap-3">
              <Link
                href="/backtest"
                className="inline-flex items-center gap-2 rounded-md bg-primary px-4 py-2 text-sm font-semibold text-primary-foreground shadow hover:bg-primary/90 transition-colors"
              >
                バックテストを開始
                <ArrowRight className="h-4 w-4" />
              </Link>
              <Link
                href="/ml"
                className="inline-flex items-center gap-2 rounded-md border border-border bg-background px-4 py-2 text-sm font-semibold text-foreground hover:bg-accent/10 transition-colors"
              >
                ML管理を見る
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* 機能カード */}
      <section className="mx-auto max-w-7xl px-6 py-8 md:py-12">
        <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
          <FeatureCard
            href="/backtest"
            icon={<BarChart3 className="h-5 w-5 text-purple-400" />}
            title="バックテスト"
            desc="取引履歴レベルの統計、ドローダウン、分布/散布チャートで戦略を精査。"
            cta="結果を見る"
          />
          <FeatureCard
            href="/ml"
            icon={<Brain className="h-5 w-5 text-green-400" />}
            title="ML管理"
            desc="特徴量重要度、単一/アンサンブル、学習/評価を統合管理。"
            cta="モデルを管理"
          />
          <FeatureCard
            href="/data"
            icon={<Database className="h-5 w-5 text-cyan-400" />}
            title="データ管理"
            desc="OHLCV/FR/建玉/恐怖貪欲指数を収集・更新・可視化。"
            cta="データを見る"
          />
        </div>
      </section>

      {/* ハイライト */}
      <section className="mx-auto max-w-7xl px-6 pb-12 md:pb-16">
        <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
          <Highlight
            title="再現性の高い分析"
            icon={<ShieldCheck className="h-5 w-5 text-blue-400" />}
          >
            バージョン固定と一貫した前処理で、実験の再現性を担保。
          </Highlight>
          <Highlight
            title="パフォーマンス重視"
            icon={<TrendingUp className="h-5 w-5 text-emerald-400" />}
          >
            計算効率と可視化応答性を両立し、大規模検証を現実的に。
          </Highlight>
          <Highlight
            title="UI/UX 最適化"
            icon={<Sparkles className="h-5 w-5 text-fuchsia-400" />}
          >
            サイドバー/ヘッダ連携の情報設計で、学習コストを最小化。
          </Highlight>
        </div>
      </section>

      {/* フッター */}
      <footer className="border-t border-sidebar-border/50">
        <div className="mx-auto max-w-7xl px-6 py-8 flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
          <p className="text-xs text-muted-foreground">
            © {new Date().getFullYear()} Trdinger. All rights reserved.
          </p>
          <div className="flex items-center gap-4 text-xs text-muted-foreground">
            <Link
              href="/backtest"
              className="hover:text-foreground transition-colors"
            >
              Backtest
            </Link>
            <Link
              href="/ml"
              className="hover:text-foreground transition-colors"
            >
              ML
            </Link>
            <Link
              href="/data"
              className="hover:text-foreground transition-colors"
            >
              Data
            </Link>
          </div>
        </div>
      </footer>
    </main>
  );
}

/**
 * 機能カード
 */
function FeatureCard({
  href,
  icon,
  title,
  desc,
  cta,
}: {
  href: string;
  icon: React.ReactNode;
  title: string;
  desc: string;
  cta: string;
}) {
  return (
    <Link
      href={href}
      className="group block rounded-xl border border-sidebar-border/60 bg-gradient-to-b from-background to-sidebar-accent/10 p-5 shadow-sm transition-all hover:border-sidebar-border hover:shadow md:p-6"
    >
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-2">
          <div className="rounded-md bg-sidebar-accent/30 p-2">{icon}</div>
          <h3 className="text-lg font-semibold">{title}</h3>
        </div>
        <ArrowRight className="h-4 w-4 text-muted-foreground transition-transform group-hover:translate-x-0.5" />
      </div>
      <p className="mt-3 text-sm text-muted-foreground">{desc}</p>
      <span className="mt-4 inline-flex items-center gap-1 text-sm font-medium text-blue-400">
        {cta}
        <ArrowRight className="h-3.5 w-3.5" />
      </span>
    </Link>
  );
}

/**
 * ハイライト項目
 */
function Highlight({
  title,
  icon,
  children,
}: {
  title: string;
  icon: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-xl border border-sidebar-border/60 bg-sidebar-accent/10 p-5">
      <div className="mb-2 flex items-center gap-2">
        {icon}
        <h4 className="text-sm font-semibold">{title}</h4>
      </div>
      <p className="text-sm text-muted-foreground">{children}</p>
    </div>
  );
}
