import React from "react";
import { BacktestResult } from "@/types/backtest";
import MetricCard from "../MetricCard";
import {
  formatPercentage,
  formatNumber,
  formatCurrency,
  getReturnColor,
  getSharpeColor,
} from "@/utils/formatters";
import {
  ArrowDownRight,
  ArrowUpRight,
  BarChart,
  Calculator,
  Calendar,
  CheckCircle,
  Clock,
  DollarSign,
  Hash,
  Info,
  List,
  Percent,
  ShieldCheck,
  TrendingDown,
  TrendingUp,
  Zap,
} from "lucide-react";

interface OverviewTabProps {
  result: BacktestResult;
}

export default function OverviewTab({ result }: OverviewTabProps) {
  const { performance_metrics: metrics } = result;

  const finalEquity =
    result.initial_capital &&
    metrics.total_return !== undefined &&
    metrics.total_return !== null
      ? result.initial_capital * (1 + metrics.total_return)
      : result.initial_capital;

  const SectionHeader = ({
    title,
    icon: Icon,
  }: {
    title: string;
    icon: React.ElementType;
  }) => (
    <div className="flex items-center mb-4">
      <div className="w-3 h-3 bg-gradient-to-r from-cyan-400 to-blue-500 rounded-full mr-3 shadow-lg shadow-cyan-500/50"></div>
      <h3 className="text-lg font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-500 font-mono tracking-wide flex items-center">
        <Icon className="w-5 h-5 mr-2" />
        {title}
      </h3>
    </div>
  );

  const InfoCard = ({
    label,
    value,
    icon: Icon,
  }: {
    label: string;
    value: string | number;
    icon: React.ElementType;
  }) => (
    <div className="bg-gradient-to-br from-gray-900/80 to-black/60 rounded-lg p-4 border border-purple-500/30 shadow-lg hover:border-purple-400/50 transition-all duration-300 hover:shadow-purple-500/20 relative overflow-hidden">
      <div className="flex items-start space-x-3">
        <Icon className="w-5 h-5 text-purple-400 flex-shrink-0" />
        <div className="flex-1 min-w-0">
          <span className="text-purple-300 text-sm font-mono uppercase tracking-wider">
            {label}
          </span>
          <span className="text-white font-mono text-lg font-semibold block truncate">
            {value}
          </span>
        </div>
      </div>
      <div className="absolute -bottom-4 -right-4 text-purple-500/10">
        <Icon size={64} strokeWidth={1} />
      </div>
    </div>
  );

  return (
    <div className="space-y-8">
      <div className="bg-gradient-to-br from-gray-900/90 to-black/80 rounded-xl p-6 border border-gray-700/50 shadow-2xl backdrop-blur-sm">
        <SectionHeader title="基本情報" icon={Info} />
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <InfoCard
            label="戦略"
            value={result.strategy_name}
            icon={Zap}
          />
          <InfoCard label="シンボル" value={result.symbol} icon={Hash} />
          <InfoCard label="時間軸" value={result.timeframe} icon={Clock} />
          <InfoCard
            label="初期資金"
            value={formatCurrency(result.initial_capital)}
            icon={DollarSign}
          />
        </div>
      </div>

      <div className="bg-gradient-to-br from-gray-900/90 to-black/80 rounded-xl p-6 border border-gray-700/50 shadow-2xl backdrop-blur-sm">
        <SectionHeader title="パフォーマンス指標" icon={BarChart} />
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <MetricCard
            title="総リターン"
            value={formatPercentage(metrics.total_return ?? 0)}
            subtitle={`最終資産: ${formatCurrency(finalEquity)}`}
            color={getReturnColor(metrics.total_return)}
            icon={<TrendingUp className="w-6 h-6" />}
          />
          <MetricCard
            title="シャープレシオ"
            value={formatNumber(metrics.sharpe_ratio ?? 0)}
            subtitle="リスク調整後リターン"
            color={getSharpeColor(metrics.sharpe_ratio)}
            icon={<ShieldCheck className="w-6 h-6" />}
          />
          <MetricCard
            title="最大ドローダウン"
            value={formatPercentage(metrics.max_drawdown ?? 0)}
            subtitle="最大下落率"
            color="red"
            icon={<TrendingDown className="w-6 h-6" />}
          />
          <MetricCard
            title="勝率"
            value={formatPercentage(metrics.win_rate ?? 0)}
            subtitle={`${metrics.winning_trades || 0}勝 / ${
              metrics.losing_trades || 0
            }敗`}
            color={
              metrics.win_rate && metrics.win_rate > 0.5 ? "green" : "yellow"
            }
            icon={<CheckCircle className="w-6 h-6" />}
          />
        </div>
      </div>

      <div className="bg-gradient-to-br from-gray-900/90 to-black/80 rounded-xl p-6 border border-gray-700/50 shadow-2xl backdrop-blur-sm">
        <SectionHeader title="詳細指標" icon={Calculator} />
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <MetricCard
            title="プロフィットファクター"
            value={formatNumber(metrics.profit_factor ?? 0)}
            subtitle="総利益 / 総損失"
            color={
              metrics.profit_factor && metrics.profit_factor > 1
                ? "green"
                : "red"
            }
            icon={<Percent className="w-6 h-6" />}
          />
          <MetricCard
            title="総取引数"
            value={metrics.total_trades || 0}
            subtitle="実行された取引の総数"
            color="blue"
            icon={<List className="w-6 h-6" />}
          />
          <MetricCard
            title="平均利益"
            value={formatCurrency(metrics.avg_win ?? 0)}
            subtitle="勝ちトレードあたり"
            color="green"
            icon={<ArrowUpRight className="w-6 h-6" />}
          />
          <MetricCard
            title="平均損失"
            value={formatCurrency(
              metrics.avg_loss !== undefined && metrics.avg_loss !== null
                ? Math.abs(metrics.avg_loss)
                : 0
            )}
            subtitle="負けトレードあたり"
            color="red"
            icon={<ArrowDownRight className="w-6 h-6" />}
          />
        </div>
      </div>

      <div className="bg-gradient-to-br from-gray-900/90 to-black/80 rounded-xl p-6 border border-gray-700/50 shadow-2xl backdrop-blur-sm">
        <SectionHeader title="バックテスト期間" icon={Calendar} />
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <InfoCard
            label="開始日"
            value={new Date(result.start_date).toLocaleDateString("ja-JP")}
            icon={Calendar}
          />
          <InfoCard
            label="終了日"
            value={new Date(result.end_date).toLocaleDateString("ja-JP")}
            icon={Calendar}
          />
          <InfoCard
            label="手数料率"
            value={formatPercentage(result.commission_rate)}
            icon={Percent}
          />
        </div>
      </div>
    </div>
  );
}
