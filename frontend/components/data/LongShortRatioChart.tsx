import React, { useMemo } from 'react';
import {
  ComposedChart,
  Line,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Area
} from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Loader2, RefreshCw, Download } from 'lucide-react';
import { LongShortRatioData } from '@/types/long-short-ratio';
import { format } from 'date-fns';

interface LongShortRatioChartProps {
  data: LongShortRatioData[];
  loading: boolean;
  collecting: boolean;
  onRefresh: () => void;
  onCollect: (mode: "incremental" | "historical") => void;
  period: string;
  symbol: string;
}

export const LongShortRatioChart: React.FC<LongShortRatioChartProps> = ({
  data,
  loading,
  collecting,
  onRefresh,
  onCollect,
  period,
  symbol
}) => {
  // チャート用にデータを整形
  const chartData = useMemo(() => {
    return [...data].sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime())
      .map(item => ({
        ...item,
        displayTime: format(new Date(item.timestamp), 'MM/dd HH:mm'),
        buyPercent: (item.buy_ratio * 100).toFixed(2),
        sellPercent: (item.sell_ratio * 100).toFixed(2),
        ratio: item.ls_ratio?.toFixed(4)
      }));
  }, [data]);

  return (
    <Card className="w-full">
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium">
          Long/Short Ratio ({symbol} - {period})
        </CardTitle>
        <div className="flex space-x-2">
          <Button
            variant="outline"
            size="sm"
            onClick={onRefresh}
            disabled={loading}
          >
            {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <RefreshCw className="h-4 w-4" />}
          </Button>
          <Button
            variant="default"
            size="sm"
            onClick={() => onCollect("incremental")}
            disabled={collecting}
          >
            {collecting ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Download className="mr-2 h-4 w-4" />}
            Collect Latest
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        <div className="h-[400px] w-full mt-4">
            {chartData.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                    <XAxis 
                        dataKey="displayTime" 
                        tick={{ fontSize: 10 }}
                        minTickGap={30}
                    />
                    <YAxis 
                        yAxisId="left" 
                        domain={[0, 1]} 
                        tickFormatter={(value) => `${(value * 100).toFixed(0)}%`} 
                    />
                    <YAxis 
                        yAxisId="right" 
                        orientation="right" 
                        domain={['auto', 'auto']}
                    />
                    <Tooltip 
                        contentStyle={{ backgroundColor: '#1f2937', border: 'none', color: '#f3f4f6' }}
                        labelStyle={{ color: '#9ca3af' }}
                    />
                    <Legend />
                    
                    {/* Buy Ratio Area (Stack) */}
                    <Area 
                        yAxisId="left"
                        type="monotone" 
                        dataKey="buy_ratio" 
                        stackId="1" 
                        stroke="#10b981" 
                        fill="#10b981" 
                        fillOpacity={0.6}
                        name="Longs" 
                    />
                    <Area 
                        yAxisId="left"
                        type="monotone" 
                        dataKey="sell_ratio" 
                        stackId="1" 
                        stroke="#ef4444" 
                        fill="#ef4444" 
                        fillOpacity={0.6}
                        name="Shorts" 
                    />

                    {/* LS Ratio Line */}
                    <Line 
                        yAxisId="right"
                        type="monotone" 
                        dataKey="ratio" 
                        stroke="#fbbf24" 
                        strokeWidth={2}
                        dot={false}
                        name="L/S Ratio" 
                    />
                    </ComposedChart>
                </ResponsiveContainer>
            ) : (
                <div className="flex h-full items-center justify-center text-muted-foreground">
                    No data available
                </div>
            )}
        </div>
      </CardContent>
    </Card>
  );
};
