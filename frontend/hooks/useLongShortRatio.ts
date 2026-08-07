import { useCallback } from "react";
import { LongShortRatioData } from "@/types/long-short-ratio";
import { useMarketDataFetching } from "./useMarketDataFetching";
import { useApiCall } from "./useApiCall";

/**
 * Long/Short Ratio データ管理フック
 */
export const useLongShortRatio = (symbol: string, period: string, initialLimit = 100) => {
  // データ取得（共通フックを利用）
  const {
    data,
    loading,
    error,
    refetch,
    setLimit,
    setParams,
    limit,
  } = useMarketDataFetching<LongShortRatioData>(
    "/api/long-short-ratio/",
    { symbol, period, limit: initialLimit },
    {
      transform: (response: any) => {
        // APIは配列を直接返すか、ラップして返すか確認が必要だが、
        // market_data.pyの実装を見ると [repository.to_dict(record) for ...] とリストを返している。
        const list = Array.isArray(response) ? response : (response.data || []);
        
        return list.map((item: any) => ({
          symbol: item.symbol,
          period: item.period,
          buy_ratio: Number(item.buy_ratio),
          sell_ratio: Number(item.sell_ratio),
          timestamp: item.timestamp,
          ls_ratio: Number(item.buy_ratio) / (Number(item.sell_ratio) || 1) // ゼロ除算回避
        }));
      },
      dependencies: [symbol, period],
      errorMessage: "Long/Short Ratioデータの取得に失敗しました",
    }
  );

  const setPeriod = useCallback(
    (newPeriod: string) => {
      setParams({ period: newPeriod });
    },
    [setParams]
  );

  // データ収集リクエスト
  const { execute, loading: collecting } = useApiCall();

  const collectData = useCallback(async (mode: "incremental" | "historical" = "incremental") => {
    const queryParams = new URLSearchParams({
      symbol,
      period,
      mode
    });
    
    await execute(`/api/long-short-ratio/collect?${queryParams}`, {
      method: "POST",
      onSuccess: () => {
        // 少し待ってからリフレッシュ
        setTimeout(refetch, 2000);
      }
    });
  }, [symbol, period, execute, refetch]);

  return {
    data,
    loading,
    collecting,
    error,
    refetch,
    collectData,
    setLimit,
    setPeriod,
    limit,
    period
  };
};