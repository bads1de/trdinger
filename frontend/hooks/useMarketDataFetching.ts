import { useParameterizedDataFetching } from "./useDataFetching";
import { useSetLimit } from "@/utils/hookUtils";

/**
 * 市場データ取得フックの共通パラメータ
 */
export interface MarketDataParams {
  symbol: string;
  limit: number;
  timeframe?: string;
  period?: string;
  start_date?: string;
  end_date?: string;
}

/**
 * 市場データ取得フックの共通結果
 */
export interface MarketDataFetchingResult<TData> {
  data: TData[];
  loading: boolean;
  error: string | null;
  refetch: () => Promise<void>;
  setLimit: (limit: number) => void;
  setParams: (params: Partial<MarketDataParams>) => void;
  limit: number;
}

/**
 * 市場データ取得フック（共通化）
 *
 * 市場データ（OHLCV、ファンディングレート、オープンインタレスト、Long/Short Ratio）の
 * 取得ロジックを共通化します。エンドポイントとデータ変換関数のみを指定します。
 *
 * @typeParam TData - 取得するデータの型
 * @param {string} endpoint - APIエンドポイント
 * @param {MarketDataParams} initialParams - 初期パラメータ
 * @param {Object} options - 追加オプション
 * @param {Function} options.transform - レスポンス変換関数
 * @param {string} options.dataPath - データパス（transformと排他）
 * @param {any[]} options.dependencies - 依存関係
 * @param {string} options.errorMessage - エラーメッセージ
 * @returns {MarketDataFetchingResult<TData>} 市場データ取得関連の状態と操作関数
 */
export const useMarketDataFetching = <TData = any>(
  endpoint: string,
  initialParams: MarketDataParams,
  options: {
    transform?: (response: any) => TData[];
    dataPath?: string;
    dependencies?: any[];
    errorMessage?: string;
  } = {}
) => {
  const { data, loading, error, params, setParams, refetch } =
    useParameterizedDataFetching<TData, MarketDataParams>(
      endpoint,
      initialParams,
      {
        transform: options.transform,
        dataPath: options.dataPath,
        dependencies: options.dependencies,
        errorMessage: options.errorMessage,
      }
    );

  const setLimit = useSetLimit(setParams);

  return {
    /** 市場データ */
    data,
    /** ローディング状態 */
    loading,
    /** エラーメッセージ */
    error,
    /** データを再取得する関数 */
    refetch,
    /** 取得数を設定する関数 */
    setLimit,
    /** パラメータを設定する関数 */
    setParams,
    /** 現在の取得数 */
    limit: params.limit,
  };
};