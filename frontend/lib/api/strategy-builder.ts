/**
 * ストラテジービルダーAPIクライアント
 * 
 * ストラテジービルダーAPIとの通信を担当する関数群
 */

// 型定義
export interface IndicatorInfo {
  type: string;
  name: string;
  description: string;
  parameters: Array<{
    name: string;
    type: string;
    default: any;
    min?: number;
    max?: number;
    description: string;
  }>;
  data_sources: string[];
}

export interface IndicatorCategories {
  [category: string]: IndicatorInfo[];
}

export interface UserStrategy {
  id: number;
  name: string;
  description?: string;
  strategy_config: any;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

export interface StrategyCreateRequest {
  name: string;
  description?: string;
  strategy_config: any;
}

export interface StrategyUpdateRequest {
  name?: string;
  description?: string;
  strategy_config?: any;
}

export interface StrategyValidateRequest {
  strategy_config: any;
}

export interface ValidationResult {
  is_valid: boolean;
  errors: string[];
}

export interface ApiResponse<T = any> {
  success: boolean;
  data: T;
  message: string;
}

// APIエラークラス
export class StrategyBuilderApiError extends Error {
  constructor(
    message: string,
    public status?: number,
    public response?: any
  ) {
    super(message);
    this.name = "StrategyBuilderApiError";
  }
}

// 共通のAPIリクエスト関数
async function apiRequest<T = any>(
  endpoint: string,
  options: RequestInit = {}
): Promise<ApiResponse<T>> {
  const url = `/api/strategy-builder${endpoint}`;
  
  const defaultOptions: RequestInit = {
    headers: {
      "Content-Type": "application/json",
      ...options.headers,
    },
    ...options,
  };

  try {
    const response = await fetch(url, defaultOptions);
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new StrategyBuilderApiError(
        errorData.detail || `HTTP ${response.status}: ${response.statusText}`,
        response.status,
        errorData
      );
    }

    const data = await response.json();
    return data;
  } catch (error) {
    if (error instanceof StrategyBuilderApiError) {
      throw error;
    }
    
    throw new StrategyBuilderApiError(
      error instanceof Error ? error.message : "ネットワークエラーが発生しました"
    );
  }
}

/**
 * 利用可能なテクニカル指標の一覧を取得
 */
export async function getAvailableIndicators(): Promise<IndicatorCategories> {
  const response = await apiRequest<{ categories: IndicatorCategories }>("/indicators");
  return response.data.categories;
}

/**
 * 戦略設定の妥当性を検証
 */
export async function validateStrategy(
  strategyConfig: any
): Promise<ValidationResult> {
  const response = await apiRequest<ValidationResult>("/validate", {
    method: "POST",
    body: JSON.stringify({ strategy_config: strategyConfig }),
  });
  return response.data;
}

/**
 * 戦略を保存
 */
export async function saveStrategy(
  request: StrategyCreateRequest
): Promise<UserStrategy> {
  const response = await apiRequest<UserStrategy>("/save", {
    method: "POST",
    body: JSON.stringify(request),
  });
  return response.data;
}

/**
 * 保存済み戦略の一覧を取得
 */
export async function getStrategies(
  activeOnly: boolean = true,
  limit?: number
): Promise<{ strategies: UserStrategy[]; count: number }> {
  const params = new URLSearchParams();
  params.append("active_only", String(activeOnly));
  if (limit) {
    params.append("limit", String(limit));
  }

  const response = await apiRequest<{ strategies: UserStrategy[]; count: number }>(
    `/strategies?${params.toString()}`
  );
  return response.data;
}

/**
 * 戦略の詳細を取得
 */
export async function getStrategy(strategyId: number): Promise<UserStrategy> {
  const response = await apiRequest<UserStrategy>(`/strategies/${strategyId}`);
  return response.data;
}

/**
 * 戦略を更新
 */
export async function updateStrategy(
  strategyId: number,
  request: StrategyUpdateRequest
): Promise<UserStrategy> {
  const response = await apiRequest<UserStrategy>(`/strategies/${strategyId}`, {
    method: "PUT",
    body: JSON.stringify(request),
  });
  return response.data;
}

/**
 * 戦略を削除（論理削除）
 */
export async function deleteStrategy(
  strategyId: number
): Promise<{ deleted: boolean; strategy_id: number }> {
  const response = await apiRequest<{ deleted: boolean; strategy_id: number }>(
    `/strategies/${strategyId}`,
    {
      method: "DELETE",
    }
  );
  return response.data;
}

// React hooks用のユーティリティ関数

/**
 * 指標一覧取得用のカスタムフック用関数
 */
export function useIndicators() {
  return {
    fetchIndicators: getAvailableIndicators,
  };
}

/**
 * 戦略管理用のカスタムフック用関数
 */
export function useStrategies() {
  return {
    fetchStrategies: getStrategies,
    fetchStrategy: getStrategy,
    createStrategy: saveStrategy,
    updateStrategy,
    deleteStrategy,
    validateStrategy,
  };
}

// エラーハンドリング用のユーティリティ

/**
 * APIエラーを人間が読みやすい形式に変換
 */
export function formatApiError(error: unknown): string {
  if (error instanceof StrategyBuilderApiError) {
    return error.message;
  }
  
  if (error instanceof Error) {
    return error.message;
  }
  
  return "不明なエラーが発生しました";
}

/**
 * バリデーションエラーを整形
 */
export function formatValidationErrors(errors: string[]): string {
  if (errors.length === 0) {
    return "";
  }
  
  if (errors.length === 1) {
    return errors[0];
  }
  
  return `以下の問題があります:\n${errors.map(error => `• ${error}`).join("\n")}`;
}

// デバッグ用の関数

/**
 * API呼び出しをログ出力付きで実行（開発環境のみ）
 */
export async function debugApiCall<T>(
  apiFunction: () => Promise<T>,
  functionName: string
): Promise<T> {
  if (process.env.NODE_ENV === "development") {
    console.log(`[StrategyBuilder API] ${functionName} - 開始`);
    const startTime = Date.now();
    
    try {
      const result = await apiFunction();
      const duration = Date.now() - startTime;
      console.log(`[StrategyBuilder API] ${functionName} - 成功 (${duration}ms)`, result);
      return result;
    } catch (error) {
      const duration = Date.now() - startTime;
      console.error(`[StrategyBuilder API] ${functionName} - エラー (${duration}ms)`, error);
      throw error;
    }
  } else {
    return apiFunction();
  }
}

// 型ガード関数

/**
 * UserStrategyオブジェクトかどうかを判定
 */
export function isUserStrategy(obj: any): obj is UserStrategy {
  return (
    obj &&
    typeof obj.id === "number" &&
    typeof obj.name === "string" &&
    typeof obj.strategy_config === "object" &&
    typeof obj.is_active === "boolean" &&
    typeof obj.created_at === "string" &&
    typeof obj.updated_at === "string"
  );
}

/**
 * IndicatorInfoオブジェクトかどうかを判定
 */
export function isIndicatorInfo(obj: any): obj is IndicatorInfo {
  return (
    obj &&
    typeof obj.type === "string" &&
    typeof obj.name === "string" &&
    typeof obj.description === "string" &&
    Array.isArray(obj.parameters) &&
    Array.isArray(obj.data_sources)
  );
}

// キャッシュ機能（オプション）

interface CacheEntry<T> {
  data: T;
  timestamp: number;
  ttl: number;
}

class ApiCache {
  private cache = new Map<string, CacheEntry<any>>();

  set<T>(key: string, data: T, ttlMs: number = 5 * 60 * 1000): void {
    this.cache.set(key, {
      data,
      timestamp: Date.now(),
      ttl: ttlMs,
    });
  }

  get<T>(key: string): T | null {
    const entry = this.cache.get(key);
    
    if (!entry) {
      return null;
    }

    if (Date.now() - entry.timestamp > entry.ttl) {
      this.cache.delete(key);
      return null;
    }

    return entry.data;
  }

  clear(): void {
    this.cache.clear();
  }
}

export const apiCache = new ApiCache();

/**
 * キャッシュ付きで指標一覧を取得
 */
export async function getCachedIndicators(): Promise<IndicatorCategories> {
  const cacheKey = "indicators";
  const cached = apiCache.get<IndicatorCategories>(cacheKey);
  
  if (cached) {
    return cached;
  }

  const indicators = await getAvailableIndicators();
  apiCache.set(cacheKey, indicators, 10 * 60 * 1000); // 10分間キャッシュ
  
  return indicators;
}
