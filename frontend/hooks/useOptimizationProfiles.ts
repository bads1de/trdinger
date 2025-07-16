import { useState, useEffect, useCallback } from "react";
import { useApiCall } from "./useApiCall";
import { OptimizationProfile } from "@/types/bayesian-optimization";

export const useOptimizationProfiles = (modelType?: string) => {
  const [profiles, setProfiles] = useState<OptimizationProfile[]>([]);
  const [error, setError] = useState<string | null>(null);
  const {
    execute: fetchProfilesApi,
    loading: isLoading,
    reset,
  } = useApiCall<{
    success: boolean;
    profiles: OptimizationProfile[];
    message?: string;
  }>();

  const fetchProfiles = useCallback(() => {
    reset();
    setError(null);

    const params = new URLSearchParams();
    if (modelType) {
      params.append("target_model_type", modelType);
    }

    fetchProfilesApi(`/api/bayesian-optimization/profiles?${params}`, {
      method: "GET",
      onSuccess: (data) => {
        if (data.success) {
          setProfiles(data.profiles || []);
        } else {
          setError(data.message || "プロファイルの取得に失敗しました");
        }
      },
      onError: (errorMessage) => {
        console.error("プロファイル取得エラー:", errorMessage);
        setError("プロファイルの取得中にエラーが発生しました");
      },
    });
  }, [modelType, fetchProfilesApi, reset]);

  useEffect(() => {
    fetchProfiles();
  }, [fetchProfiles]);

  return {
    profiles,
    isLoading,
    error,
    fetchProfiles,
  };
};
