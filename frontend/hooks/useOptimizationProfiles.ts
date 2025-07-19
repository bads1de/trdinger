import { OptimizationProfile } from "@/types/bayesian-optimization";
import { useDataFetching } from "./useDataFetching";

interface OptimizationProfilesParams {
  target_model_type?: string;
}

export const useOptimizationProfiles = (modelType?: string) => {
  const {
    data: profiles,
    loading: isLoading,
    error,
    refetch: fetchProfiles,
  } = useDataFetching<OptimizationProfile, OptimizationProfilesParams>({
    endpoint: "/api/bayesian-optimization/profiles",
    initialParams: { target_model_type: modelType },
    dataPath: "profiles",
    dependencies: [modelType],
    errorMessage: "プロファイルの取得中にエラーが発生しました",
  });

  return {
    profiles,
    isLoading,
    error,
    fetchProfiles,
  };
};
