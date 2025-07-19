import { useState } from "react";
import { OptimizationProfile } from "@/types/bayesian-optimization";
import { useDataFetching } from "./useDataFetching";

interface OptimizationProfilesParams {
  target_model_type?: string;
}

interface DeleteProfileResponse {
  success: boolean;
  message: string;
  timestamp: string;
}

export const useOptimizationProfiles = (modelType?: string) => {
  const [deleteLoading, setDeleteLoading] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);

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

  const deleteProfile = async (profileId: number): Promise<boolean> => {
    setDeleteLoading(true);
    setDeleteError(null);

    try {
      const response = await fetch(
        `/api/bayesian-optimization/profiles/${profileId}`,
        {
          method: "DELETE",
          headers: {
            "Content-Type": "application/json",
          },
        }
      );

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(
          errorData.error || `削除に失敗しました (${response.status})`
        );
      }

      const result: DeleteProfileResponse = await response.json();

      if (!result.success) {
        throw new Error(result.message || "削除に失敗しました");
      }

      // 削除成功後、プロファイル一覧を再取得
      await fetchProfiles();
      return true;
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : "削除中にエラーが発生しました";
      setDeleteError(errorMessage);
      console.error("プロファイル削除エラー:", error);
      return false;
    } finally {
      setDeleteLoading(false);
    }
  };

  return {
    profiles,
    isLoading,
    error,
    fetchProfiles,
    deleteProfile,
    deleteLoading,
    deleteError,
  };
};
