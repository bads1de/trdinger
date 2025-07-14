import { useState, useEffect, useCallback } from "react";
import { useApiCall } from "./useApiCall";
import {
  ParameterSpace,
  DefaultParameterSpaceResponse,
} from "@/types/bayesian-optimization";

export const useBayesianOptimizationForm = (modelType: string) => {
  const [customParameterSpace, setCustomParameterSpace] = useState("");
  const [defaultParameterSpace, setDefaultParameterSpace] = useState<Record<
    string,
    ParameterSpace
  > | null>(null);

  const { execute: fetchParameterSpace, loading: isLoadingDefaultParams } =
    useApiCall<DefaultParameterSpaceResponse>();

  const fetchDefaultParameterSpace = useCallback(async () => {
    await fetchParameterSpace(
      `/api/bayesian-optimization/parameter-spaces/${modelType}`,
      {
        method: "GET",
        onSuccess: (data) => {
          if (data.success && data.parameter_space) {
            setDefaultParameterSpace(data.parameter_space);
            setCustomParameterSpace(
              JSON.stringify(data.parameter_space, null, 2)
            );
          }
        },
        onError: (errorMessage) => {
          console.error("デフォルトパラメータ空間の取得に失敗:", errorMessage);
        },
      }
    );
  }, [modelType, fetchParameterSpace]);

  useEffect(() => {
    fetchDefaultParameterSpace();
  }, [fetchDefaultParameterSpace]);

  return {
    customParameterSpace,
    setCustomParameterSpace,
    defaultParameterSpace,
    isLoadingDefaultParams,
  };
};
