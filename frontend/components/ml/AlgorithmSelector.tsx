"use client";

import React, { useState, useMemo } from "react";
import { useAlgorithms, Algorithm } from "../../hooks/useAlgorithms";

/**
 * アルゴリズム選択コンポーネント
 *
 * ユーザーがアルゴリズムを選択できるコンポーネントです。
 * アルゴリズムのフィルタリング、検索、推奨アルゴリズムの表示などをサポートします。
 *
 */

import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import {
  Accordion,
  AccordionItem,
  AccordionTrigger,
  AccordionContent,
} from "@/components/ui/accordion";
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
} from "@/components/ui/select";

import {
  CheckCircle,
  AlertTriangle,
  Info,
  Gauge,
  TrendingUp,
} from "lucide-react";

interface AlgorithmSelectorProps {
  selectedAlgorithms?: string[];
  onSelectionChange?: (algorithms: string[]) => void;
  maxSelection?: number;
  showRecommendations?: boolean;
  requirements?: {
    dataSize?: "small" | "medium" | "large";
    needsProbability?: boolean;
    needsFeatureImportance?: boolean;
    needsSpeed?: boolean;
    needsAccuracy?: boolean;
    hasNoise?: boolean;
  };
}

const AlgorithmSelector: React.FC<AlgorithmSelectorProps> = ({
  selectedAlgorithms = [],
  onSelectionChange,
  maxSelection = 5,
  showRecommendations = true,
  requirements = {},
}) => {
  const {
    algorithms,
    algorithmsByType,
    probabilityAlgorithms,
    featureImportanceAlgorithms,
    statistics,
    searchAlgorithms,
    getRecommendedAlgorithms,
    isLoading,
    error,
    getTypeLabel,
  } = useAlgorithms();

  const [searchQuery, setSearchQuery] = useState("");
  const [filterType, setFilterType] = useState<string>("all");
  const [filterCapability, setFilterCapability] = useState<string>("all");

  // フィルタリングされたアルゴリズム
  const filteredAlgorithms = useMemo(() => {
    let filtered = searchAlgorithms(searchQuery);

    if (filterType !== "all") {
      filtered = filtered.filter((algo) => algo.type === filterType);
    }

    if (filterCapability !== "all") {
      filtered = filtered.filter((algo) =>
        algo.capabilities.includes(filterCapability as any)
      );
    }

    return filtered;
  }, [searchAlgorithms, searchQuery, filterType, filterCapability]);

  // 推奨アルゴリズム
  const recommendedAlgorithms = useMemo(() => {
    if (!showRecommendations) return [];
    return getRecommendedAlgorithms(requirements);
  }, [getRecommendedAlgorithms, requirements, showRecommendations]);

  // アルゴリズム選択処理
  const handleAlgorithmToggle = (algorithmName: string) => {
    if (!onSelectionChange) return;

    const isSelected = selectedAlgorithms.includes(algorithmName);
    let newSelection: string[];

    if (isSelected) {
      newSelection = selectedAlgorithms.filter(
        (name) => name !== algorithmName
      );
    } else {
      if (selectedAlgorithms.length >= maxSelection) {
        return; // 最大選択数に達している
      }
      newSelection = [...selectedAlgorithms, algorithmName];
    }

    onSelectionChange(newSelection);
  };

  // アルゴリズムカードコンポーネント
  const AlgorithmCard: React.FC<{
    algorithm: Algorithm;
    isRecommended?: boolean;
  }> = ({ algorithm, isRecommended = false }) => {
    const isSelected = selectedAlgorithms.includes(algorithm.name);
    const canSelect = !isSelected && selectedAlgorithms.length < maxSelection;

    return (
      <Card
        className={`relative ${onSelectionChange ? "cursor-pointer" : ""} ${
          isSelected
            ? "border-2 border-primary bg-primary/10"
            : "border border-border"
        } ${onSelectionChange ? "hover:border-primary hover:shadow-md" : ""}`}
        onClick={() =>
          onSelectionChange &&
          canSelect &&
          handleAlgorithmToggle(algorithm.name)
        }
      >
        {isRecommended && (
          <Badge variant="secondary" className="absolute top-2 right-2 z-10">
            推奨
          </Badge>
        )}

        <CardContent>
          <div className="flex items-center mb-2">
            {isSelected && (
              <CheckCircle className="mr-2 text-primary h-5 w-5" />
            )}
            <h3 className="text-lg font-semibold">{algorithm.display_name}</h3>
          </div>

          <p className="text-sm text-muted-foreground mb-4">
            {algorithm.description}
          </p>

          <div className="mb-4 flex flex-wrap gap-1">
            <Badge variant="outline" className="mr-1 mb-1">
              {getTypeLabel(algorithm.type)}
            </Badge>
            {algorithm.has_probability_prediction && (
              <Badge variant="secondary" className="mr-1 mb-1">
                確率予測
              </Badge>
            )}
            {algorithm.has_feature_importance && (
              <Badge variant="success" className="mr-1 mb-1">
                特徴量重要度
              </Badge>
            )}
          </div>

          <Accordion type="single" collapsible>
            <AccordionItem value="details">
              <AccordionTrigger className="text-sm">詳細情報</AccordionTrigger>
              <AccordionContent>
                <div className="space-y-4">
                  <div>
                    <h4 className="text-sm font-semibold text-green-600 mb-1">
                      長所:
                    </h4>
                    <ul className="space-y-1">
                      {algorithm.pros.map((pro, index) => (
                        <li key={index} className="flex items-start py-0.5">
                          <CheckCircle className="h-4 w-4 text-green-600 mr-2 mt-0.5 flex-shrink-0" />
                          <span className="text-sm">{pro}</span>
                        </li>
                      ))}
                    </ul>
                  </div>

                  <div>
                    <h4 className="text-sm font-semibold text-yellow-600 mb-1">
                      短所:
                    </h4>
                    <ul className="space-y-1">
                      {algorithm.cons.map((con, index) => (
                        <li key={index} className="flex items-start py-0.5">
                          <AlertTriangle className="h-4 w-4 text-yellow-600 mr-2 mt-0.5 flex-shrink-0" />
                          <span className="text-sm">{con}</span>
                        </li>
                      ))}
                    </ul>
                  </div>

                  <div>
                    <h4 className="text-sm font-semibold text-blue-600 mb-1">
                      適用場面:
                    </h4>
                    <ul className="space-y-1">
                      {algorithm.best_for.map((use, index) => (
                        <li key={index} className="flex items-start py-0.5">
                          <Info className="h-4 w-4 text-blue-600 mr-2 mt-0.5 flex-shrink-0" />
                          <span className="text-sm">{use}</span>
                        </li>
                      ))}
                    </ul>
                  </div>

                  {algorithm.note && (
                    <Alert className="mt-2 bg-yellow-50 border-yellow-200">
                      <AlertDescription>{algorithm.note}</AlertDescription>
                    </Alert>
                  )}
                </div>
              </AccordionContent>
            </AccordionItem>
          </Accordion>
        </CardContent>
      </Card>
    );
  };

  if (isLoading) {
    return (
      <div className="flex justify-center items-center p-6">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
      </div>
    );
  }

  if (error) {
    return (
      <Alert variant="destructive" className="m-2">
        <AlertDescription>{error}</AlertDescription>
      </Alert>
    );
  }

  return (
    <div>
      {/* 統計情報 */}
      {statistics && (
        <div className="mb-6">
          <h3 className="text-lg font-semibold mb-2">
            利用可能なアルゴリズム ({statistics.total}個)
          </h3>
          <div className="flex flex-wrap gap-2">
            <Badge variant="outline" className="flex items-center gap-1">
              <TrendingUp className="h-3.5 w-3.5" />
              確率予測対応: {statistics.probabilityCount}個
            </Badge>
            <Badge variant="outline" className="flex items-center gap-1">
              <Gauge className="h-3.5 w-3.5" />
              特徴量重要度: {statistics.featureImportanceCount}個
            </Badge>
          </div>
        </div>
      )}

      {/* 選択状況 */}
      {onSelectionChange && (
        <div className="mb-6">
          <h4 className="text-base font-medium mb-2">
            選択済み: {selectedAlgorithms.length} / {maxSelection}
          </h4>
          <div className="flex flex-wrap gap-2">
            {selectedAlgorithms.map((name) => (
              <Badge
                key={name}
                variant="secondary"
                className="flex items-center gap-1 cursor-pointer"
                onClick={() => handleAlgorithmToggle(name)}
              >
                {algorithms.find((a) => a.name === name)?.display_name || name}
                <button className="ml-1 h-3.5 w-3.5 rounded-full">
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    width="14"
                    height="14"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  >
                    <path d="M18 6 6 18" />
                    <path d="m6 6 12 12" />
                  </svg>
                </button>
              </Badge>
            ))}
          </div>
        </div>
      )}

      {/* フィルター */}
      <div className="mb-6">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <Input
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="アルゴリズム名、説明で検索..."
            />
          </div>
          <div>
            <Select
              value={filterType}
              onValueChange={(value) => setFilterType(value)}
            >
              <SelectTrigger>
                <SelectValue placeholder="タイプ" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">すべて</SelectItem>
                {statistics?.byType.map(({ type, count }) => (
                  <SelectItem key={type} value={type}>
                    {type} ({count})
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div>
            <Select
              value={filterCapability}
              onValueChange={(value) => setFilterCapability(value)}
            >
              <SelectTrigger>
                <SelectValue placeholder="機能" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">すべて</SelectItem>
                {statistics?.byCapability.map(({ capability, count }) => (
                  <SelectItem key={capability} value={capability}>
                    {capability} ({count})
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>
      </div>

      {/* 推奨アルゴリズム */}
      {showRecommendations && recommendedAlgorithms.length > 0 && (
        <div className="mb-8">
          <h3 className="text-lg font-semibold mb-4">推奨アルゴリズム</h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
            {recommendedAlgorithms.slice(0, 3).map((algorithm) => (
              <div key={algorithm.name}>
                <AlgorithmCard algorithm={algorithm} isRecommended />
              </div>
            ))}
          </div>
        </div>
      )}

      {/* アルゴリズム一覧 */}
      <h3 className="text-lg font-semibold mb-4">全アルゴリズム</h3>
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
        {filteredAlgorithms.map((algorithm) => (
          <div key={algorithm.name}>
            <AlgorithmCard algorithm={algorithm} />
          </div>
        ))}
      </div>

      {filteredAlgorithms.length === 0 && (
        <Alert className="mt-4">
          <AlertDescription>
            条件に一致するアルゴリズムが見つかりませんでした。
          </AlertDescription>
        </Alert>
      )}
    </div>
  );
};

export default AlgorithmSelector;
