"use client";

import React, { useState } from "react";
import {
  CheckCircle,
  Info,
  TrendingUp,
  Gauge,
  Target,
  Database,
  Brain,
} from "lucide-react";
import { Card } from "@/components/ui/card";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
} from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Badge } from "@/components/ui/badge";
import { useAlgorithms, Algorithm } from "../../hooks/useAlgorithms";

const AlgorithmCatalog: React.FC = () => {
  const {
    algorithms,
    algorithmsByType,
    statistics,
    searchAlgorithms,
    isLoading,
    error,
    getTypeLabel,
  } = useAlgorithms();

  const [searchQuery, setSearchQuery] = useState("");
  const [selectedType, setSelectedType] = useState<string>("all");
  const [tabValue, setTabValue] = useState(0);

  // フィルタリングされたアルゴリズム
  const filteredAlgorithms = React.useMemo(() => {
    let filtered = searchAlgorithms(searchQuery);

    if (selectedType !== "all") {
      filtered = filtered.filter((algo) => algo.type === selectedType);
    }

    return filtered;
  }, [searchAlgorithms, searchQuery, selectedType]);

  // タイプアイコンの取得
  const getTypeIcon = (type: string) => {
    switch (type) {
      case "tree_based":
        return <TrendingUp className="h-5 w-5" />;
      case "linear":
        return <Gauge className="h-5 w-5" />;
      case "boosting":
        return <Target className="h-5 w-5" />;
      case "probabilistic":
        return <Brain className="h-5 w-5" />;
      case "instance_based":
        return <Database className="h-5 w-5" />;
      default:
        return <Info className="h-5 w-5" />;
    }
  };

  // アルゴリズムカードコンポーネント
  const AlgorithmCard: React.FC<{ algorithm: Algorithm }> = ({ algorithm }) => (
    <Card className="h-full flex flex-col">
      <div className="p-6 flex-grow">
        <div className="flex items-center mb-4">
          {getTypeIcon(algorithm.type)}
          <h3 className="text-lg font-semibold ml-2">
            {algorithm.display_name}
          </h3>
        </div>

        <p className="text-sm text-muted-foreground mb-4">
          {algorithm.description}
        </p>

        <div className="mb-4 flex flex-wrap gap-2">
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
          {algorithm.note && (
            <Badge variant="warning" className="mr-1 mb-1">
              注意事項あり
            </Badge>
          )}
        </div>

        <h4 className="text-sm font-medium text-green-600 mb-2">長所:</h4>
        <ul className="space-y-1 mb-4">
          {algorithm.pros.slice(0, 3).map((pro, index) => (
            <li key={index} className="flex items-start py-0">
              <CheckCircle className="h-4 w-4 text-green-600 mr-2 mt-0.5 flex-shrink-0" />
              <span className="text-sm">{pro}</span>
            </li>
          ))}
        </ul>

        <h4 className="text-sm font-medium text-blue-600 mb-2">適用場面:</h4>
        <ul className="space-y-1 mb-4">
          {algorithm.best_for.slice(0, 2).map((use, index) => (
            <li key={index} className="flex items-start py-0">
              <Info className="h-4 w-4 text-blue-600 mr-2 mt-0.5 flex-shrink-0" />
              <span className="text-sm">{use}</span>
            </li>
          ))}
        </ul>

        {algorithm.note && (
          <Alert variant="destructive" className="mt-4">
            <AlertDescription>{algorithm.note}</AlertDescription>
          </Alert>
        )}
      </div>
    </Card>
  );

  if (isLoading) {
    return (
      <div className="flex justify-center p-8">
        <div className="animate-spin h-8 w-8 border-4 border-primary border-t-transparent rounded-full"></div>
      </div>
    );
  }

  if (error) {
    return (
      <Alert variant="destructive" className="m-4">
        <AlertDescription>{error}</AlertDescription>
      </Alert>
    );
  }

  return (
    <div className="container mx-auto">
      {/* ヘッダー */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2">🤖 MLアルゴリズムカタログ</h1>
        <p className="text-muted-foreground">
          利用可能な機械学習アルゴリズムの詳細情報と特徴を確認できます。
        </p>
      </div>

      {/* 統計情報 */}
      {statistics && (
        <div className="bg-card rounded-lg border shadow-sm p-6 mb-8">
          <h2 className="text-xl font-semibold mb-4">📊 統計情報</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center">
              <p className="text-4xl font-bold text-primary mb-1">
                {statistics.total}
              </p>
              <p className="text-sm text-muted-foreground">総アルゴリズム数</p>
            </div>
            <div className="text-center">
              <p className="text-4xl font-bold text-blue-500 mb-1">
                {statistics.probabilityCount}
              </p>
              <p className="text-sm text-muted-foreground">確率予測対応</p>
            </div>
            <div className="text-center">
              <p className="text-4xl font-bold text-green-600 mb-1">
                {statistics.featureImportanceCount}
              </p>
              <p className="text-sm text-muted-foreground">特徴量重要度対応</p>
            </div>
            <div className="text-center">
              <p className="text-4xl font-bold text-purple-500 mb-1">
                {statistics.byType.length}
              </p>
              <p className="text-sm text-muted-foreground">
                アルゴリズムタイプ
              </p>
            </div>
          </div>
        </div>
      )}

      {/* タブ */}
      <Tabs
        value={String(tabValue)}
        onValueChange={(value) => setTabValue(Number(value))}
        className="mb-6"
      >
        <TabsList className="w-full">
          <TabsTrigger value="0" className="flex-1">
            全アルゴリズム
          </TabsTrigger>
          <TabsTrigger value="1" className="flex-1">
            タイプ別
          </TabsTrigger>
          <TabsTrigger value="2" className="flex-1">
            機能別
          </TabsTrigger>
        </TabsList>
      </Tabs>

      {/* 全アルゴリズムタブ */}
      <TabsContent value="0">
        {/* フィルター */}
        <div className="mb-6">
          <div className="grid grid-cols-1 md:grid-cols-12 gap-4">
            <div className="md:col-span-8">
              <Input
                type="text"
                placeholder="アルゴリズム名、説明、特徴で検索..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full"
              />
            </div>
            <div className="md:col-span-4">
              <Select
                value={selectedType}
                onValueChange={(value) => setSelectedType(value)}
              >
                <SelectTrigger>
                  <SelectValue placeholder="タイプを選択" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">すべて</SelectItem>
                  {statistics?.byType.map(({ type }) => (
                    <SelectItem
                      key={type}
                      value={type.toLowerCase().replace(" ", "_")}
                    >
                      {type}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
        </div>

        {/* アルゴリズム一覧 */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
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
      </TabsContent>

      {/* タイプ別タブ */}
      <TabsContent value="1">
        {Object.entries(algorithmsByType).map(([type, algos]) => (
          <div key={type} className="mb-8">
            <div className="flex items-center mb-2">
              {getTypeIcon(algos[0]?.type)}
              <h3 className="text-xl font-semibold ml-2">
                {type} ({algos.length}個)
              </h3>
            </div>
            <Separator className="mb-4" />
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {algos.map((algorithm) => (
                <div key={algorithm.name}>
                  <AlgorithmCard algorithm={algorithm} />
                </div>
              ))}
            </div>
          </div>
        ))}
      </TabsContent>

      {/* 機能別タブ */}
      <TabsContent value="2">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <Card>
            <div className="p-6">
              <h3 className="text-lg font-semibold text-blue-600 mb-2">
                🎯 確率予測対応アルゴリズム
              </h3>
              <p className="text-sm text-muted-foreground mb-4">
                予測確率を出力できるアルゴリズム
              </p>
              <ul className="space-y-3">
                {algorithms
                  .filter((a) => a.has_probability_prediction)
                  .map((algo) => (
                    <li key={algo.name} className="border-b pb-2">
                      <p className="font-medium">{algo.display_name}</p>
                      <p className="text-sm text-muted-foreground">
                        {algo.description}
                      </p>
                    </li>
                  ))}
              </ul>
            </div>
          </Card>
          <Card>
            <div className="p-6">
              <h3 className="text-lg font-semibold text-green-600 mb-2">
                📊 特徴量重要度対応アルゴリズム
              </h3>
              <p className="text-sm text-muted-foreground mb-4">
                特徴量の重要度を算出できるアルゴリズム
              </p>
              <ul className="space-y-3">
                {algorithms
                  .filter((a) => a.has_feature_importance)
                  .map((algo) => (
                    <li key={algo.name} className="border-b pb-2">
                      <p className="font-medium">{algo.display_name}</p>
                      <p className="text-sm text-muted-foreground">
                        {algo.description}
                      </p>
                    </li>
                  ))}
              </ul>
            </div>
          </Card>
        </div>
      </TabsContent>
    </div>
  );
};

export default AlgorithmCatalog;
