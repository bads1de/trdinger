"use client";

import React, { useState, useEffect } from "react";

const TestStrategiesPage: React.FC = () => {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchData = async () => {
    try {
      setLoading(true);
      setError(null);
      
      console.log("Fetching data from /api/strategies/unified...");
      
      const response = await fetch("/api/strategies/unified");
      
      console.log("Response status:", response.status);
      console.log("Response ok:", response.ok);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const result = await response.json();
      console.log("Response data:", result);
      
      setData(result);
    } catch (err) {
      console.error("Fetch error:", err);
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  return (
    <div className="min-h-screen bg-gray-100 p-8">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-3xl font-bold mb-6">戦略データテスト</h1>
        
        <button 
          onClick={fetchData}
          className="mb-4 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
          disabled={loading}
        >
          {loading ? "読み込み中..." : "データを再取得"}
        </button>

        {error && (
          <div className="mb-4 p-4 bg-red-100 border border-red-400 text-red-700 rounded">
            <h2 className="font-bold">エラー:</h2>
            <p>{error}</p>
          </div>
        )}

        {loading && (
          <div className="mb-4 p-4 bg-blue-100 border border-blue-400 text-blue-700 rounded">
            データを読み込み中...
          </div>
        )}

        {data && (
          <div className="bg-white p-6 rounded-lg shadow">
            <h2 className="text-xl font-semibold mb-4">取得データ:</h2>
            <div className="mb-4">
              <p><strong>成功:</strong> {data.success ? "はい" : "いいえ"}</p>
              <p><strong>戦略数:</strong> {data.strategies?.length || 0}</p>
              <p><strong>総数:</strong> {data.total_count || 0}</p>
            </div>
            
            {data.strategies && data.strategies.length > 0 && (
              <div>
                <h3 className="text-lg font-semibold mb-2">戦略一覧:</h3>
                <div className="space-y-4">
                  {data.strategies.map((strategy: any, index: number) => (
                    <div key={strategy.id || index} className="border p-4 rounded">
                      <h4 className="font-semibold">{strategy.name}</h4>
                      <p className="text-sm text-gray-600">{strategy.description}</p>
                      <div className="mt-2 text-sm">
                        <span className="inline-block bg-blue-100 text-blue-800 px-2 py-1 rounded mr-2">
                          {strategy.category}
                        </span>
                        <span className="inline-block bg-green-100 text-green-800 px-2 py-1 rounded mr-2">
                          リターン: {(strategy.expected_return * 100).toFixed(1)}%
                        </span>
                        <span className="inline-block bg-yellow-100 text-yellow-800 px-2 py-1 rounded">
                          シャープ: {strategy.sharpe_ratio.toFixed(2)}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
            
            <details className="mt-6">
              <summary className="cursor-pointer font-semibold">生データを表示</summary>
              <pre className="mt-2 p-4 bg-gray-100 rounded text-xs overflow-auto">
                {JSON.stringify(data, null, 2)}
              </pre>
            </details>
          </div>
        )}
      </div>
    </div>
  );
};

export default TestStrategiesPage;
