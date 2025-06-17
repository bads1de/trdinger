# 差分更新機能の実装ドキュメント

## 概要

フロントエンドの差分更新ボタンとバックエンドAPIの接続を実装し、ユーザーフィードバックとエラーハンドリングを改善しました。

## 実装内容

### 1. フロントエンド改善

#### 1.1 メッセージ状態の追加
- `incrementalUpdateMessage` 状態を追加
- 成功・失敗メッセージの統一的な表示

#### 1.2 差分更新機能の改善
**ファイル**: `frontend/app/data/page.tsx`

```typescript
const handleIncrementalUpdate = async () => {
  try {
    setUpdating(true);
    setError("");
    setIncrementalUpdateMessage("");

    const response = await fetch(
      `${BACKEND_API_URL}/api/data-collection/update?symbol=${selectedSymbol}&timeframe=${selectedTimeFrame}`,
      { method: "POST" }
    );

    const result = await response.json();

    if (result.success) {
      // 成功メッセージを表示
      const savedCount = result.saved_count || 0;
      setIncrementalUpdateMessage(
        `✅ 差分更新完了！ ${selectedSymbol} ${selectedTimeFrame} - ${savedCount}件のデータを更新しました`
      );

      // 更新後に全てのデータを再取得
      await Promise.all([
        fetchOHLCVData(),
        fetchFundingRateData(),
        fetchOpenInterestData(),
      ]);

      // データ状況も更新
      fetchDataStatus();

      // 10秒後にメッセージをクリア
      setTimeout(() => setIncrementalUpdateMessage(""), 10000);
    } else {
      // エラーハンドリング
      const errorMessage = result.message || "差分更新に失敗しました";
      setError(errorMessage);
      setIncrementalUpdateMessage(`❌ ${errorMessage}`);
      setTimeout(() => setIncrementalUpdateMessage(""), 10000);
    }
  } catch (err) {
    // ネットワークエラーハンドリング
    const errorMessage = "差分更新中にエラーが発生しました";
    setError(errorMessage);
    setIncrementalUpdateMessage(`❌ ${errorMessage}`);
    console.error("差分更新エラー:", err);
    setTimeout(() => setIncrementalUpdateMessage(""), 10000);
  } finally {
    setUpdating(false);
  }
};
```

#### 1.3 DataControlsコンポーネントの更新
**ファイル**: `frontend/app/data/components/DataControls.tsx`

- `incrementalUpdateMessage` プロパティを追加
- メッセージ表示セクションに差分更新メッセージを追加

### 2. バックエンド確認

#### 2.1 既存の実装
**ファイル**: `backend/app/api/data_collection.py`

```python
@router.post("/update")
async def update_incremental_data(
    symbol: str = "BTC/USDT", timeframe: str = "1h", db: Session = Depends(get_db)
) -> Dict:
    """
    差分データを更新
    """
    try:
        service = HistoricalDataService()
        repository = OHLCVRepository(db)

        result = await service.collect_incremental_data(symbol, timeframe, repository)

        return result

    except Exception as e:
        logger.error(f"差分データ更新エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

#### 2.2 HistoricalDataServiceの実装
**ファイル**: `backend/app/core/services/historical_data_service.py`

```python
async def collect_incremental_data(
    self,
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    repository: Optional[OHLCVRepository] = None,
) -> Dict:
    """
    差分データを収集（最新タイムスタンプ以降）
    """
    if not repository:
        return {"success": False, "message": "リポジトリが必要です"}

    try:
        latest_timestamp = repository.get_latest_timestamp(symbol, timeframe)
        since_ms = (
            int(latest_timestamp.timestamp() * 1000) if latest_timestamp else None
        )

        if since_ms:
            logger.info(f"差分データ収集開始: {symbol} {timeframe} (since: {since_ms})")
        else:
            logger.info(f"初回データ収集開始: {symbol} {timeframe}")

        ohlcv_data = await self.market_service.fetch_ohlcv_data(
            symbol, timeframe, 1000, since=since_ms
        )

        if not ohlcv_data:
            return {
                "success": True,
                "message": "新しいデータはありません",
                "saved_count": 0,
            }

        saved_count = await self.market_service._save_ohlcv_to_database(
            ohlcv_data, symbol, timeframe, repository
        )
        logger.info(f"差分データ収集完了: {saved_count}件保存")
        return {
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "saved_count": saved_count,
        }
    except Exception as e:
        logger.error(f"差分データ収集エラー: {e}")
        return {"success": False, "message": str(e)}
```

### 3. テストの実装

#### 3.1 バックエンドテスト
**ファイル**: `backend/tests/test_incremental_update.py`

- APIエンドポイントのテスト
- HistoricalDataServiceのテスト
- 成功・失敗・エラーケースのテスト

#### 3.2 フロントエンドテスト
**ファイル**: `frontend/__tests__/incremental-update.test.tsx`

- 差分更新ボタンの表示・動作テスト
- API連携のテスト
- メッセージ表示のテスト
- エラーハンドリングのテスト

## 改善点

### 1. ユーザーフィードバックの向上
- ✅ 成功時のメッセージ表示
- ✅ 更新されたデータ件数の表示
- ✅ エラーメッセージの統一
- ✅ メッセージの自動クリア（10秒後）

### 2. データ整合性の向上
- ✅ 差分更新後の全データ再取得（OHLCV, FR, OI）
- ✅ データ状況の更新
- ✅ 並列データ取得による高速化

### 3. エラーハンドリングの改善
- ✅ ネットワークエラーの適切な処理
- ✅ APIエラーレスポンスの処理
- ✅ ユーザーへの分かりやすいエラーメッセージ

## 使用方法

### 1. 差分更新の実行
1. データページ（http://localhost:3000/data）にアクセス
2. 通貨ペアと時間軸を選択
3. 「差分更新」ボタンをクリック
4. 成功メッセージまたはエラーメッセージを確認

### 2. 期待される動作
- **成功時**: 「✅ 差分更新完了！ BTC/USDT:USDT 1h - X件のデータを更新しました」
- **新データなし**: 「✅ 差分更新完了！ BTC/USDT:USDT 1h - 0件のデータを更新しました」
- **エラー時**: 「❌ [エラーメッセージ]」

## API仕様

### エンドポイント
```
POST /api/data-collection/update
```

### パラメータ
- `symbol` (optional): 取引ペア（デフォルト: "BTC/USDT"）
- `timeframe` (optional): 時間軸（デフォルト: "1h"）

### レスポンス
```json
{
  "success": true,
  "symbol": "BTC/USDT:USDT",
  "timeframe": "1h",
  "saved_count": 5
}
```

## 今後の改善案

1. **リアルタイム更新**: WebSocketを使用したリアルタイムデータ更新
2. **バッチ処理**: 複数通貨ペア・時間軸の一括差分更新
3. **進捗表示**: 長時間の処理における進捗バーの表示
4. **自動更新**: 定期的な自動差分更新機能
5. **通知機能**: 更新完了時のブラウザ通知

## 注意事項

- 差分更新は既存データの最新タイムスタンプ以降のデータのみを取得します
- 初回実行時は全データを取得する場合があります
- ネットワーク状況により処理時間が変動する可能性があります
- エラーが発生した場合は、ログを確認して原因を特定してください
