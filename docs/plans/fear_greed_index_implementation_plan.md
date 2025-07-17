# Fear & Greed Index データ収集・DB 設計実装計画

## 概要

Alternative.me Fear & Greed Index API を使用してセンチメント分析データを収集し、既存のトレーディングシステムに統合するための実装計画です。

## 1. データソース分析

### Alternative.me Fear & Greed Index API

- **エンドポイント**: `https://api.alternative.me/fng/`
- **更新頻度**: 1 日 1 回（UTC 00:00 頃）
- **レート制限**: 明記されていないが、適切な間隔での取得を推奨
- **データ形式**: JSON

### API レスポンス例

```json
{
  "name": "Fear and Greed Index",
  "data": [
    {
      "value": "74",
      "value_classification": "Greed",
      "timestamp": "1752710400",
      "time_until_update": "73129"
    }
  ],
  "metadata": {
    "error": null
  }
}
```

### パラメータ

- `limit`: 取得するデータ数（デフォルト: 1、最大: 不明）
- `date_format`: タイムスタンプ形式（デフォルト: Unix timestamp）

## 2. データベース設計

### 新規テーブル: `fear_greed_index_data`

```sql
CREATE TABLE fear_greed_index_data (
    id SERIAL PRIMARY KEY,

    -- Fear & Greed Index データ
    value INTEGER NOT NULL,                    -- インデックス値 (0-100)
    value_classification VARCHAR(20) NOT NULL, -- 分類 (Extreme Fear, Fear, Neutral, Greed, Extreme Greed)

    -- タイムスタンプ
    data_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,  -- データの日付（UTC）
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,       -- データ取得時刻（UTC）

    -- メタデータ
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- インデックス
    CONSTRAINT uq_fear_greed_data_timestamp UNIQUE (data_timestamp)
);

-- インデックス定義
CREATE INDEX idx_fear_greed_data_timestamp ON fear_greed_index_data (data_timestamp);
CREATE INDEX idx_fear_greed_timestamp ON fear_greed_index_data (timestamp);
CREATE INDEX idx_fear_greed_value ON fear_greed_index_data (value);
```

### 既存パターンとの整合性

- `OHLCVData`, `FundingRateData`, `OpenInterestData`と同様の構造
- TimescaleDB ハイパーテーブル対応
- 重複データ防止のユニーク制約
- 作成・更新時刻の自動管理

## 3. 実装アーキテクチャ

### 3.1 データモデル (`backend/database/models.py`)

```python
class FearGreedIndexData(Base):
    """
    Fear & Greed Index データテーブル

    Alternative.me APIから取得したセンチメント指標を保存します。

    """

    __tablename__ = "fear_greed_index_data"

    # 主キー
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Fear & Greed Index データ
    value = Column(Integer, nullable=False)
    value_classification = Column(String(20), nullable=False)

    # タイムスタンプ
    data_timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)

    # メタデータ
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # インデックス定義
    __table_args__ = (
        Index("idx_fear_greed_data_timestamp", "data_timestamp"),
        Index("idx_fear_greed_timestamp", "timestamp"),
        Index("idx_fear_greed_value", "value"),
        Index("uq_fear_greed_data_timestamp", "data_timestamp", unique=True),
    )
```

### 3.2 リポジトリ (`backend/database/repositories/fear_greed_repository.py`)

```python
class FearGreedIndexRepository(BaseRepository):
    """Fear & Greed Index データのリポジトリクラス"""

    def __init__(self, db: Session):
        super().__init__(db, FearGreedIndexData)

    def insert_fear_greed_data(self, records: List[dict]) -> int:
        """Fear & Greed Index データを一括挿入（重複は無視）"""

    def get_fear_greed_data(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[FearGreedIndexData]:
        """Fear & Greed Index データを取得"""

    def get_latest_fear_greed_data(self, limit: int = 30) -> List[FearGreedIndexData]:
        """最新のFear & Greed Index データを取得"""
```

### 3.3 データ収集サービス (`backend/app/core/services/data_collection/fear_greed_service.py`)

```python
class FearGreedIndexService:
    """Fear & Greed Index データ収集サービス"""

    def __init__(self):
        self.api_url = "https://api.alternative.me/fng/"
        self.session = aiohttp.ClientSession()

    async def fetch_fear_greed_data(self, limit: int = 30) -> List[dict]:
        """Alternative.me APIからFear & Greed Indexデータを取得"""

    async def fetch_and_save_fear_greed_data(
        self,
        limit: int = 30,
        repository: Optional[FearGreedIndexRepository] = None,
    ) -> dict:
        """Fear & Greed Indexデータを取得してデータベースに保存"""
```

## 4. 実装手順

### Phase 1: データベース・モデル実装

1. **データモデル追加** (`backend/database/models.py`)

   - `FearGreedIndexData`クラスの実装
   - マイグレーションファイル作成

2. **リポジトリ実装** (`backend/database/repositories/fear_greed_repository.py`)
   - 基本的な CRUD 操作
   - 既存パターンに準拠した実装

### Phase 2: データ収集サービス実装

3. **Fear & Greed Index サービス** (`backend/app/core/services/data_collection/fear_greed_service.py`)

   - Alternative.me API クライアント
   - データ変換・検証ロジック
   - エラーハンドリング

4. **外部市場データ収集器** (`backend/data_collector/external_market_collector.py`)
   - 新規ファイル作成
   - Fear & Greed Index 収集機能
   - 将来的な他の外部データソース対応

### Phase 3: API・統合実装

5. **API エンドポイント** (`backend/app/api/fear_greed.py`)

   - データ取得 API
   - 履歴データ収集 API
   - 差分更新 API

6. **フロントエンド API** (`frontend/app/api/data/fear-greed/`)
   - バックエンド API 転送
   - エラーハンドリング

### Phase 4: 全期間データ収集

7. **履歴データ収集**
   - 過去データの一括取得（可能な限り）
   - データ整合性チェック
   - 欠損データの補完

## 5. データ収集戦略

### 5.1 初回データ収集

- Alternative.me API から取得可能な最大限の履歴データを収集
- API の`limit`パラメータを使用して大量データを取得
- データの連続性を確認

### 5.2 定期更新

- 1 日 1 回の定期実行（UTC 01:00 頃を推奨）
- 差分データのみ取得
- 既存データとの重複チェック

### 5.3 エラーハンドリング

- API レート制限対応
- ネットワークエラー時のリトライ機能
- データ検証とログ記録

## 6. 特徴量統合準備

### 6.1 データ形式標準化

- タイムスタンプの UTC 統一
- 日次データとしての扱い
- OHLCV データとの時間軸合わせ

### 6.2 特徴量エンジニアリング準備

- `FearAndGreedIndex`: 生の値 (0-100)
- `FearAndGreedIndex_MA7`: 7 日移動平均
- `FearAndGreedIndex_Change`: 前日比変化
- `FearAndGreedIndex_Classification`: カテゴリ値のエンコーディング

## 7. テスト計画

### 7.1 単体テスト

- API クライアントのテスト
- データ変換ロジックのテスト
- リポジトリ操作のテスト

### 7.2 統合テスト

- エンドツーエンドのデータ収集テスト
- データベース整合性テスト
- API エンドポイントのテスト

## 8. 運用・監視

### 8.1 ログ監視

- データ収集成功/失敗の記録
- API レスポンス時間の監視
- データ品質チェック

### 8.2 アラート設定

- データ収集失敗時の通知
- API エラー時の通知
- データ欠損検出時の通知

## 9. 今後の拡張計画

### 9.1 他のセンチメント指標

- Social media sentiment
- News sentiment analysis
- Google Trends data

### 9.2 データ分析機能

- センチメント指標の可視化
- 価格との相関分析
- 予測モデルへの統合

## 10. 実装優先度

**高優先度**

1. データモデル・リポジトリ実装
2. Fear & Greed Index サービス実装
3. 基本的な API エンドポイント

**中優先度** 4. 履歴データ収集機能 5. フロントエンド統合 6. 定期実行機能

**低優先度** 7. 高度な特徴量エンジニアリング 8. 可視化機能 9. 他のセンチメント指標追加

この計画に基づいて段階的に実装を進めることで、安定した Fear & Greed Index データ収集システムを構築できます。
