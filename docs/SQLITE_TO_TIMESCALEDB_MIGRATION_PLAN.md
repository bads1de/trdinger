# SQLite から TimescaleDB への移行プラン

## 📋 概要

現在の Trdinger プロジェクトは開発環境で SQLite を使用していますが、時系列データの効率的な処理とスケーラビリティ向上のため、TimescaleDB への移行を実施します。

## 🎯 移行の目的

### 現在の課題

- **SQLite の制約**: 大量の時系列データ処理に限界
- **パフォーマンス**: 複数時間軸の OHLCV データクエリが遅い
- **スケーラビリティ**: データ量増加に対する拡張性不足
- **並行処理**: 複数ユーザーの同時アクセス制限

### TimescaleDB の利点

- **時系列データ最適化**: ハイパーテーブルによる自動パーティショニング
- **高速クエリ**: 時間ベースのクエリ最適化
- **データ圧縮**: 自動圧縮によるストレージ効率化
- **PostgreSQL 互換**: 既存の SQLAlchemy コードとの高い互換性

## 🔍 現在のシステム分析

### データベース構造

```
📊 主要テーブル:
- ohlcv_data (時系列データ - 最重要)
- funding_rate_data (時系列データ)
- open_interest_data (時系列データ)
- data_collection_log (ログデータ)
- ga_experiment (実験データ)
- generated_strategy (戦略データ)
- strategy_showcase (ショーケースデータ)
```

### 現在の設定

- **データベース URL**: `sqlite:///./trdinger.db` (デフォルト)
- **ORM**: SQLAlchemy 2.0.23
- **マイグレーション**: Alembic 1.13.1 (インストール済み、未設定)
- **接続プール**: QueuePool 設定済み

### 依存関係

- `psycopg2-binary==2.9.9` (既にインストール済み)
- PostgreSQL 互換のコードが一部実装済み

## 📋 移行計画

### フェーズ 1: 環境準備 (1-2 日)

#### 1.1 TimescaleDB 環境構築

```bash
# Docker Composeでの環境構築
version: '3.8'
services:
  timescaledb:
    image: timescale/timescaledb:latest-pg15
    environment:
      POSTGRES_DB: trdinger
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: your_password
    ports:
      - "5432:5432"
    volumes:
      - timescaledb_data:/var/lib/postgresql/data
```

#### 1.2 設定ファイル更新

- `.env` ファイルの更新
- `settings.py` の TimescaleDB 対応確認
- 接続文字列の変更

### フェーズ 2: スキーマ移行 (2-3 日)

#### 2.1 Alembic マイグレーション設定

```bash
# Alembic初期化 (migrationsディレクトリが空の場合)
alembic init alembic

# 初期マイグレーション作成
alembic revision --autogenerate -m "Initial migration to TimescaleDB"
```

#### 2.2 ハイパーテーブル設定

```sql
-- OHLCVデータをハイパーテーブル化
SELECT create_hypertable('ohlcv_data', 'timestamp');

-- ファンディングレートデータをハイパーテーブル化
SELECT create_hypertable('funding_rate_data', 'funding_timestamp');

-- オープンインタレストデータをハイパーテーブル化
SELECT create_hypertable('open_interest_data', 'data_timestamp');
```

#### 2.3 インデックス最適化

```sql
-- 時系列クエリ最適化用インデックス
CREATE INDEX idx_ohlcv_symbol_time ON ohlcv_data (symbol, timestamp DESC);
CREATE INDEX idx_ohlcv_timeframe_time ON ohlcv_data (timeframe, timestamp DESC);
```

### フェーズ 3: データ移行 (1-2 日)

#### 3.1 データエクスポート

```python
# SQLiteからデータエクスポート
def export_sqlite_data():
    # 既存のSQLiteデータを読み込み
    # CSVまたはJSONファイルとして出力
    pass
```

#### 3.2 データインポート

```python
# TimescaleDBへデータインポート
def import_to_timescaledb():
    # エクスポートしたデータを読み込み
    # バッチ処理でTimescaleDBに挿入
    pass
```

### フェーズ 4: アプリケーション更新 (2-3 日)

#### 4.1 接続設定更新

- `database/connection.py` の更新
- 環境変数の設定
- 接続プールの最適化

#### 4.2 クエリ最適化

- 時系列クエリの最適化
- ハイパーテーブル機能の活用
- パフォーマンス向上の実装

#### 4.3 リポジトリ層の更新

- TimescaleDB 固有機能の活用
- バッチ処理の最適化
- エラーハンドリングの強化

### フェーズ 5: テスト・検証 (2-3 日)

#### 5.1 機能テスト

- 全 API エンドポイントの動作確認
- データ整合性の検証
- パフォーマンステスト

#### 5.2 負荷テスト

- 大量データでの性能測定
- 並行アクセステスト
- メモリ使用量の監視

## 🛠️ 実装詳細

### 必要なファイル変更

#### 1. 環境設定ファイル

```
📁 変更対象:
- .env.example
- backend/app/config/settings.py
- docker-compose.yml (新規作成)
```

#### 2. データベース関連

```
📁 変更対象:
- backend/database/connection.py
- backend/database/models.py (ハイパーテーブル対応)
- backend/database/migrations/ (新規マイグレーション)
```

#### 3. リポジトリ層

```
📁 変更対象:
- backend/database/repositories/*.py
- backend/app/core/utils/database_utils.py
```

### 新規作成ファイル

#### 1. マイグレーションスクリプト

```
📁 新規作成:
- backend/scripts/migrate_to_timescaledb.py
- backend/scripts/verify_migration.py
- backend/database/migrations/001_initial_timescaledb.py
```

#### 2. Docker 設定

```
📁 新規作成:
- docker-compose.yml
- docker-compose.dev.yml
- .dockerignore
```

## ⚠️ リスク管理

### 潜在的リスク

1. **データ損失**: 移行中のデータ破損
2. **ダウンタイム**: サービス停止時間
3. **互換性問題**: PostgreSQL 固有の問題
4. **パフォーマンス**: 期待した性能向上が得られない

### 対策

1. **完全バックアップ**: 移行前の全データバックアップ（SQLite ファイルのコピー）
2. **段階的移行**: フェーズ分けによるリスク分散
3. **ロールバック計画**: 問題発生時の復旧手順（SQLite への復帰）
4. **テスト環境**: 本番前の十分な検証
5. **データ検証**: 移行前後のデータ整合性チェック

## 📊 期待される効果

### パフォーマンス向上

- **クエリ速度**: 時系列クエリで 50-80%の高速化
- **データ圧縮**: ストレージ使用量 30-50%削減
- **並行処理**: 複数ユーザー同時アクセス対応

### 機能拡張

- **リアルタイム分析**: 連続集計機能
- **データ保持ポリシー**: 自動データアーカイブ
- **高度な時系列関数**: TimescaleDB 固有機能

## 🗓️ スケジュール

```
週1: 環境準備・設計
├── TimescaleDB環境構築
├── 設定ファイル準備
└── マイグレーション計画詳細化

週2: 実装・移行
├── スキーマ移行
├── データ移行
└── アプリケーション更新

週3: テスト・最適化
├── 機能テスト
├── パフォーマンステスト
└── 本番デプロイ準備
```

## ✅ 成功基準

1. **機能完全性**: 全既存機能が正常動作
2. **データ整合性**: 移行前後でデータ一致
3. **パフォーマンス**: クエリ速度 50%以上向上
4. **安定性**: 24 時間連続稼働テスト成功

## 📚 参考資料

- [TimescaleDB 公式ドキュメント](https://docs.timescale.com/)
- [PostgreSQL 移行ガイド](https://www.postgresql.org/docs/current/migration.html)
- [SQLAlchemy TimescaleDB 対応](https://docs.sqlalchemy.org/en/20/dialects/postgresql.html)

---

**次のステップ**: このプランの承認後、フェーズ 1 の環境準備から開始します。
