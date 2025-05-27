# データベース自動初期化機能

## 概要

OHLCVデータ取得時にデータベースが初期化されていない場合、自動的に初期化を行う機能を実装しました。

## 実装内容

### 1. データベース接続モジュールの拡張 (`database/connection.py`)

#### 新規追加関数

- **`check_db_initialized()`**: データベースが初期化されているかチェック
  - SQLiteとPostgreSQLの両方に対応
  - OHLCVDataテーブルの存在を確認

- **`ensure_db_initialized()`**: データベース初期化を保証
  - 接続テスト → 初期化チェック → 必要に応じて初期化実行
  - 初期化後の確認まで含む完全な保証機能

### 2. データ収集クラスの修正 (`data_collector/collector.py`)

#### DataCollectorクラス
- **コンストラクタ**: インスタンス作成時に自動初期化チェック
- **collect_historical_data()**: 過去データ収集前の初期化確認（削除済み - 重複回避）
- **collect_latest_data()**: 最新データ収集前の初期化確認（削除済み - 重複回避）

#### 便利関数
- **`collect_btc_daily_data()`**: BTC日足データ収集前の初期化確認（削除済み - 重複回避）

### 3. リポジトリクラスの修正 (`database/repository.py`)

#### SQLite対応の改善
- **`insert_ohlcv_data()`**: SQLiteとPostgreSQLで異なる重複処理を実装
  - SQLite: 一件ずつINSERT OR IGNORE相当の処理
  - PostgreSQL: ON CONFLICT DO NOTHINGを使用

## 動作フロー

```mermaid
graph TD
    A[DataCollector インスタンス作成] --> B[ensure_db_initialized() 呼び出し]
    B --> C{データベース接続テスト}
    C -->|失敗| D[エラー終了]
    C -->|成功| E{初期化チェック}
    E -->|初期化済み| F[正常終了]
    E -->|未初期化| G[init_db() 実行]
    G --> H{初期化確認}
    H -->|成功| I[正常終了]
    H -->|失敗| J[エラー終了]
    F --> K[OHLCVデータ収集処理]
    I --> K
```

## 使用例

### 基本的な使用方法

```python
from data_collector.collector import DataCollector

# インスタンス作成時に自動初期化
collector = DataCollector()

# データ収集（DBが自動初期化済み）
count = await collector.collect_historical_data(
    symbol="BTC/USDT",
    timeframe="1d",
    days_back=30
)
```

### 手動での初期化確認

```python
from database.connection import ensure_db_initialized

# 手動で初期化を保証
if ensure_db_initialized():
    print("データベースは初期化済みです")
else:
    print("データベースの初期化に失敗しました")
```

## エラーハンドリング

### 初期化失敗時の動作
- `RuntimeError("データベースの初期化に失敗しました")` を発生
- ログに詳細なエラー情報を出力

### 対応するエラーケース
1. データベース接続失敗
2. テーブル作成失敗
3. 初期化後の確認失敗

## テスト

### 自動テストスクリプト
- `test_db_auto_init.py`: 基本的な自動初期化機能のテスト
- `test_ohlcv_collection_with_auto_init.py`: OHLCV収集での自動初期化テスト

### テスト実行方法

```bash
cd backend
python test_db_auto_init.py
python test_ohlcv_collection_with_auto_init.py
```

## 対応データベース

- **SQLite**: 開発環境用（デフォルト）
- **PostgreSQL**: 本番環境用

## 注意事項

1. **パフォーマンス**: 初期化チェックはDataCollectorインスタンス作成時のみ実行
2. **スレッドセーフティ**: SQLAlchemyのセッション管理に依存
3. **エラー処理**: 初期化失敗時は即座にエラーを発生させる設計

## 今後の改善点

1. **キャッシュ機能**: 初期化状態のキャッシュによる性能向上
2. **マイグレーション**: データベーススキーマ変更時の自動マイグレーション
3. **ヘルスチェック**: 定期的なデータベース状態確認機能
