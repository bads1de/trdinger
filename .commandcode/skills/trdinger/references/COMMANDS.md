# Trdinger CLI コマンドリファレンス

`backend/` で `uv run trdinger <command>` の形式で実行する。

## exp run — GA実験の実行

```
uv run trdinger exp run [OPTIONS]
```

| オプション | 型 | デフォルト | 説明 |
|---|---|---|---|
| `--name` / `-n` | str | "CLI experiment" | 実験名 |
| `--population` / `-p` | int | 20 | 個体数（8以上推奨） |
| `--generations` / `-g` | int | 10 | 世代数 |
| `--elite-size` / `-e` | int | 2 | エリート保存数（個体数未満） |
| `--crossover-rate` | float | 0.8 | 交叉率（0-1） |
| `--mutation-rate` | float | 0.2 | 突然変異率（0-1） |
| `--symbol` / `-s` | str | BTC/USDT:USDT | 取引ペア |
| `--timeframe` / `-t` | str | 4h | 時間足 |
| `--start-date` | str | 2024-01-01 | バックテスト開始日 |
| `--end-date` | str | 2024-06-30 | バックテスト終了日 |
| `--initial-capital` | float | 100000.0 | 初期資本 |
| `--no-parallel` | flag | false | 並列評価を無効化 |
| `--no-validation` | flag | false | WFA自動検証を無効化 |
| `--no-seeds` | flag | false | シード戦略注入を無効化 |
| `--min-trades` | int | なし | 最小取引回数制約（0で無効化） |
| `--smoke` | flag | false | 高速スモークモード（最小構成） |
| `--mtf` | flag | false | マルチタイムフレーム指標を有効化 |
| `--mtf-timeframes` | str | 1d | MTFタイムフレーム（カンマ区切り） |
| `--mtf-probability` | float | 0.3 | MTF指標生成確率 |
| `--indicator-universe` | str | curated | インジケーターユニバース |
| `--max-indicators` | int | 10 | 1戦略あたりの最大インジケーター数 |
| `--min-non-price` | int | 0 | 非価格指標の最低数 |
| `--non-price-probability` | float | 0.3 | 非価格指標の選択確率 |
| `--max-conditions` | int | 3 | エントリー条件の最大数 |
| `--verbose` / `-v` | flag | false | 詳細ログ |
| `--json` | flag | false | 結果をJSONで出力 |

戻り値: 成功時 exit 0。`experiment_id`（UUID）を stdout に出力。検証エラー・上限超過時は exit 1。

## exp list — 実験一覧

```
uv run trdinger exp list [--json]
```

`[status] db_id name fitness=... progress=...` の形式。status は completed / running / stopped / failed。

## exp show — 実験詳細

```
uv run trdinger exp show <experiment_uuid> [--json]
```

引数はフロントエンド生成の UUID（`experiment_id`）。

## exp stop — 実行中実験の停止

```
uv run trdinger exp stop <experiment_uuid>
```

- アクティブエンジンがあれば停止シグナル送信
- 実行コンテキストが無く DB が running ならステータスを stopped に更新
- どちらも該当しなければ exit 1

## exp delete — 実験削除

```
uv run trdinger exp delete <experiment_uuid> [--yes]
```

- カスケード削除: 実験 → 戦略 → BT結果（他戦略から未参照のもののみ）
- running 中の実験は削除不可
- `--yes` を付けないと確認プロンプトが出る

## strategy list — 生成済み戦略一覧

```
uv run trdinger strategy list [--limit N] [--min-fitness F] [--json]
```

fitness 降順。`total_count` / `has_more` は JSON 出力でのみ確認可能。

## strategy show — 戦略詳細

```
uv run trdinger strategy show <auto_ID> [--json]
```

- `auto_42` 形式（または `42` でも可）
- BT結果にリンク済みなら成績（expected_return / sharpe / max_drawdown / win_rate）も表示

## data fetch — OHLCV履歴データの収集

```
uv run trdinger data fetch [--symbol S] [--timeframe T] [--force] [--start-date D] [--json]
```

- 取引所（Bybit）から履歴OHLCVを取得してDBへ保存（同期実行）
- `--force` / `-f`: 既存データを削除して再取得
- `--start-date`: 収集開始日（YYYY-MM-DD）。未指定は取引所の最大履歴
- 既にデータが存在し `--force` なしの場合は `status=exists` で何もしない

## data update — 差分更新

```
uv run trdinger data update [--symbol S] [--json]
```

DB末尾から現在時刻までの不足分を一括取得（OHLCV全時間足 + FR + OI）。

## data status — 収集状況の確認

```
uv run trdinger data status [--symbol S] [--timeframe T] [--json]
```

件数と最古・最新タイムスタンプを表示。

## data overview — 全データの総覧

```
uv run trdinger data overview [--json]
```

OHLCV / ファンディングレート / オープンインタレストの総件数を表示。

## data reset — データ削除

```
uv run trdinger data reset <all|ohlcv|funding-rates|open-interest> [--symbol S] [--yes]
```

- `--symbol` 指定時はそのシンボルの全データ種別を削除
- `--yes` / `-y` なしだと確認プロンプトが出る
- 削除は取り消せないため、実行前に `data overview` で現状確認を推奨

## 制約

- 同時実行上限: running が 2 実験に達すると `exp run` は拒否される
- `population_size` < 二段階選抜候補数（デフォルト5）だと ConfigValidator エラー
- CLI は同期実行。サーバー経由のバックグラウンド実行は従来どおり API で
