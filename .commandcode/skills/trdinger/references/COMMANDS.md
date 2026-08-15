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
| `--no-validation` | flag | false | WFA自動検証を無効化 |
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

## 制約

- 同時実行上限: running が 2 実験に達すると `exp run` は拒否される
- `population_size` < 二段階選抜候補数（デフォルト5）だと ConfigValidator エラー
- CLI は同期実行。サーバー経由のバックグラウンド実行は従来どおり API で
