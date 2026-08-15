---
name: trdinger
description: Trdinger オートストラテジーのターミナル操作。GA 実験の実行・一覧・詳細・停止・削除、生成済み戦略の閲覧を CLI で行う。ユーザーが「実験を回して」「戦略を生成して」「GA を実行して」「trdinger」等と頼んだ時に使う。
argument-hint: "[exp|strategy] [subcommand] [options]"
compatibility: backend/.venv の Python 環境（uv 管理）が必要
---

# Trdinger CLI 操作

GA 実験と生成済み戦略をターミナルから操作する。サーバー（uvicorn）は不要。

## 前提

- 必ず `backend/` ディレクトリで `uv run trdinger ...` の形式で実行する
- 出力が文字化けする場合は `$env:PYTHONIOENCODING="utf-8"` を先に設定する（PowerShell）

## コマンド体系

### 実験の実行（最重要）

```powershell
cd backend
uv run trdinger exp run --name <実験名> --population 20 --generations 10 `
  --symbol BTC/USDT:USDT --timeframe 4h `
  --start-date 2024-01-01 --end-date 2024-06-30
```

**注意（必須）**:
- `population_size` は 8 以上を推奨（二段階選抜のデフォルト候補数 5 を下回ると検証エラー）
- クイック確認なら `--no-validation --generations 1 --population 8` で高速化
- 本番探索は `--generations 50 --population 100` 程度
- 実行は同期（終わるまで待つ）。長い実験は CI/cron や別ターミナルで

主要オプション:
| オプション | デフォルト | 意味 |
|---|---|---|
| `--population` / `-p` | 20 | 個体数（8以上推奨） |
| `--generations` / `-g` | 10 | 世代数 |
| `--elite-size` / `-e` | 2 | エリート保存数（個体数未満） |
| `--crossover-rate` | 0.8 | 交叉率 |
| `--mutation-rate` | 0.2 | 突然変異率 |
| `--symbol` / `-s` | BTC/USDT:USDT | 取引ペア |
| `--timeframe` / `-t` | 4h | 時間足 |
| `--start-date` | 2024-01-01 | BT開始日 |
| `--end-date` | 2024-06-30 | BT終了日 |
| `--no-validation` | false | WFA自動検証を無効化 |
| `--json` | false | 結果をJSONで出力 |

### 実験の管理

```powershell
uv run trdinger exp list                          # 一覧（status, fitness, progress）
uv run trdinger exp show <uuid>                   # 詳細
uv run trdinger exp stop <uuid>                   # 実行中実験の停止
uv run trdinger exp delete <uuid>                 # 削除（--yes で確認スキップ）
```

削除は実験・戦略・BT結果をカスケード削除する。実行中（running）の実験は削除不可。

### 戦略の閲覧

```powershell
uv run trdinger strategy list --limit 20          # 一覧（fitness順）
uv run trdinger strategy list --min-fitness 50    # fitness閾値フィルター
uv run trdinger strategy show auto_42             # 詳細（BT成績込み）
```

## ユーザーの意図 → コマンドの翻訳例

| 依頼 | 実行コマンド |
|---|---|
| 「実験を回して」 | `exp run --name <説明的な名前> --population 50 --generations 30` |
| 「クイックに試したい」 | `exp run --no-validation --population 8 --generations 3` |
| 「実験の進捗は？」 | `exp list` して running を確認 |
| 「この実験消して」 | `exp delete <uuid> --yes`（ユーザーが明示的に削除を望んだ場合のみ） |
| 「良い戦略ある？」 | `strategy list --min-fitness 50` → 上位を `strategy show` |

## よくあるエラーと対処

| エラー | 対処 |
|---|---|
| `二段階選抜候補数は個体数以下` | `--population` を 8 以上に上げる |
| `同時実行できるGA実験の上限` | 既に 2 実験が running。`exp stop` か完了を待つ |
| `ModuleNotFoundError: database` | `uv sync --all-extras` で再インストール |

## 詳細

CLI の全コマンドとオプションのリファレンスは [references/COMMANDS.md](references/COMMANDS.md) を参照。
