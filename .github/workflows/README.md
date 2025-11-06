# CI/CD パイプライン

## 概要

Trdinger プロジェクトの CI/CD パイプラインは、GitHub Actions を使用してコード品質の維持とテストの自動化を実現しています。このパイプラインは、バックエンド（Python/FastAPI）とフロントエンド（Next.js/TypeScript）の両方に対して、コードの変更があるたびに自動的に品質チェックを実行します。

### 自動化されるプロセス

- **コードフォーマットチェック**: コーディング規約への準拠を確認
- **静的解析**: 型チェックとリンティングによるコード品質の検証
- **自動テスト**: ユニットテストと統合テストの実行
- **ビルド検証**: 本番環境でのビルドが成功することを確認
- **カバレッジレポート**: テストカバレッジの測定と報告

## ワークフローの説明

### 1. バックエンド CI (`backend-ci.yml`)

バックエンドの Python/FastAPI コードに対する継続的インテグレーションワークフローです。

#### トリガー条件

以下の条件でワークフローが自動実行されます：

- **Push**: `main`ブランチへの直接プッシュ時
- **Pull Request**: `main`ブランチへのプルリクエスト作成/更新時
- **対象ファイル**: `backend/**` 配下のファイル変更時のみ

#### 実行されるチェック

1. **Black フォーマットチェック**

   - コードが[Black](https://github.com/psf/black)のフォーマット規則に準拠しているか確認
   - 行長: 88 文字

2. **isort インポート順チェック**

   - インポート文が[isort](https://pycqa.github.io/isort/)の規則に従って整理されているか確認
   - プロファイル: Black 互換

3. **Flake8 リンティング**

   - [Flake8](https://flake8.pycqa.org/)によるコード品質チェック
   - PEP 8 準拠の確認

4. **MyPy 型チェック**

   - [MyPy](http://mypy-lang.org/)による静的型チェック
   - 厳密な型チェックモード

5. **Pytest テスト実行**
   - [pytest](https://docs.pytest.org/)によるユニットテストと統合テストの実行
   - カバレッジレポート生成（XML 形式）
   - 80%以上のカバレッジ目標

#### 使用するツールとバージョン

- Python: 3.11
- Black: 最新版
- isort: 最新版
- Flake8: 最新版
- MyPy: 最新版
- pytest: カバレッジプラグイン付き

### 2. フロントエンド CI (`frontend-ci.yml`)

フロントエンドの Next.js/TypeScript コードに対する継続的インテグレーションワークフローです。

#### トリガー条件

以下の条件でワークフローが自動実行されます：

- **Push**: `main`ブランチへの直接プッシュ時
- **Pull Request**: `main`ブランチへのプルリクエスト作成/更新時
- **対象ファイル**:
  - `frontend/**` 配下のファイル変更時
  - `.github/workflows/frontend-ci.yml` 自体の変更時

#### 実行されるチェック

1. **TypeScript 型チェック**

   - TypeScript コンパイラによる型検証
   - 厳密モードでの型チェック

2. **ESLint によるコード品質チェック**

   - [ESLint](https://eslint.org/)によるコード品質とスタイルの検証
   - React/Next.js 固有のルール適用

3. **Jest テストの実行**

   - [Jest](https://jestjs.io/)によるユニットテストの実行
   - カバレッジレポート生成
   - CI 環境での最適化実行

4. **本番ビルドの確認**
   - Next.js の本番ビルド実行
   - ビルド成果物（`.next`ディレクトリ）の生成確認

#### 使用するツールとバージョン

- Node.js: 20.x
- TypeScript: 最新版
- ESLint: Next.js 推奨設定
- Jest: React Testing Library 統合
- Next.js: 15.x

## ワークフローの動作

### 実行タイミング

両ワークフローは以下のタイミングで実行されます：

1. **コミット時**:

   - `main`ブランチへの直接プッシュ時に即座に実行
   - 対象ディレクトリのファイルが変更された場合のみ

2. **プルリクエスト時**:
   - プルリクエストの作成時
   - プルリクエストへの追加コミット時
   - 対象ファイルパスに変更がある場合のみ

### 並列実行

- バックエンドとフロントエンドのワークフローは独立して並列実行されます
- 各ワークフロー内の複数のチェックステップは順次実行されます
- ファイルパスフィルタにより、不要なワークフローは実行されません

### キャッシュ戦略

効率的な実行のため、以下のキャッシュが使用されます：

**バックエンド**:

- Pip キャッシュ: `~/.cache/pip`
- キャッシュキー: `pyproject.toml`のハッシュ値

**フロントエンド**:

- npm キャッシュ: `node_modules`
- キャッシュキー: `package-lock.json`のハッシュ値

## ローカルでの実行方法

CI 環境と同じチェックをローカルで実行する方法を説明します。

### バックエンド

```bash
# バックエンドディレクトリに移動
cd backend

# 依存関係のインストール（初回のみ）
pip install -e .[test,dev]

# すべてのチェックを順次実行
black --check .              # フォーマットチェック
isort --check-only .         # インポート順チェック
flake8 app/                  # リンティング
mypy .                       # 型チェック
pytest --cov=app --cov-report=term  # テスト実行

# または、すべてを一度に実行
black --check . && isort --check-only . && flake8 app/ && mypy . && pytest --cov=app
```

#### 自動修正コマンド

```bash
# コードを自動フォーマット（チェックではなく修正）
black .
isort .

# すべての品質チェックを実行（修正後）
flake8 app/ && mypy . && pytest
```

### フロントエンド

```bash
# フロントエンドディレクトリに移動
cd frontend

# 依存関係のインストール（初回のみ）
npm ci

# すべてのチェックを順次実行
npx tsc --noEmit            # 型チェック
npm run lint                # ESLintチェック
npm test -- --ci            # テスト実行（カバレッジ付き）
npm run build               # 本番ビルド

# Watchモードでテストを実行（開発中）
npm test -- --watch
```

#### 自動修正コマンド

```bash
# ESLintの自動修正
npm run lint -- --fix

# すべてのチェックを実行（修正後）
npx tsc --noEmit && npm run lint && npm test -- --ci && npm run build
```

## トラブルシューティング

### よくあるエラーと解決方法

#### バックエンド

**1. Black フォーマットエラー**

```
error: cannot format <file>: File '<file>' is not formatted correctly
```

**解決方法**:

```bash
cd backend
black .
```

**2. isort インポート順エラー**

```
ERROR: <file> Imports are incorrectly sorted
```

**解決方法**:

```bash
cd backend
isort .
```

**3. MyPy 型エラー**

```
error: Incompatible types in assignment
```

**解決方法**:

- 型アノテーションを追加または修正
- 必要に応じて`# type: ignore`コメントを追加（推奨しません）
- `mypy .`で詳細なエラーメッセージを確認

**4. Pytest 失敗**

```
FAILED tests/test_example.py::test_function
```

**解決方法**:

```bash
# 詳細な出力でテストを実行
pytest -v

# 特定のテストのみ実行
pytest tests/test_example.py::test_function -v

# デバッグモードで実行
pytest --pdb
```

**5. カバレッジ不足**

```
Coverage: XX% (required: 80%)
```

**解決方法**:

- テストされていないコードブロックを特定
- 新しいテストケースを追加
- カバレッジレポートを確認: `pytest --cov=app --cov-report=html`

#### フロントエンド

**1. TypeScript 型エラー**

```
error TS2322: Type 'string' is not assignable to type 'number'
```

**解決方法**:

- 型定義を修正
- 適切な型アサーションを使用
- `types/`ディレクトリの型定義を確認

**2. ESLint エラー**

```
error: 'variable' is assigned a value but never used
```

**解決方法**:

```bash
# 自動修正可能なエラーを修正
npm run lint -- --fix

# 手動で未使用の変数や関数を削除
```

**3. Jest テスト失敗**

```
FAIL components/__tests__/Example.test.tsx
```

**解決方法**:

```bash
# 特定のテストファイルのみ実行
npm test -- Example.test.tsx

# Watchモードでテスト
npm test -- --watch

# デバッグモードで実行
node --inspect-brk node_modules/.bin/jest --runInBand
```

**4. ビルドエラー**

```
Error: Build failed
```

**解決方法**:

- TypeScript 型エラーを修正
- ESLint エラーを解決
- 依存関係を再インストール: `rm -rf node_modules && npm ci`
- Next.js キャッシュをクリア: `rm -rf .next`

### ワークフロー失敗時の対処法

1. **GitHub Actions ログの確認**

   - リポジトリの「Actions」タブを開く
   - 失敗したワークフロー実行をクリック
   - 失敗したステップの詳細ログを確認

2. **ローカルでの再現**

   - 失敗したステップのコマンドをローカルで実行
   - 同じエラーが発生するか確認

3. **依存関係の問題**

   - キャッシュの問題の可能性がある場合、ワークフローを再実行
   - または、`pyproject.toml`/`package.json`の依存関係を更新

4. **環境の違い**

   - CI 環境（Ubuntu）とローカル環境の違いを確認
   - パス区切り文字、改行コードなどに注意

5. **プルリクエストの修正**
   - エラーを修正してコミット
   - プッシュすると自動的にワークフローが再実行される

## バッジの追加方法（オプション）

GitHub Actions のステータスバッジを README に追加して、ビルドステータスを視覚的に表示できます。

### バッジの形式

```markdown
![Backend CI](https://github.com/<username>/<repository>/workflows/Backend%20CI/badge.svg)
![Frontend CI](https://github.com/<username>/<repository>/workflows/フロントエンド%20CI/badge.svg)
```

### プロジェクトルート README への追加例

```markdown
# Trdinger

[![Backend CI](https://github.com/<username>/trading/workflows/Backend%20CI/badge.svg)](https://github.com/<username>/trading/actions?query=workflow%3A%22Backend+CI%22)
[![Frontend CI](https://github.com/<username>/trading/workflows/フロントエンド%20CI/badge.svg)](https://github.com/<username>/trading/actions?query=workflow%3A%22フロントエンド+CI%22)

遺伝的アルゴリズムと機械学習を組み合わせた暗号通貨取引戦略自動化システム
```

### カスタマイズオプション

バッジの URL パラメータでカスタマイズできます：

```markdown
<!-- 特定のブランチのステータス -->

![Backend CI](https://github.com/<username>/<repository>/workflows/Backend%20CI/badge.svg?branch=main)

<!-- 特定のイベントのステータス -->

![Backend CI](https://github.com/<username>/<repository>/workflows/Backend%20CI/badge.svg?event=push)
```

## 参考リンク

- [GitHub Actions ドキュメント](https://docs.github.com/ja/actions)
- [Python 用 GitHub Actions](https://docs.github.com/ja/actions/automating-builds-and-tests/building-and-testing-python)
- [Node.js 用 GitHub Actions](https://docs.github.com/ja/actions/automating-builds-and-tests/building-and-testing-nodejs)
- [ワークフロー構文](https://docs.github.com/ja/actions/using-workflows/workflow-syntax-for-github-actions)
- [キャッシュの使用](https://docs.github.com/ja/actions/using-workflows/caching-dependencies-to-speed-up-workflows)

## まとめ

この CI/CD パイプラインにより、以下が実現されます：

- ✅ コード品質の自動チェック
- ✅ テストの自動実行
- ✅ 問題の早期発見
- ✅ コードレビューの効率化
- ✅ 本番環境への安全なデプロイ準備

プルリクエストを作成する前に、必ずローカルですべてのチェックを実行することを推奨します。これにより、CI 環境での失敗を防ぎ、開発効率を向上させることができます。
