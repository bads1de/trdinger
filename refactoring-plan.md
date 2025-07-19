# フロントエンド・バックエンド連携リファクタリング計画書

## 1. 目的

現在、フロントエンド(Next.js)とバックエンド(FastAPI)の間に Next.js API Routes が介在しており、コードの冗長性と開発の複雑性を生んでいる。
このリファクタリングでは、Next.js API Routes を廃止し、フロントエンドから直接バックエンド API を呼び出す構成に変更することで、アーキテクチャをシンプルにし、開発効率とメンテナンス性を向上させる。

## 2. 計画

### ステップ 1: バックエンド(FastAPI)に CORS 設定を追加

フロントエンドのオリジン (`http://localhost:3000`) からの API リクエストを許可するため、FastAPI アプリケーションに CORS ミドルウェアを追加する。

- **対象ファイル**: `backend/app/main.py`
- **修正内容**:
  - `FastAPIMiddleware` から `CORSMiddleware` をインポートする。
  - `app.add_middleware` を使用して、`CORSMiddleware` を設定する。
  - 許可するオリジンとして `http://localhost:3000` を指定する。

```python
# backend/app/main.py への追加コード例

from fastapi.middleware.cors import CORSMiddleware

# ... FastAPIのappインスタンス作成後 ...

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### ステップ 2: フロントエンドの API 呼び出しを修正

Next.js の API ルートを介さず、直接 FastAPI のエンドポイントを叩くように変更する。

1.  **環境変数の設定**

    - **対象ファイル**: `frontend/.env.local` (なければ作成)
    - **修正内容**: バックエンド API のベース URL を環境変数として定義する。

    ```
    NEXT_PUBLIC_BACKEND_API_URL=http://localhost:8000
    ```

2.  **API 呼び出し箇所の修正**

    - **対象**: `useApiCall`フックを使用しているコンポーネントや、`fetch`を直接使っている箇所。
    - **修正内容**:

      - API のエンドポイントを相対パス (`/api/...`) から、環境変数を使った絶対パス (`${process.env.NEXT_PUBLIC_BACKEND_API_URL}/api/...`) に変更する。
      - 例えば、`useApiCall` の呼び出しは以下のようになる。

      ```typescript
      // 変更前
      execute('/api/data/funding-rates', { ... });

      // 変更後
      const backendApiUrl = process.env.NEXT_PUBLIC_BACKEND_API_URL;
      execute(`${backendApiUrl}/api/funding-rates`, { ... });
      ```

    - `constants.ts`などで定義されている `BACKEND_API_URL` も `process.env.NEXT_PUBLIC_BACKEND_API_URL` を参照するように修正する。

### ステップ 3: 不要な Next.js API Routes を削除

リファクタリングが完了し、直接バックエンドを呼び出すように変更されたら、不要になった Next.js の API ルートファイルを削除する。

- **対象ディレクトリ**: `frontend/app/api/`
- **作業内容**: このディレクトリ内の、バックエンドへのプロキシとしてのみ機能していたファイルを安全に削除する。

### ステップ 4: 動作確認

アプリケーション全体が正常に動作することを確認する。

- データ取得、表示、更新などの主要機能が問題なく動作するか。
- ブラウザのコンソールで CORS エラーやその他のネットワークエラーが発生していないか。

## 3. タイムライン

- **ステップ 1**: 0.5 時間
- **ステップ 2**: 2 時間
- **ステップ 3**: 0.5 時間
- **ステップ 4**: 1 時間
- **合計**: 4 時間

---
