/**
 * Jestセットアップファイル
 *
 * Jestテスト実行前に読み込まれるグローバル設定ファイルです。
 * テスト環境の初期化やグローバルモックの設定に使用されます。
 *
 * @see https://jestjs.io/docs/configuration#setupfilesafterenv-array
 */

// React Testing LibraryのカスタムJestマッチャーをインポート
// toBeInTheDocument()、toHaveClass()などの便利なアサーションを提供
import '@testing-library/jest-dom'

// Next.js APIルート用のWeb APIモック
// Request、Responseオブジェクトをグローバルに設定
Object.defineProperty(global, 'Request', {
  value: class MockRequest {
    constructor(url, options = {}) {
      this.url = url;
      this.method = options.method || 'GET';
      this.headers = new Map(Object.entries(options.headers || {}));
    }
  },
});

Object.defineProperty(global, 'Response', {
  value: class MockResponse {
    constructor(body, init = {}) {
      this.body = body;
      this.status = init.status || 200;
      this.statusText = init.statusText || 'OK';
      this.headers = new Map(Object.entries(init.headers || {}));
    }

    json() {
      return Promise.resolve(typeof this.body === 'string' ? JSON.parse(this.body) : this.body);
    }

    text() {
      return Promise.resolve(typeof this.body === 'string' ? this.body : JSON.stringify(this.body));
    }
  },
});

// NextResponseのモック
const { NextResponse } = require('next/server');

// NextResponse.jsonをモック
jest.mock('next/server', () => ({
  NextResponse: {
    json: (data, init = {}) => {
      const response = {
        status: init.status || 200,
        statusText: init.statusText || 'OK',
        headers: new Map(Object.entries({
          'Content-Type': 'application/json',
          ...init.headers,
        })),
        json: () => Promise.resolve(data),
        text: () => Promise.resolve(JSON.stringify(data)),
      };
      return response;
    },
  },
}));
