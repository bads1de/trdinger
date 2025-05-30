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

// DOM環境の設定
import { TextEncoder, TextDecoder } from 'util';

// グローバルにTextEncoderとTextDecoderを設定
global.TextEncoder = TextEncoder;
global.TextDecoder = TextDecoder;

// DOM環境のセットアップ
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: jest.fn().mockImplementation(query => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: jest.fn(), // deprecated
    removeListener: jest.fn(), // deprecated
    addEventListener: jest.fn(),
    removeEventListener: jest.fn(),
    dispatchEvent: jest.fn(),
  })),
});

// ResizeObserverのモック
global.ResizeObserver = jest.fn().mockImplementation(() => ({
  observe: jest.fn(),
  unobserve: jest.fn(),
  disconnect: jest.fn(),
}));

// IntersectionObserverのモック
global.IntersectionObserver = jest.fn().mockImplementation(() => ({
  observe: jest.fn(),
  unobserve: jest.fn(),
  disconnect: jest.fn(),
}));

// URL.createObjectURLのモック
global.URL.createObjectURL = jest.fn(() => 'mocked-url');
global.URL.revokeObjectURL = jest.fn();

// Blobのモック
global.Blob = jest.fn().mockImplementation((content, options) => ({
  content,
  options,
  size: content ? content.join('').length : 0,
  type: options?.type || '',
}));

// DOM要素の作成をモック
const originalCreateElement = document.createElement;
document.createElement = jest.fn().mockImplementation((tagName) => {
  const element = originalCreateElement.call(document, tagName);

  // aタグの場合、clickメソッドをモック
  if (tagName === 'a') {
    element.click = jest.fn();
    element.setAttribute = jest.fn();
    Object.defineProperty(element, 'style', {
      value: { visibility: '' },
      writable: true,
    });
  }

  return element;
});

// document.bodyのappendChild/removeChildをモック
const originalAppendChild = document.body.appendChild;
const originalRemoveChild = document.body.removeChild;

document.body.appendChild = jest.fn().mockImplementation((node) => {
  return originalAppendChild.call(document.body, node);
});

document.body.removeChild = jest.fn().mockImplementation((node) => {
  return originalRemoveChild.call(document.body, node);
});

// 環境変数のモック設定
process.env.BACKEND_URL = 'http://localhost:8000';
process.env.NODE_ENV = 'test';

// fetch関数のモック設定
global.fetch = jest.fn(() =>
  Promise.resolve({
    ok: true,
    status: 200,
    json: () => Promise.resolve({}),
    text: () => Promise.resolve(''),
  })
);

// Next.js APIルート用のWeb APIモック
// Request、Responseオブジェクトをグローバルに設定（既に存在しない場合のみ）
if (!global.Request) {
  Object.defineProperty(global, 'Request', {
    value: class MockRequest {
      constructor(url, options = {}) {
        this.url = url;
        this.method = options.method || 'GET';
        this.headers = new Map(Object.entries(options.headers || {}));
      }
    },
  });
}

if (!global.Response) {
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
}

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
