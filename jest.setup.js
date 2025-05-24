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
