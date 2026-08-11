/**
 * 汎用的な型定義
 *
 * アプリケーション全体で使用される共通の型定義をここに集約します。
 */

/**
 * テーブルカラムの定義
 */
export interface TableColumn<T> {
  /** カラムのキー */
  key: keyof T;
  /** カラムのヘッダー表示名 */
  header: string;
  /** カラムの幅（CSS値） */
  width?: string;
  /** ソート可能かどうか */
  sortable?: boolean;
  /** セルの値をフォーマットする関数（値は非破壊で整形表示する想定） */
  formatter?: (value: any, row: T) => React.ReactNode;
  /** セルのクラス名 */
  cellClassName?: string;
}

/**
 * データテーブルのプロパティ
 */
export interface DataTableProps<T> {
  /** テーブルデータ */
  data: T[];
  /** カラム定義 */
  columns: TableColumn<T>[];
  /** テーブルのタイトル */
  title?: string;
  /** ローディング状態 */
  loading?: boolean;
  /** エラーメッセージ */
  error?: string;
  /** 1ページあたりの表示件数（未指定時は実装側のデフォルト使用） */
  pageSize?: number;
  /** CSVエクスポート機能を有効にするか */
  enableExport?: boolean;
  /** 検索機能を有効にするか */
  enableSearch?: boolean;
  /** 検索対象のキー（enableSearch=true のときに参照） */
  searchKeys?: (keyof T)[];
  /** テーブルのクラス名 */
  className?: string;
}
