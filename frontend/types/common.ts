/**
 * 汎用的な型定義
 *
 * アプリケーション全体で使用される共通の型定義をここに集約します。
 */

/**
 * モーダルのサイズ
 */
export type ModalSize = "sm" | "md" | "lg" | "xl" | "2xl" | "full";

/**
 * モーダルコンポーネントのプロパティ
 */
export interface ModalProps {
  /** モーダルの表示状態 */
  isOpen: boolean;
  /** モーダルを閉じる関数 */
  onClose: () => void;
  /** モーダルのタイトル */
  title?: string;
  /** モーダルのサイズ */
  size?: ModalSize;
  /** 外側クリックで閉じるかどうか（未指定時は true を推奨） */
  closeOnOverlayClick?: boolean;
  /** ESCキーで閉じるかどうか（未指定時は true を推奨） */
  closeOnEscape?: boolean;
  /** 閉じるボタンを表示するかどうか（未指定時は表示推奨） */
  showCloseButton?: boolean;
  /** モーダルの内容 */
  children: React.ReactNode;
  /** 追加のクラス名 */
  className?: string;
  /** ヘッダーの追加クラス名 */
  headerClassName?: string;
  /** コンテンツエリアの追加クラス名 */
  contentClassName?: string;
}

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

/**
 * API エラーレスポンス
 */
export interface APIError {
  /** 常に false 固定 */
  success: false;
  /** エラー要約 */
  error: string;
  /** 詳細説明（スタックや補足など任意） */
  detail?: string;
  /** 発生時刻（ISO文字列） */
  timestamp: string;
}
