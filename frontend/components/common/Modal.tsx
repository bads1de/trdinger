/**
 * 汎用モーダルコンポーネント
 *
 * ポータル機能、オーバーレイ、アニメーション、アクセシビリティ対応を含む
 * 汎用的なモーダルコンポーネントです。
 *
 */

"use client";

import React, { useEffect, useRef } from "react";
import { createPortal } from "react-dom";

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
  /** 外側クリックで閉じるかどうか */
  closeOnOverlayClick?: boolean;
  /** ESCキーで閉じるかどうか */
  closeOnEscape?: boolean;
  /** 閉じるボタンを表示するかどうか */
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
 * サイズ別のクラス定義
 */
const sizeClasses: Record<ModalSize, string> = {
  sm: "max-w-md",
  md: "max-w-lg",
  lg: "max-w-2xl",
  xl: "max-w-4xl",
  "2xl": "max-w-6xl",
  full: "max-w-full mx-4",
};

/**
 * 汎用モーダルコンポーネント
 */
const Modal: React.FC<ModalProps> = ({
  isOpen,
  onClose,
  title,
  size = "lg",
  closeOnOverlayClick = true,
  closeOnEscape = true,
  showCloseButton = true,
  children,
  className = "",
  headerClassName = "",
  contentClassName = "",
}) => {
  const modalRef = useRef<HTMLDivElement>(null);

  // ESCキーでモーダルを閉じる
  useEffect(() => {
    if (!closeOnEscape || !isOpen) return;

    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onClose();
      }
    };

    document.addEventListener("keydown", handleEscape);
    return () => document.removeEventListener("keydown", handleEscape);
  }, [isOpen, closeOnEscape, onClose]);

  // モーダルが開いている間はbodyのスクロールを無効化
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = "hidden";
    } else {
      document.body.style.overflow = "unset";
    }

    return () => {
      document.body.style.overflow = "unset";
    };
  }, [isOpen]);

  // フォーカス管理
  useEffect(() => {
    if (isOpen && modalRef.current) {
      modalRef.current.focus();
    }
  }, [isOpen]);

  // オーバーレイクリックでモーダルを閉じる
  const handleOverlayClick = (event: React.MouseEvent<HTMLDivElement>) => {
    if (closeOnOverlayClick && event.target === event.currentTarget) {
      onClose();
    }
  };

  if (!isOpen) return null;

  const modalContent = (
    <div
      className={`fixed inset-0 z-50 flex items-center justify-center p-4 ${
        isOpen ? "animate-fade-in" : "animate-fade-out"
      }`}
      onClick={handleOverlayClick}
    >
      {/* オーバーレイ背景 */}
      <div className="absolute inset-0 bg-black bg-opacity-50 backdrop-blur-sm" />

      {/* モーダルコンテンツ */}
      <div
        ref={modalRef}
        tabIndex={-1}
        className={`
          relative w-full ${sizeClasses[size]} max-h-[90vh] 
          bg-gray-900 rounded-lg shadow-2xl border border-gray-700
          flex flex-col overflow-hidden
          ${isOpen ? "animate-scale-in" : "animate-scale-out"}
          ${className}
        `}
        role="dialog"
        aria-modal="true"
        aria-labelledby={title ? "modal-title" : undefined}
      >
        {/* ヘッダー */}
        {(title || showCloseButton) && (
          <div
            className={`
              flex items-center justify-between p-6 border-b border-gray-700
              ${headerClassName}
            `}
          >
            {title && (
              <h2 id="modal-title" className="text-xl font-semibold text-white">
                {title}
              </h2>
            )}
            {showCloseButton && (
              <button
                onClick={onClose}
                className="
                  p-2 text-gray-400 hover:text-white hover:bg-gray-800 
                  rounded-lg transition-colors focus:outline-none focus:ring-2 
                  focus:ring-blue-500
                "
                aria-label="モーダルを閉じる"
              >
                <svg
                  className="w-5 h-5"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M6 18L18 6M6 6l12 12"
                  />
                </svg>
              </button>
            )}
          </div>
        )}

        {/* コンテンツエリア */}
        <div
          className={`
            flex-1 overflow-y-auto p-6
            ${contentClassName}
          `}
        >
          {children}
        </div>
      </div>
    </div>
  );

  // ポータルを使用してbody直下にレンダリング
  return createPortal(modalContent, document.body);
};

export default Modal;
