"use client";

import React from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogClose,
} from "@/components/ui/dialog";
import { cn } from "@/lib/utils";
import { ModalProps, ModalSize } from "@/types/common";

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
  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent
        className={cn(
          "fixed left-[50%] top-[50%] z-50 grid w-full translate-x-[-50%] translate-y-[-50%] gap-4 shadow-lg duration-200 data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95 data-[state=closed]:slide-out-to-left-1/2 data-[state=closed]:slide-out-to-top-[48%] data-[state=open]:slide-in-from-left-1/2 data-[state=open]:slide-in-from-top-[48%] sm:rounded-lg",
          "bg-black rounded-lg shadow-2xl border border-black flex flex-col overflow-hidden",
          sizeClasses[size],
          "max-h-[90vh]",
          className
        )}
        onEscapeKeyDown={closeOnEscape ? undefined : (e) => e.preventDefault()}
        onPointerDownOutside={closeOnOverlayClick ? undefined : (e) => e.preventDefault()}
      >
        {(title || showCloseButton) && (
          <DialogHeader className={cn("p-6 border-b border-gray-700", headerClassName)}>
            {title && (
              <DialogTitle className="text-xl font-semibold text-white">
                {title}
              </DialogTitle>
            )}
            {showCloseButton && (
              <DialogClose
                className={cn(
                  "absolute right-4 top-4 rounded-sm opacity-70 ring-offset-background transition-opacity hover:opacity-100 focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:pointer-events-none data-[state=open]:bg-accent data-[state=open]:text-muted-foreground",
                  "p-2 text-gray-400 hover:text-white hover:bg-gray-800 rounded-lg transition-colors focus:ring-blue-500"
                )}
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
                <span className="sr-only">Close</span>
              </DialogClose>
            )}
          </DialogHeader>
        )}

        <div
          className={cn(
            "flex-1 overflow-y-auto p-6",
            contentClassName
          )}
        >
          {children}
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default Modal;
