import React from "react";
import { X, Info } from "lucide-react";

interface InfoModalProps {
  isOpen: boolean;
  onClose: () => void;
  title: string;
  children: React.ReactNode;
}

const InfoModal: React.FC<InfoModalProps> = ({
  isOpen,
  onClose,
  title,
  children,
}) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* オーバーレイ */}
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* モーダルコンテンツ */}
      <div className="relative bg-background rounded-xl shadow-2xl max-w-lg w-full mx-4 border border-border overflow-hidden">
        {/* ヘッダー */}
        <div className="flex items-center justify-between p-6 border-b border-border bg-muted/40">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-primary/10 rounded-lg">
              <Info className="h-5 w-5 text-primary/80" />
            </div>
            <h2 className="text-lg font-semibold text-foreground">{title}</h2>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-muted rounded-lg transition-colors group"
            aria-label="閉じる"
          >
            <X className="w-5 h-5 text-muted-foreground group-hover:text-foreground" />
          </button>
        </div>

        {/* コンテンツ */}
        <div className="p-6">
          <div className="text-muted-foreground leading-relaxed">
            {children}
          </div>
        </div>

        {/* フッター */}
        <div className="flex justify-end p-4 border-t border-border bg-muted/30">
          <button
            onClick={onClose}
            className="px-6 py-2 bg-primary hover:bg-primary/90 text-primary-foreground rounded-lg transition-colors font-medium"
          >
            閉じる
          </button>
        </div>
      </div>
    </div>
  );
};

export default InfoModal;
