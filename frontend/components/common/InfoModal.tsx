import React from "react";
import { X, Info } from "lucide-react";
import Modal from "./Modal";

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
  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      title={undefined}
      showCloseButton={false}
      className="bg-secondary-950 rounded-lg shadow-xl max-w-2xl w-full max-h-[80vh] overflow-hidden border border-secondary-700"
    >
      <div className="flex items-center justify-between p-5 border-b border-secondary-700">
        <div className="flex items-center space-x-3">
          <Info className="h-6 w-6 text-blue-400" />
          <h2 className="text-xl font-bold text-secondary-100">{title}</h2>
        </div>
        <button
          onClick={onClose}
          className="p-2 hover:bg-secondary-700 rounded-lg transition-colors"
        >
          <X className="w-6 h-6 text-secondary-400" />
        </button>
      </div>
      <div className="p-6 overflow-y-auto max-h-[calc(80vh-130px)] text-secondary-300 prose prose-invert prose-sm">
        {children}
      </div>
      <div className="flex items-center justify-end p-4 border-t border-secondary-700 bg-secondary-900">
        <button
          onClick={onClose}
          className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-md transition-colors"
        >
          閉じる
        </button>
      </div>
    </Modal>
  );
};

export default InfoModal;
