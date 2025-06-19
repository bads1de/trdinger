import React from "react";

interface TabButtonProps {
  label: string;
  isActive: boolean;
  onClick: () => void;
}

const TabButton: React.FC<TabButtonProps> = ({ label, isActive, onClick }) => {
  const activeClasses = isActive
    ? "bg-blue-600 text-white"
    : "bg-gray-200 text-gray-800 hover:bg-gray-300";

  return (
    <button
      className={`px-4 py-2 rounded-md text-sm font-medium focus:outline-none transition-colors duration-200 ${activeClasses}`}
      onClick={onClick}
    >
      {label}
    </button>
  );
};

export default TabButton;
