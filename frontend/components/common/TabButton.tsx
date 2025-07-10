import React from "react";

interface TabButtonProps {
  label: string;
  isActive: boolean;
  onClick: () => void;
  variant?: "primary" | "secondary" | "minimal";
  size?: "sm" | "md" | "lg";
  disabled?: boolean;
  icon?: React.ReactNode;
  badge?: string | number;
}

const TabButton: React.FC<TabButtonProps> = ({
  label,
  isActive,
  onClick,
  variant = "primary",
  size = "md",
  disabled = false,
  icon,
  badge,
}) => {
  const getVariantClasses = () => {
    if (disabled) {
      return "bg-gray-800 text-gray-600 cursor-not-allowed border-gray-700";
    }

   
    switch (variant) {
      case "primary":
        return isActive
          ? "bg-cyan-900/60 text-cyan-300 border-cyan-500 shadow-[0_0_12px_rgba(0,255,255,0.25)]"
          : "bg-gray-900/50 text-gray-400 border-gray-700 hover:bg-gray-800/70 hover:text-cyan-400 hover:border-cyan-600";
      case "secondary":
        return isActive
          ? "bg-fuchsia-900/60 text-fuchsia-300 border-fuchsia-500 shadow-[0_0_12px_rgba(217,70,239,0.25)]"
          : "bg-gray-900/50 text-gray-400 border-gray-700 hover:bg-gray-800/70 hover:text-fuchsia-400 hover:border-fuchsia-600";
      case "minimal":
        return isActive
          ? "bg-transparent text-cyan-300 border-b-2 border-cyan-400 rounded-none"
          : "bg-transparent text-gray-500 border-b-2 border-transparent rounded-none hover:text-cyan-400";
      default:
        return "bg-gray-900/50 text-gray-400 border-gray-700 hover:bg-gray-800/70";
    }
  };

  const getSizeClasses = () => {
    switch (size) {
      case "sm":
        return "px-3 py-1.5 text-xs";
      case "md":
        return "px-4 py-2 text-sm";
      case "lg":
        return "px-6 py-3 text-base";
      default:
        return "px-4 py-2 text-sm";
    }
  };

  const baseClasses = `
    relative inline-flex items-center justify-center gap-2 
    border rounded-md font-semibold tracking-wide
    transition-all duration-200 ease-out
    focus:outline-none focus:ring-2 focus:ring-cyan-500/50 focus:ring-offset-2 focus:ring-offset-gray-900
  `
    .replace(/\s+/g, " ")
    .trim();

  return (
    <button
      className={`${baseClasses} ${getSizeClasses()} ${getVariantClasses()}`}
      onClick={onClick}
      disabled={disabled}
      role="tab"
      aria-selected={isActive}
    >
      {icon && <span className="flex-shrink-0 z-10">{icon}</span>}
      <span className="relative z-10 whitespace-nowrap">{label}</span>
      {badge && (
        <span
          className={`
          ml-2 px-2 py-0.5 text-xs font-bold rounded-full z-10
          ${
            isActive
              ? "bg-cyan-400/20 text-cyan-200"
              : "bg-gray-700 text-gray-300"
          }
        `}
        >
          {badge}
        </span>
      )}
    </button>
  );
};

export default TabButton;
