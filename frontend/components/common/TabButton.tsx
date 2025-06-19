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
      return "bg-gray-100 text-gray-400 cursor-not-allowed";
    }

    switch (variant) {
      case "primary":
        return isActive
          ? "bg-gradient-to-r from-blue-600 to-blue-700 text-white shadow-lg shadow-blue-500/25 border-blue-600"
          : "bg-white text-gray-700 border-gray-200 hover:bg-gray-50 hover:border-gray-300 hover:shadow-md";

      case "secondary":
        return isActive
          ? "bg-gradient-to-r from-slate-800 to-slate-900 text-white shadow-lg shadow-slate-500/25 border-slate-800"
          : "bg-slate-50 text-slate-700 border-slate-200 hover:bg-slate-100 hover:border-slate-300 hover:shadow-md";

      case "minimal":
        return isActive
          ? "bg-blue-50 text-blue-700 border-blue-200 shadow-sm"
          : "bg-transparent text-gray-600 border-transparent hover:bg-gray-50 hover:text-gray-800";

      default:
        return "bg-white text-gray-700 border-gray-200 hover:bg-gray-50";
    }
  };

  const getSizeClasses = () => {
    switch (size) {
      case "sm":
        return "px-3 py-1.5 text-xs";
      case "md":
        return "px-4 py-2.5 text-sm";
      case "lg":
        return "px-6 py-3 text-base";
      default:
        return "px-4 py-2.5 text-sm";
    }
  };

  const baseClasses = `
    relative inline-flex items-center justify-center gap-2 
    border rounded-lg font-semibold tracking-wide
    transition-all duration-300 ease-out
    focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:ring-offset-2
    transform hover:scale-105 active:scale-95
    ${!disabled && isActive ? "animate-pulse" : ""}
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
      {/* Active indicator */}
      {isActive && (
        <div className="absolute inset-0 rounded-lg bg-gradient-to-r from-transparent via-white/20 to-transparent animate-shimmer" />
      )}

      {/* Icon */}
      {icon && <span className="flex-shrink-0">{icon}</span>}

      {/* Label */}
      <span className="relative z-10 whitespace-nowrap">{label}</span>

      {/* Badge */}
      {badge && (
        <span
          className={`
          ml-1 px-2 py-0.5 text-xs font-bold rounded-full
          ${isActive ? "bg-white/20 text-white" : "bg-blue-100 text-blue-800"}
        `}
        >
          {badge}
        </span>
      )}

      {/* Glow effect for active state */}
      {isActive && !disabled && (
        <div className="absolute inset-0 rounded-lg blur-xl opacity-30 bg-gradient-to-r from-blue-400 to-blue-600 -z-10" />
      )}
    </button>
  );
};

export default TabButton;
