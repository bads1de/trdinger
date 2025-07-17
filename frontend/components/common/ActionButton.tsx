import React from "react";
import { Button, ButtonProps } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import LoadingSpinner from "./LoadingSpinner"; // Import LoadingSpinner

interface ActionButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "secondary" | "success" | "warning" | "danger";
  size?: "sm" | "md" | "lg";
  icon?: React.ReactNode;
  loading?: boolean;
  loadingText?: string;
  children: React.ReactNode;
}

const ActionButton: React.FC<ActionButtonProps> = ({
  children,
  variant = "primary",
  size = "md",
  icon,
  loading = false,
  loadingText = "処理中...",
  className,
  disabled,
  ...props
}) => {
  const isDisabled = disabled || loading;

  const getButtonVariant = (): ButtonProps["variant"] => {
    switch (variant) {
      case "primary":
        return "default";
      case "secondary":
        return "secondary";
      case "danger":
        return "destructive";
      case "success":
      case "warning":
      default:
        return "default";
    }
  };

  const getButtonSize = (): ButtonProps["size"] => {
    switch (size) {
      case "sm":
        return "sm";
      case "md":
        return "default";
      case "lg":
        return "lg";
      default:
        return "default";
    }
  };

  const getCustomClasses = () => {
    switch (variant) {
      case "primary":
        return "bg-cyan-600/90 text-white border-cyan-500 hover:bg-cyan-600 hover:shadow-[0_0_15px_rgba(0,255,255,0.3)] focus:ring-cyan-500 disabled:bg-cyan-800/50 disabled:text-cyan-500/50";
      case "secondary":
        return "bg-purple-600/90 text-white border-purple-500 hover:bg-purple-600 hover:shadow-[0_0_15px_rgba(192,132,252,0.3)] focus:ring-purple-500 disabled:bg-purple-800/50 disabled:text-purple-500/50";
      case "success":
        return "bg-green-600/90 text-white border-green-500 hover:bg-green-600 hover:shadow-[0_0_15px_rgba(74,222,128,0.3)] focus:ring-green-500 disabled:bg-green-800/50 disabled:text-green-500/50";
      case "warning":
        return "bg-yellow-500/90 text-white border-yellow-400 hover:bg-yellow-500 hover:shadow-[0_0_15px_rgba(234,179,8,0.3)] focus:ring-yellow-500 disabled:bg-yellow-800/50 disabled:text-yellow-500/50";
      case "danger":
        return "bg-red-600/90 text-white border-red-500 hover:bg-red-600 hover:shadow-[0_0_15px_rgba(255,59,48,0.3)] focus:ring-red-500 disabled:bg-red-800/50 disabled:text-red-500/50";
      default:
        return "bg-gray-700 text-white border-gray-600 hover:bg-gray-600 focus:ring-gray-500";
    }
  };

  return (
    <Button
      variant={getButtonVariant()}
      size={getButtonSize()}
      className={cn(
        "border rounded-lg font-semibold tracking-wide transition-all duration-200 ease-out focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-gray-900 disabled:cursor-not-allowed disabled:shadow-none",
        getCustomClasses(),
        className
      )}
      disabled={isDisabled}
      {...props}
    >
      {loading ? (
        <>
          <LoadingSpinner size="sm" />
          <span className="whitespace-nowrap">{loadingText}</span>
        </>
      ) : (
        <>
          {icon && <span className="flex-shrink-0">{icon}</span>}
          <span className="whitespace-nowrap">{children}</span>
        </>
      )}
    </Button>
  );
};

export default ActionButton;