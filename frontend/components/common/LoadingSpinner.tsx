import React from "react";

interface LoadingSpinnerProps {
  text?: string;
  size?: "sm" | "md" | "lg";
}

const sizeClasses = {
  sm: "h-4 w-4",
  md: "h-6 w-6",
  lg: "h-8 w-8",
};

const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({
  text = "読み込み中...",
  size = "md",
}) => {
  return (
    <div className="flex items-center justify-center p-8">
      <div
        className={`animate-spin rounded-full border-b-2 border-primary ${sizeClasses[size]}`}
      ></div>
      <span className="ml-2 text-muted-foreground">{text}</span>
    </div>
  );
};

export default LoadingSpinner;
