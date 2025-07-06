import React from "react";

export interface MetricCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  color?: "green" | "red" | "blue" | "gray" | "yellow";
  icon?: React.ReactNode;
}

export default function MetricCard({
  title,
  value,
  subtitle,
  color = "gray",
  icon,
}: MetricCardProps) {
  const colorClasses = {
    green: "bg-green-900/20 border-green-500/30 text-green-400",
    red: "bg-red-900/20 border-red-500/30 text-red-400",
    blue: "bg-blue-900/20 border-blue-500/30 text-blue-400",
    yellow: "bg-yellow-900/20 border-yellow-500/30 text-yellow-400",
    gray: "bg-secondary-900/50 border-secondary-600 text-gray-300",
  };

  return (
    <div className={`p-4 rounded-lg border ${colorClasses[color]}`}>
      <div className="flex items-center justify-between">
        <div className="flex-1">
          <p className="text-sm font-medium text-gray-400 mb-1">{title}</p>
          <p className="text-2xl font-bold">{value}</p>
          {subtitle && <p className="text-xs text-gray-500 mt-1">{subtitle}</p>}
        </div>
        {icon && <div className="ml-3 opacity-60">{icon}</div>}
      </div>
    </div>
  );
}
