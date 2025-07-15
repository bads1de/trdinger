"use client";

import React from "react";
import { Info } from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface InputFieldProps {
  label: string;
  labelAddon?: React.ReactNode;
  value: any;
  onChange: (value: any) => void;
  type?: string;
  min?: number;
  max?: number;
  step?: number;
  required?: boolean;
  placeholder?: string;
  className?: string;
  disabled?: boolean;
  description?: string;
}

export const InputField: React.FC<InputFieldProps> = ({
  label,
  labelAddon,
  value,
  onChange,
  type = "text",
  min,
  max,
  step,
  required = false,
  placeholder = "",
  className = "",
  disabled = false,
  description,
}) => (
  <div>
    <div className="flex items-center justify-between mb-2">
      <div className="flex items-center gap-2">
        <label className="block text-sm font-medium text-gray-300">
          {label}
        </label>
        {description && (
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Info size={16} className="text-gray-400 cursor-pointer" />
              </TooltipTrigger>
              <TooltipContent className="bg-black text-white border-gray-600">
                <p className="max-w-xs">{description}</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        )}
      </div>
      {labelAddon}
    </div>
    <input
      type={type}
      value={value}
      onChange={(e) => {
        if (type === "number") {
          onChange(Number(e.target.value));
        } else {
          onChange(e.target.value);
        }
      }}
      className={`w-full p-3 bg-gray-800 border border-secondary-700 text-white rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent ${className}`}
      min={min}
      max={max}
      step={step}
      required={required}
      placeholder={placeholder}
      disabled={disabled}
    />
  </div>
);
