"use client";

import React from "react";
import { Info } from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";

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
          <Tooltip>
            <TooltipTrigger asChild>
              <Info
                size={16}
                className="text-gray-400 cursor-pointer hover:text-gray-300 transition-colors"
              />
            </TooltipTrigger>
            <TooltipContent
              className="bg-black text-white border-gray-600 max-w-xs z-[60]"
              side="top"
              sideOffset={5}
            >
              <p>{description}</p>
            </TooltipContent>
          </Tooltip>
        )}
      </div>
      {labelAddon}
    </div>
    <Input // Use the Input component
      type={type}
      value={value}
      onChange={(e) => {
        if (type === "number") {
          onChange(Number(e.target.value));
        } else {
          onChange(e.target.value);
        }
      }}
      className={cn(
        "w-full p-3 bg-gray-800 border border-secondary-700 text-white rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent",
        className
      )}
      min={min}
      max={max}
      step={step}
      required={required}
      placeholder={placeholder}
      disabled={disabled}
    />
  </div>
);
