"use client";

import React from "react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { cn } from "@/lib/utils";

interface SelectFieldProps {
  label: string;
  labelAddon?: React.ReactNode;
  value: string;
  onChange: (value: string) => void;
  options: { value: string; label: string }[];
  required?: boolean;
  className?: string;
  disabled?: boolean;
}

export const SelectField: React.FC<SelectFieldProps> = ({
  label,
  labelAddon,
  value,
  onChange,
  options,
  required = false,
  className = "",
  disabled = false,
}) => (
  <div>
    <div className="flex items-center justify-between mb-2">
      <label className="block text-sm font-medium text-gray-300">
        {label}
      </label>
      {labelAddon}
    </div>
    <Select onValueChange={onChange} value={value} disabled={disabled}>
      <SelectTrigger
        className={cn(
          "w-full p-3 bg-gray-800 border border-secondary-700 text-white rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent",
          className
        )}
      >
        <SelectValue placeholder={label} />
      </SelectTrigger>
      <SelectContent>
        {options.map((option) => (
          <SelectItem key={option.value} value={option.value}>
            {option.label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  </div>
);
