"use client";

import React from "react";

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
}) => (
  <div>
    <div className="flex items-center justify-between mb-2">
      <label className="block text-sm font-medium text-gray-300">
        {label}
      </label>
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
