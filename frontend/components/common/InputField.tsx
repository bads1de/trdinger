"use client";

import React from "react";

interface InputFieldProps {
  label: string;
  value: any;
  onChange: (value: any) => void;
  type?: string;
  min?: number;
  max?: number;
  step?: number;
  required?: boolean;
  placeholder?: string;
  className?: string;
}

export const InputField: React.FC<InputFieldProps> = ({
  label,
  value,
  onChange,
  type = "text",
  min,
  max,
  step,
  required = false,
  placeholder = "",
  className = "",
}) => (
  <div>
    <label className="block text-sm font-medium text-gray-300 mb-2">
      {label}
    </label>
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
    />
  </div>
);
