/**
 * Tailwind CSS設定ファイル - エンタープライズデザインシステム
 *
 * エンタープライズレベルのモダンなデザインシステムを実現するための
 * Tailwind CSSカスタマイズ設定を定義します。
 *
 * @see https://tailwindcss.com/docs/configuration
 * @type {import('tailwindcss').Config}
 */
module.exports = {
  // Tailwindがスキャンするファイルパスの指定
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],

  // ダークモード設定 - 常にダークモード
  darkMode: "media",

  // エンタープライズデザインシステムのテーマ拡張
  theme: {
    extend: {
      // エンタープライズカラーパレット
      colors: {
        // プライマリーブランドカラー（深いブルー系）
        primary: {
          50: "#eff6ff",
          100: "#dbeafe",
          200: "#bfdbfe",
          300: "#93c5fd",
          400: "#60a5fa",
          500: "#3b82f6",
          600: "#2563eb",
          700: "#1d4ed8",
          800: "#1e40af",
          900: "#1e3a8a",
          950: "#172554",
        },
        // セカンダリーカラー（グレー系）
        secondary: {
          50: "#f8fafc",
          100: "#f1f5f9",
          200: "#e2e8f0",
          300: "#cbd5e1",
          400: "#94a3b8",
          500: "#64748b",
          600: "#475569",
          700: "#334155",
          800: "#1e293b",
          900: "#0f172a",
          950: "#020617",
        },
        // アクセントカラー（エメラルド系）
        accent: {
          50: "#ecfdf5",
          100: "#d1fae5",
          200: "#a7f3d0",
          300: "#6ee7b7",
          400: "#34d399",
          500: "#10b981",
          600: "#059669",
          700: "#047857",
          800: "#065f46",
          900: "#064e3b",
          950: "#022c22",
        },
        // 成功カラー
        success: {
          50: "#f0fdf4",
          100: "#dcfce7",
          500: "#22c55e",
          600: "#16a34a",
          700: "#15803d",
        },
        // 警告カラー
        warning: {
          50: "#fffbeb",
          100: "#fef3c7",
          200: "#fed7aa",
          500: "#f59e0b",
          600: "#d97706",
          700: "#b45309",
          800: "#92400e",
          900: "#78350f",
        },
        // エラーカラー
        error: {
          50: "#fef2f2",
          100: "#fee2e2",
          200: "#fecaca",
          500: "#ef4444",
          600: "#dc2626",
          700: "#b91c1c",
          800: "#991b1b",
          900: "#7f1d1d",
        },
        // 情報カラー
        info: {
          50: "#eff6ff",
          100: "#dbeafe",
          500: "#3b82f6",
          600: "#2563eb",
          700: "#1d4ed8",
        },
      },

      // エンタープライズタイポグラフィ
      fontFamily: {
        sans: [
          "Inter",
          "-apple-system",
          "BlinkMacSystemFont",
          "Segoe UI",
          "Roboto",
          "Helvetica Neue",
          "Arial",
          "sans-serif",
        ],
        mono: [
          "JetBrains Mono",
          "Fira Code",
          "Monaco",
          "Consolas",
          "Liberation Mono",
          "Courier New",
          "monospace",
        ],
      },

      // エンタープライズシャドウシステム
      boxShadow: {
        "enterprise-sm": "0 1px 2px 0 rgba(0, 0, 0, 0.05)",
        enterprise:
          "0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)",
        "enterprise-md":
          "0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)",
        "enterprise-lg":
          "0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)",
        "enterprise-xl":
          "0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)",
        "enterprise-2xl": "0 25px 50px -12px rgba(0, 0, 0, 0.25)",
        "enterprise-inner": "inset 0 2px 4px 0 rgba(0, 0, 0, 0.06)",
      },

      // エンタープライズボーダーラディウス
      borderRadius: {
        "enterprise-sm": "0.25rem",
        enterprise: "0.375rem",
        "enterprise-md": "0.5rem",
        "enterprise-lg": "0.75rem",
        "enterprise-xl": "1rem",
        "enterprise-2xl": "1.5rem",
      },

      // エンタープライズスペーシング
      spacing: {
        18: "4.5rem",
        88: "22rem",
        128: "32rem",
      },

      // エンタープライズアニメーション
      animation: {
        "fade-in": "fadeIn 0.5s ease-in-out",
        "slide-up": "slideUp 0.3s ease-out",
        "slide-down": "slideDown 0.3s ease-out",
        "scale-in": "scaleIn 0.2s ease-out",
        "pulse-slow": "pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite",
      },

      // カスタムキーフレーム
      keyframes: {
        fadeIn: {
          "0%": { opacity: "0" },
          "100%": { opacity: "1" },
        },
        slideUp: {
          "0%": { transform: "translateY(10px)", opacity: "0" },
          "100%": { transform: "translateY(0)", opacity: "1" },
        },
        slideDown: {
          "0%": { transform: "translateY(-10px)", opacity: "0" },
          "100%": { transform: "translateY(0)", opacity: "1" },
        },
        scaleIn: {
          "0%": { transform: "scale(0.95)", opacity: "0" },
          "100%": { transform: "scale(1)", opacity: "1" },
        },
      },

      // エンタープライズグラデーション
      backgroundImage: {
        "gradient-radial": "radial-gradient(var(--tw-gradient-stops))",
        "gradient-conic":
          "conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))",
        "enterprise-gradient":
          "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
        "enterprise-gradient-light":
          "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)",
        "enterprise-gradient-dark":
          "linear-gradient(135deg, #4c63d2 0%, #152331 100%)",
      },

      // エンタープライズブレークポイント
      screens: {
        xs: "475px",
        "3xl": "1600px",
      },
    },
  },

  // プラグイン（将来的な拡張用）
  plugins: [],
};
