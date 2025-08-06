/* eslint.config.js - minimal flat config for ESLint v9 to remove unused imports in frontend */
import js from "@eslint/js";
import tseslint from "typescript-eslint";
import importPlugin from "eslint-plugin-import";

export default [
  {
    files: ["frontend/**/*.{ts,tsx,js,jsx}"],
    languageOptions: {
      parser: tseslint.parser,
      parserOptions: {
        ecmaVersion: "latest",
        sourceType: "module",
        project: false
      }
    },
    plugins: {
      import: importPlugin,
      "@typescript-eslint": tseslint.plugin
    },
    rules: {
      "no-unused-vars": "off",
      "@typescript-eslint/no-unused-vars": ["warn", { argsIgnorePattern: "^_", varsIgnorePattern: "^_", caughtErrorsIgnorePattern: "^_" }],
      "import/no-unused-modules": ["warn", { unusedExports: true, missingExports: false }],
      "no-undef": "off"
    },
    ignores: [
      "frontend/.next/**",
      "frontend/node_modules/**",
      "frontend/.swc/**",
      "**/dist/**",
      "**/build/**"
    ]
  },
  js.configs.recommended,
  ...tseslint.configs.recommended
];
