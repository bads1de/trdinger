/**
 * 通貨ペア選択コンポーネント テスト
 *
 * SymbolSelector コンポーネントのテストケースです。
 * ユーザーインタラクション、状態管理、カテゴリ表示をテストします。
 *
 * @author Trdinger Development Team
 * @version 1.0.0
 */

import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import "@testing-library/jest-dom";
import SymbolSelector from "@/components/common/SymbolSelector";
import { SUPPORTED_TRADING_PAIRS } from "@/constants";

describe("SymbolSelector", () => {
  const mockOnSymbolChange = jest.fn();

  beforeEach(() => {
    mockOnSymbolChange.mockClear();
  });

  describe("レンダリングテスト", () => {
    test("基本的な要素が表示される（compactモード）", () => {
      const testSymbols = [
        { symbol: "BTC/USDT", name: "Bitcoin / USDT", base: "BTC", quote: "USDT" },
        { symbol: "ETH/USDT", name: "Ethereum / USDT", base: "ETH", quote: "USDT" }
      ];

      render(
        <SymbolSelector
          symbols={testSymbols}
          selectedSymbol="BTC/USDT"
          onSymbolChange={mockOnSymbolChange}
          mode="compact"
        />
      );

      expect(screen.getByText("通貨ペア")).toBeInTheDocument();
      expect(screen.getByRole("combobox")).toBeInTheDocument();
    });

    test("シンプル表示の場合、従来のセレクトボックスが表示される", () => {
      const testSymbols = [
        { symbol: "BTC/USDT", name: "Bitcoin / USDT", base: "BTC", quote: "USDT" },
        { symbol: "ETH/USDT", name: "Ethereum / USDT", base: "ETH", quote: "USDT" }
      ];

      render(
        <SymbolSelector
          symbols={testSymbols}
          selectedSymbol="BTC/USDT"
          onSymbolChange={mockOnSymbolChange}
          mode="compact"
        />
      );

      expect(screen.getByRole("combobox")).toBeInTheDocument();
    });

    test("選択されたペアの情報が正しく表示される", () => {
      const testSymbols = [
        { symbol: "BTC/USDT", name: "Bitcoin / USDT", base: "BTC", quote: "USDT" },
        { symbol: "ETH/USDT", name: "Ethereum / USDT", base: "ETH", quote: "USDT" }
      ];

      render(
        <SymbolSelector
          symbols={testSymbols}
          selectedSymbol="BTC/USDT"
          onSymbolChange={mockOnSymbolChange}
          mode="compact"
        />
      );

      const select = screen.getByRole("combobox");
      expect(select).toHaveValue("BTC/USDT");
    });
  });

  describe("ユーザーインタラクションテスト", () => {
    test("シンプル表示でペアを選択するとコールバックが呼ばれる", () => {
      const testSymbols = [
        { symbol: "BTC/USDT", name: "Bitcoin / USDT", base: "BTC", quote: "USDT" },
        { symbol: "ETH/USDT", name: "Ethereum / USDT", base: "ETH", quote: "USDT" }
      ];

      render(
        <SymbolSelector
          symbols={testSymbols}
          selectedSymbol="BTC/USDT"
          onSymbolChange={mockOnSymbolChange}
          mode="compact"
        />
      );

      const select = screen.getByRole("combobox");
      fireEvent.change(select, { target: { value: "ETH/USDT" } });

      expect(mockOnSymbolChange).toHaveBeenCalledWith("ETH/USDT");
    });
  });

  describe("ローディング状態テスト", () => {
    test("loading=trueの場合、ローディングメッセージが表示される", () => {
      render(
        <SymbolSelector
          selectedSymbol=""
          onSymbolChange={mockOnSymbolChange}
          loading={true}
          mode="compact"
        />
      );

      expect(screen.getByText("読み込み中...")).toBeInTheDocument();
      expect(screen.getByRole("combobox")).toBeDisabled();
    });
  });

  describe("無効化状態テスト", () => {
    test("disabled=trueの場合、セレクトボックスが無効化される", () => {
      const testSymbols = [
        { symbol: "BTC/USDT", name: "Bitcoin / USDT", base: "BTC", quote: "USDT" }
      ];

      render(
        <SymbolSelector
          symbols={testSymbols}
          selectedSymbol="BTC/USDT"
          onSymbolChange={mockOnSymbolChange}
          disabled={true}
          mode="compact"
        />
      );

      const select = screen.getByRole("combobox");
      expect(select).toBeDisabled();
    });
  });

  describe("カスタムシンボルリストテスト", () => {
    test("カスタムシンボルリストが正しく表示される", () => {
      const customSymbols = [
        {
          symbol: "CUSTOM/USDT",
          name: "Custom Token / USDT",
          base: "CUSTOM",
          quote: "USDT"
        }
      ];

      render(
        <SymbolSelector
          symbols={customSymbols}
          selectedSymbol="CUSTOM/USDT"
          onSymbolChange={mockOnSymbolChange}
          mode="compact"
        />
      );

      const option = screen.getByRole('option', { name: /CUSTOM\/USDT/ });
      expect(option).toBeInTheDocument();
    });

    test("空のシンボルリストの場合、適切なメッセージが表示される", () => {
      render(
        <SymbolSelector
          symbols={[]}
          selectedSymbol=""
          onSymbolChange={mockOnSymbolChange}
          mode="compact"
        />
      );

      expect(screen.getByText("利用可能な通貨ペアがありません")).toBeInTheDocument();
    });
  });

  describe("アクセシビリティテスト", () => {
    test("キーボードナビゲーションが可能である", () => {
      const testSymbols = [
        { symbol: "BTC/USDT", name: "Bitcoin / USDT", base: "BTC", quote: "USDT" }
      ];

      render(
        <SymbolSelector
          symbols={testSymbols}
          selectedSymbol="BTC/USDT"
          onSymbolChange={mockOnSymbolChange}
          mode="compact"
        />
      );

      const select = screen.getByRole("combobox");
      select.focus();
      expect(select).toHaveFocus();
    });
  });
});
