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
    test("基本的な要素が表示される", () => {
      render(
        <SymbolSelector
          selectedSymbol="BTC/USDT"
          onSymbolChange={mockOnSymbolChange}
        />
      );

      expect(screen.getByText("通貨ペア選択")).toBeInTheDocument();
      expect(screen.getByText("BTC/USDT")).toBeInTheDocument();
    });

    test("カテゴリ表示が有効な場合、スポットと先物のセクションが表示される", async () => {
      render(
        <SymbolSelector
          selectedSymbol="BTC/USDT"
          onSymbolChange={mockOnSymbolChange}
          showCategories={true}
        />
      );

      // ドロップダウンを開く
      const button = screen.getByRole("button");
      fireEvent.click(button);

      await waitFor(() => {
        expect(screen.getByText("💰 スポット取引")).toBeInTheDocument();
        expect(screen.getByText("⚡ 永続契約")).toBeInTheDocument();
      });
    });

    test("シンプル表示の場合、従来のセレクトボックスが表示される", () => {
      render(
        <SymbolSelector
          selectedSymbol="BTC/USDT"
          onSymbolChange={mockOnSymbolChange}
          showCategories={false}
        />
      );

      expect(screen.getByRole("combobox")).toBeInTheDocument();
    });

    test("選択されたペアの情報が正しく表示される", () => {
      render(
        <SymbolSelector
          selectedSymbol="BTC/USDT"
          onSymbolChange={mockOnSymbolChange}
          showCategories={true}
        />
      );

      expect(screen.getByText("BTC/USDT")).toBeInTheDocument();
      expect(screen.getByText("₿")).toBeInTheDocument(); // Bitcoin icon
    });
  });

  describe("ユーザーインタラクションテスト", () => {
    test("カテゴリ表示でペアを選択するとコールバックが呼ばれる", async () => {
      render(
        <SymbolSelector
          selectedSymbol="BTC/USDT"
          onSymbolChange={mockOnSymbolChange}
          showCategories={true}
        />
      );

      // ドロップダウンを開く
      const button = screen.getByRole("button");
      fireEvent.click(button);

      // ETH/USDTを選択
      await waitFor(() => {
        const ethOption = screen.getByText("ETH/USDT");
        fireEvent.click(ethOption);
      });

      expect(mockOnSymbolChange).toHaveBeenCalledWith("ETH/USDT");
    });

    test("シンプル表示でペアを選択するとコールバックが呼ばれる", () => {
      render(
        <SymbolSelector
          selectedSymbol="BTC/USDT"
          onSymbolChange={mockOnSymbolChange}
          showCategories={false}
        />
      );

      const select = screen.getByRole("combobox");
      fireEvent.change(select, { target: { value: "ETH/USDT" } });

      expect(mockOnSymbolChange).toHaveBeenCalledWith("ETH/USDT");
    });

    test("ドロップダウンの開閉が正しく動作する", async () => {
      render(
        <SymbolSelector
          selectedSymbol="BTC/USDT"
          onSymbolChange={mockOnSymbolChange}
          showCategories={true}
        />
      );

      const button = screen.getByRole("button");

      // 開く
      fireEvent.click(button);
      await waitFor(() => {
        expect(screen.getByText("💰 スポット取引")).toBeInTheDocument();
      });

      // 閉じる
      fireEvent.click(button);
      await waitFor(() => {
        expect(screen.queryByText("💰 スポット取引")).not.toBeInTheDocument();
      });
    });
  });

  describe("ローディング状態テスト", () => {
    test("loading=trueの場合、ローディングスピナーが表示される", () => {
      render(
        <SymbolSelector
          selectedSymbol="BTC/USDT"
          onSymbolChange={mockOnSymbolChange}
          loading={true}
        />
      );

      expect(screen.getByRole("button")).toBeDisabled();
    });

    test("loading=trueの場合、ドロップダウンが開かない", () => {
      render(
        <SymbolSelector
          selectedSymbol="BTC/USDT"
          onSymbolChange={mockOnSymbolChange}
          loading={true}
          showCategories={true}
        />
      );

      const button = screen.getByRole("button");
      fireEvent.click(button);

      expect(screen.queryByText("💰 スポット取引")).not.toBeInTheDocument();
    });
  });

  describe("無効化状態テスト", () => {
    test("disabled=trueの場合、ボタンが無効化される", () => {
      render(
        <SymbolSelector
          selectedSymbol="BTC/USDT"
          onSymbolChange={mockOnSymbolChange}
          disabled={true}
        />
      );

      const button = screen.getByRole("button");
      expect(button).toBeDisabled();
    });

    test("disabled=trueの場合、セレクトボックスが無効化される", () => {
      render(
        <SymbolSelector
          selectedSymbol="BTC/USDT"
          onSymbolChange={mockOnSymbolChange}
          disabled={true}
          showCategories={false}
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
          showCategories={false}
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
          showCategories={false}
        />
      );

      expect(screen.getByText("⚠️ 利用可能な通貨ペアがありません")).toBeInTheDocument();
    });
  });

  describe("アイコン表示テスト", () => {
    test("各通貨の正しいアイコンが表示される", () => {
      render(
        <SymbolSelector
          selectedSymbol="BTC/USDT"
          onSymbolChange={mockOnSymbolChange}
          showCategories={true}
        />
      );

      expect(screen.getByText("₿")).toBeInTheDocument(); // Bitcoin
    });
  });

  describe("市場タイプ表示テスト", () => {
    test("スポットペアに正しいバッジが表示される", async () => {
      render(
        <SymbolSelector
          selectedSymbol="BTC/USDT"
          onSymbolChange={mockOnSymbolChange}
          showCategories={true}
        />
      );

      const button = screen.getByRole("button");
      fireEvent.click(button);

      await waitFor(() => {
        const spotBadges = screen.getAllByText("スポット");
        expect(spotBadges.length).toBeGreaterThan(0);
      });
    });

    test("永続契約ペアに正しいバッジが表示される", async () => {
      render(
        <SymbolSelector
          selectedSymbol="BTC/USDT"
          onSymbolChange={mockOnSymbolChange}
          showCategories={true}
        />
      );

      const button = screen.getByRole("button");
      fireEvent.click(button);

      await waitFor(() => {
        const perpetualBadges = screen.getAllByText(/永続契約/);
        expect(perpetualBadges.length).toBeGreaterThan(0);
      });
    });
  });

  describe("アクセシビリティテスト", () => {
    test("適切なaria属性が設定されている", () => {
      render(
        <SymbolSelector
          selectedSymbol="BTC/USDT"
          onSymbolChange={mockOnSymbolChange}
        />
      );

      const button = screen.getByRole("button");
      expect(button).toBeInTheDocument();
    });

    test("キーボードナビゲーションが可能である", () => {
      render(
        <SymbolSelector
          selectedSymbol="BTC/USDT"
          onSymbolChange={mockOnSymbolChange}
          showCategories={false}
        />
      );

      const select = screen.getByRole("combobox");
      select.focus();
      expect(select).toHaveFocus();
    });
  });

  describe("統計情報表示テスト", () => {
    test("利用可能なペア数が正しく表示される", () => {
      render(
        <SymbolSelector
          selectedSymbol="BTC/USDT"
          onSymbolChange={mockOnSymbolChange}
          showCategories={true}
        />
      );

      expect(screen.getByText(`${SUPPORTED_TRADING_PAIRS.length}ペア利用可能`)).toBeInTheDocument();
    });
  });
});
