/**
 * 時間軸選択コンポーネント テスト
 *
 * TimeFrameSelector コンポーネントのテストケースです。
 * ユーザーインタラクション、状態管理をテストします。
 *
 */

import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import "@testing-library/jest-dom";
import TimeFrameSelector from "@/components/common/TimeFrameSelector";
import { TimeFrame } from "@/types/market-data";
import { SUPPORTED_TIMEFRAMES } from "@/constants";

describe("TimeFrameSelector", () => {
  const mockOnTimeFrameChange = jest.fn();

  beforeEach(() => {
    mockOnTimeFrameChange.mockClear();
  });

  describe("レンダリングテスト", () => {
    test("すべての時間軸オプションが表示される（compactモード）", () => {
      render(
        <TimeFrameSelector
          selectedTimeFrame="1h"
          onTimeFrameChange={mockOnTimeFrameChange}
          mode="compact"
        />
      );

      expect(screen.getByText("時間軸")).toBeInTheDocument();
      expect(screen.getByText("15分")).toBeInTheDocument();
      expect(screen.getByText("30分")).toBeInTheDocument();
      expect(screen.getByText("1時間")).toBeInTheDocument();
      expect(screen.getByText("4時間")).toBeInTheDocument();
      expect(screen.getByText("1日")).toBeInTheDocument();
    });

    test("すべての時間軸ボタンが表示される（buttonsモード）", () => {
      render(
        <TimeFrameSelector
          selectedTimeFrame="1d"
          onTimeFrameChange={mockOnTimeFrameChange}
          mode="buttons"
        />
      );

      expect(screen.getByText("時間軸選択")).toBeInTheDocument();
      expect(screen.getByText("15分")).toBeInTheDocument();
      expect(screen.getByText("30分")).toBeInTheDocument();
      expect(screen.getByText("1時間")).toBeInTheDocument();
      expect(screen.getByText("4時間")).toBeInTheDocument();
      expect(screen.getByText("1日")).toBeInTheDocument();
    });

    test("選択された時間軸が正しくハイライトされる（buttonsモード）", () => {
      render(
        <TimeFrameSelector
          selectedTimeFrame="1h"
          onTimeFrameChange={mockOnTimeFrameChange}
          mode="buttons"
        />
      );

      const selectedButton = screen.getByText("1時間").closest("button");
      expect(selectedButton).toHaveClass("bg-primary-600", "text-white");
    });

    test("選択されていない時間軸が正しいスタイルを持つ（buttonsモード）", () => {
      render(
        <TimeFrameSelector
          selectedTimeFrame="1h"
          onTimeFrameChange={mockOnTimeFrameChange}
          mode="buttons"
        />
      );

      const unselectedButton = screen.getByText("15分").closest("button");
      expect(unselectedButton).toHaveClass("bg-gray-800");
      expect(unselectedButton).not.toHaveClass("bg-primary-600", "text-white");
    });

    test("compactモードでselectが正しく動作する", () => {
      render(
        <TimeFrameSelector
          selectedTimeFrame="1h"
          onTimeFrameChange={mockOnTimeFrameChange}
          mode="compact"
        />
      );

      const select = screen.getByRole("combobox");
      expect(select).toHaveValue("1h");
    });
  });

  describe("ユーザーインタラクションテスト", () => {
    test("時間軸ボタンをクリックするとコールバックが呼ばれる（buttonsモード）", () => {
      render(
        <TimeFrameSelector
          selectedTimeFrame="1d"
          onTimeFrameChange={mockOnTimeFrameChange}
          mode="buttons"
        />
      );

      const button = screen.getByText("1時間");
      fireEvent.click(button);

      expect(mockOnTimeFrameChange).toHaveBeenCalledTimes(1);
      expect(mockOnTimeFrameChange).toHaveBeenCalledWith("1h");
    });

    test("selectで時間軸を変更するとコールバックが呼ばれる（compactモード）", () => {
      render(
        <TimeFrameSelector
          selectedTimeFrame="1d"
          onTimeFrameChange={mockOnTimeFrameChange}
          mode="compact"
        />
      );

      const select = screen.getByRole("combobox");
      fireEvent.change(select, { target: { value: "1h" } });

      expect(mockOnTimeFrameChange).toHaveBeenCalledTimes(1);
      expect(mockOnTimeFrameChange).toHaveBeenCalledWith("1h");
    });

    test("すべての時間軸ボタンが正しい値でコールバックを呼ぶ（buttonsモード）", () => {
      const timeFrames = SUPPORTED_TIMEFRAMES.map((tf) => ({
        label: tf.label,
        value: tf.value,
      }));

      render(
        <TimeFrameSelector
          selectedTimeFrame="1d"
          onTimeFrameChange={mockOnTimeFrameChange}
          mode="buttons"
        />
      );

      timeFrames.forEach(({ label, value }) => {
        const button = screen.getByText(label);
        fireEvent.click(button);
        expect(mockOnTimeFrameChange).toHaveBeenCalledWith(value);
      });

      expect(mockOnTimeFrameChange).toHaveBeenCalledTimes(timeFrames.length);
    });

    test("選択済みの時間軸をクリックしてもコールバックが呼ばれる（buttonsモード）", () => {
      render(
        <TimeFrameSelector
          selectedTimeFrame="1h"
          onTimeFrameChange={mockOnTimeFrameChange}
          mode="buttons"
        />
      );

      const selectedButton = screen.getByText("1時間");
      fireEvent.click(selectedButton);

      expect(mockOnTimeFrameChange).toHaveBeenCalledTimes(1);
      expect(mockOnTimeFrameChange).toHaveBeenCalledWith("1h");
    });
  });

  describe("無効化状態テスト", () => {
    test("disabled=trueの場合、すべてのボタンが無効化される（buttonsモード）", () => {
      render(
        <TimeFrameSelector
          selectedTimeFrame="1d"
          onTimeFrameChange={mockOnTimeFrameChange}
          disabled={true}
          mode="buttons"
        />
      );

      const buttons = screen.getAllByRole("button");
      buttons.forEach((button) => {
        expect(button).toBeDisabled();
        expect(button).toHaveClass("opacity-50", "cursor-not-allowed");
      });
    });

    test("disabled=trueの場合、selectが無効化される（compactモード）", () => {
      render(
        <TimeFrameSelector
          selectedTimeFrame="1d"
          onTimeFrameChange={mockOnTimeFrameChange}
          disabled={true}
          mode="compact"
        />
      );

      const select = screen.getByRole("combobox");
      expect(select).toBeDisabled();
    });

    test("disabled=trueの場合、ボタンクリックでコールバックが呼ばれない（buttonsモード）", () => {
      render(
        <TimeFrameSelector
          selectedTimeFrame="1d"
          onTimeFrameChange={mockOnTimeFrameChange}
          disabled={true}
          mode="buttons"
        />
      );

      const button = screen.getByText("1時間");
      fireEvent.click(button);

      expect(mockOnTimeFrameChange).not.toHaveBeenCalled();
    });

    test("disabled=falseの場合、ボタンが有効化される（buttonsモード）", () => {
      render(
        <TimeFrameSelector
          selectedTimeFrame="1d"
          onTimeFrameChange={mockOnTimeFrameChange}
          disabled={false}
          mode="buttons"
        />
      );

      const buttons = screen.getAllByRole("button");
      buttons.forEach((button) => {
        expect(button).not.toBeDisabled();
        expect(button).not.toHaveClass("opacity-50", "cursor-not-allowed");
      });
    });
  });

  describe("アクセシビリティテスト", () => {
    test("各ボタンに適切なtitle属性が設定される（buttonsモード）", () => {
      render(
        <TimeFrameSelector
          selectedTimeFrame="1d"
          onTimeFrameChange={mockOnTimeFrameChange}
          mode="buttons"
        />
      );

      expect(screen.getByText("15分").closest("button")).toHaveAttribute(
        "title",
        "15分足データ"
      );
      expect(screen.getByText("30分").closest("button")).toHaveAttribute(
        "title",
        "30分足データ"
      );
      expect(screen.getByText("1時間").closest("button")).toHaveAttribute(
        "title",
        "1時間足データ"
      );
      expect(screen.getByText("4時間").closest("button")).toHaveAttribute(
        "title",
        "4時間足データ"
      );
      expect(screen.getByText("1日").closest("button")).toHaveAttribute(
        "title",
        "日足データ"
      );
    });

    test("フォーカス時に適切なスタイルが適用される（buttonsモード）", () => {
      render(
        <TimeFrameSelector
          selectedTimeFrame="1d"
          onTimeFrameChange={mockOnTimeFrameChange}
          mode="buttons"
        />
      );

      const button = screen.getByText("1時間").closest("button");
      expect(button).toHaveClass(
        "focus:outline-none",
        "focus:ring-2",
        "focus:ring-primary-500"
      );
    });

    test("ボタンがキーボードでアクセス可能である（buttonsモード）", () => {
      render(
        <TimeFrameSelector
          selectedTimeFrame="1d"
          onTimeFrameChange={mockOnTimeFrameChange}
          mode="buttons"
        />
      );

      const button = screen.getByText("1時間").closest("button");
      button!.focus();
      expect(button).toHaveFocus();

      // Enterキーではなく、クリックイベントをテスト
      fireEvent.click(button!);
      expect(mockOnTimeFrameChange).toHaveBeenCalledWith("1h");
    });
  });

  describe("レスポンシブデザインテスト", () => {
    test("flex-wrapクラスが適用されている（buttonsモード）", () => {
      const { container } = render(
        <TimeFrameSelector
          selectedTimeFrame="1d"
          onTimeFrameChange={mockOnTimeFrameChange}
          mode="buttons"
        />
      );

      const wrapper = container.querySelector(".flex.flex-wrap.gap-2");
      expect(wrapper).toBeInTheDocument();
    });
  });

  describe("プロパティテスト", () => {
    test("異なる選択状態で正しくレンダリングされる（buttonsモード）", () => {
      const timeFrames: TimeFrame[] = ["15m", "30m", "1h", "4h", "1d"];

      timeFrames.forEach((selectedTimeFrame) => {
        const { rerender } = render(
          <TimeFrameSelector
            selectedTimeFrame={selectedTimeFrame}
            onTimeFrameChange={mockOnTimeFrameChange}
            mode="buttons"
          />
        );

        // 選択された時間軸のボタンがハイライトされていることを確認
        const buttons = screen.getAllByRole("button");
        const selectedButton = buttons.find((button) =>
          button.classList.contains("bg-primary-600")
        );
        expect(selectedButton).toBeInTheDocument();

        rerender(<div />); // クリーンアップ
      });
    });

    test("compactモードで正しい値が選択される", () => {
      const timeFrames: TimeFrame[] = ["15m", "30m", "1h", "4h", "1d"];

      timeFrames.forEach((selectedTimeFrame) => {
        const { rerender } = render(
          <TimeFrameSelector
            selectedTimeFrame={selectedTimeFrame}
            onTimeFrameChange={mockOnTimeFrameChange}
            mode="compact"
          />
        );

        const select = screen.getByRole("combobox");
        expect(select).toHaveValue(selectedTimeFrame);

        rerender(<div />); // クリーンアップ
      });
    });
  });
});
