/**
 * LabelGenerationSettingsコンポーネントのテスト
 *
 * ラベル生成設定UIコンポーネントの機能をテストします。
 * - プリセット使用/カスタム設定の切り替え
 * - プリセット選択
 * - カスタム設定の入力
 * - 閾値計算方法の選択
 */

import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import "@testing-library/jest-dom";
import { LabelGenerationSettings } from "@/components/ml/LabelGenerationSettings";
import type { LabelGenerationConfig } from "@/types/ml-config";

// TooltipProviderをモック
jest.mock("@/components/ui/tooltip", () => ({
  Tooltip: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  TooltipContent: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  TooltipTrigger: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  TooltipProvider: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
}));

describe("LabelGenerationSettings", () => {
  const mockConfig: LabelGenerationConfig = {
    usePreset: true,
    defaultPreset: "4h_4bars",
    timeframe: "4h",
    horizonN: 4,
    threshold: 0.002,
    priceColumn: "close",
    thresholdMethod: "FIXED",
  };

  const mockOnChange = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe("基本レンダリング", () => {
    it("コンポーネントが正しく表示される", () => {
      render(
        <LabelGenerationSettings config={mockConfig} onChange={mockOnChange} />
      );

      expect(screen.getByText("ラベル生成設定")).toBeInTheDocument();
    });

    it("プリセット使用チェックボックスが正しく表示される", () => {
      render(
        <LabelGenerationSettings config={mockConfig} onChange={mockOnChange} />
      );

      const checkbox = screen.getByRole("checkbox");
      expect(checkbox).toBeInTheDocument();
      expect(checkbox).toBeChecked();

      // ラベルテキストを確認
      expect(screen.getByText("プリセットを使用")).toBeInTheDocument();
    });
  });

  describe("プリセット使用モード", () => {
    it("プリセット選択が表示される（usePreset=trueの場合）", () => {
      render(
        <LabelGenerationSettings config={mockConfig} onChange={mockOnChange} />
      );

      // プリセットラベルが表示されることを確認
      expect(screen.getByText("プリセット")).toBeInTheDocument();
    });

    it("カスタム設定が非表示になる（usePreset=trueの場合）", () => {
      render(
        <LabelGenerationSettings config={mockConfig} onChange={mockOnChange} />
      );

      // カスタム設定のフィールドが表示されないことを確認
      expect(screen.queryByText("時間足")).not.toBeInTheDocument();
      expect(screen.queryByText("ホライズン（N本先）")).not.toBeInTheDocument();
    });

    it("選択されたプリセットの詳細が表示される", () => {
      render(
        <LabelGenerationSettings config={mockConfig} onChange={mockOnChange} />
      );

      // プリセット詳細情報の確認（Alertコンテナ内のテキストを確認）
      expect(screen.getByText(/時間足:/)).toBeInTheDocument();
      expect(screen.getAllByText(/4時間足/).length).toBeGreaterThan(0);
      expect(screen.getByText(/ホライズン:/)).toBeInTheDocument();
      expect(screen.getAllByText(/4本先/).length).toBeGreaterThan(0);
      expect(screen.getByText(/閾値:/)).toBeInTheDocument();
      expect(screen.getAllByText(/0.2%/).length).toBeGreaterThan(0);
      expect(screen.getByText(/閾値計算方法:/)).toBeInTheDocument();
      // "固定閾値"は複数箇所に表示されるため、getAllByTextで確認
      expect(screen.getAllByText(/固定閾値/).length).toBeGreaterThan(0);
    });
  });

  describe("カスタム設定モード", () => {
    const customConfig: LabelGenerationConfig = {
      ...mockConfig,
      usePreset: false,
    };

    it("カスタム設定が表示される（usePreset=falseの場合）", () => {
      render(
        <LabelGenerationSettings
          config={customConfig}
          onChange={mockOnChange}
        />
      );

      // カスタム設定のフィールドが表示されることを確認
      expect(screen.getByText("時間足")).toBeInTheDocument();
      expect(screen.getByText("ホライズン（N本先）")).toBeInTheDocument();
      expect(screen.getByText("閾値")).toBeInTheDocument();
      expect(screen.getByText("閾値計算方法")).toBeInTheDocument();
      expect(screen.getByText("価格カラム")).toBeInTheDocument();
    });

    it("カスタム設定の数値入力が動作する（horizonN）", () => {
      render(
        <LabelGenerationSettings
          config={customConfig}
          onChange={mockOnChange}
        />
      );

      const horizonInput = screen
        .getAllByRole("spinbutton")
        .find((input) => {
          const label = input.closest("div")?.querySelector("label");
          return label?.textContent?.includes("ホライズン");
        });

      expect(horizonInput).toBeDefined();
      if (horizonInput) {
        fireEvent.change(horizonInput, { target: { value: "8" } });
        expect(mockOnChange).toHaveBeenCalledWith("horizonN", 8);
      }
    });

    it("カスタム設定の数値入力が動作する（threshold）", () => {
      render(
        <LabelGenerationSettings
          config={customConfig}
          onChange={mockOnChange}
        />
      );

      const thresholdInput = screen
        .getAllByRole("spinbutton")
        .find((input) => {
          const label = input.closest("div")?.querySelector("label");
          return label?.textContent?.includes("閾値") && !label?.textContent?.includes("計算方法");
        });

      expect(thresholdInput).toBeDefined();
      if (thresholdInput) {
        fireEvent.change(thresholdInput, { target: { value: "0.003" } });
        expect(mockOnChange).toHaveBeenCalledWith("threshold", 0.003);
      }
    });

    it("価格カラムのテキスト入力が動作する", () => {
      render(
        <LabelGenerationSettings
          config={customConfig}
          onChange={mockOnChange}
        />
      );

      const priceColumnInput = screen
        .getAllByRole("textbox")
        .find((input) => {
          const label = input.closest("div")?.querySelector("label");
          return label?.textContent?.includes("価格カラム");
        });

      expect(priceColumnInput).toBeDefined();
      if (priceColumnInput) {
        fireEvent.change(priceColumnInput, { target: { value: "high" } });
        expect(mockOnChange).toHaveBeenCalledWith("priceColumn", "high");
      }
    });

    it("閾値計算方法の説明が表示される", () => {
      render(
        <LabelGenerationSettings
          config={customConfig}
          onChange={mockOnChange}
        />
      );

      // 閾値計算方法の説明テキストを確認
      expect(
        screen.getByText(/固定された閾値を使用します/)
      ).toBeInTheDocument();
    });
  });

  describe("インタラクション", () => {
    it("プリセット使用チェックボックスの切り替えが動作する", async () => {
      render(
        <LabelGenerationSettings config={mockConfig} onChange={mockOnChange} />
      );

      const checkbox = screen.getByRole("checkbox");
      fireEvent.click(checkbox);

      await waitFor(() => {
        expect(mockOnChange).toHaveBeenCalledWith("usePreset", false);
      });
    });

    it("プリセット使用をオフにするとカスタム設定が表示される", () => {
      const { rerender } = render(
        <LabelGenerationSettings config={mockConfig} onChange={mockOnChange} />
      );

      // 初期状態ではカスタム設定が非表示
      expect(screen.queryByText("時間足")).not.toBeInTheDocument();

      // usePresetをfalseに変更
      const updatedConfig = { ...mockConfig, usePreset: false };
      rerender(
        <LabelGenerationSettings
          config={updatedConfig}
          onChange={mockOnChange}
        />
      );

      // カスタム設定が表示される
      expect(screen.getByText("時間足")).toBeInTheDocument();
    });
  });

  describe("説明文", () => {
    it("コンポーネントの説明文が表示される", () => {
      render(
        <LabelGenerationSettings config={mockConfig} onChange={mockOnChange} />
      );

      expect(
        screen.getByText(
          /ラベル生成設定は、機械学習モデルが予測する目的変数/
        )
      ).toBeInTheDocument();
    });
  });

  describe("バリデーション", () => {
    it("horizonNの数値範囲が正しく設定されている", () => {
      const customConfig = { ...mockConfig, usePreset: false };
      render(
        <LabelGenerationSettings
          config={customConfig}
          onChange={mockOnChange}
        />
      );

      const horizonInput = screen
        .getAllByRole("spinbutton")
        .find((input) => {
          const label = input.closest("div")?.querySelector("label");
          return label?.textContent?.includes("ホライズン");
        }) as HTMLInputElement;

      expect(horizonInput).toBeDefined();
      if (horizonInput) {
        expect(horizonInput.min).toBe("1");
        expect(horizonInput.max).toBe("100");
      }
    });

    it("thresholdの数値範囲が正しく設定されている", () => {
      const customConfig = { ...mockConfig, usePreset: false };
      render(
        <LabelGenerationSettings
          config={customConfig}
          onChange={mockOnChange}
        />
      );

      const thresholdInput = screen
        .getAllByRole("spinbutton")
        .find((input) => {
          const label = input.closest("div")?.querySelector("label");
          return label?.textContent?.includes("閾値") && !label?.textContent?.includes("計算方法");
        }) as HTMLInputElement;

      expect(thresholdInput).toBeDefined();
      if (thresholdInput) {
        expect(thresholdInput.min).toBe("0");
        expect(thresholdInput.max).toBe("1");
        expect(thresholdInput.step).toBe("0.001");
      }
    });
  });

  describe("エッジケース", () => {
    it("プリセットが存在しない場合でもエラーにならない", () => {
      const configWithInvalidPreset: LabelGenerationConfig = {
        ...mockConfig,
        defaultPreset: "invalid_preset",
      };

      expect(() => {
        render(
          <LabelGenerationSettings
            config={configWithInvalidPreset}
            onChange={mockOnChange}
          />
        );
      }).not.toThrow();
    });

    it("空のコンフィグでもレンダリングできる", () => {
      const emptyConfig: LabelGenerationConfig = {
        usePreset: false,
        defaultPreset: "",
        timeframe: "4h",
        horizonN: 0,
        threshold: 0,
        priceColumn: "",
        thresholdMethod: "FIXED",
      };

      expect(() => {
        render(
          <LabelGenerationSettings
            config={emptyConfig}
            onChange={mockOnChange}
          />
        );
      }).not.toThrow();
    });
  });
});