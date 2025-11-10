/**
 * FeatureProfileSettingsコンポーネントのテスト
 *
 * 特徴量プロファイル設定UIコンポーネントの機能をテストします。
 * - プロファイル選択（research/production）
 * - カスタムallowlistの入力とバリデーション
 * - JSON形式のバリデーション
 * - エラーハンドリング
 */

import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import "@testing-library/jest-dom";
import { FeatureProfileSettings } from "@/components/ml/FeatureProfileSettings";
import type { FeatureEngineeringConfig } from "@/types/ml-config";

describe("FeatureProfileSettings", () => {
  const mockConfig: FeatureEngineeringConfig = {
    profile: "production",
    customAllowlist: null,
  };

  const mockOnChange = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe("基本レンダリング", () => {
    it("コンポーネントが正しく表示される", () => {
      render(
        <FeatureProfileSettings config={mockConfig} onChange={mockOnChange} />
      );

      expect(screen.getByText("特徴量エンジニアリング設定")).toBeInTheDocument();
    });

    it("プロファイル選択ラジオボタンが表示される", () => {
      render(
        <FeatureProfileSettings config={mockConfig} onChange={mockOnChange} />
      );

      // Research と Production のラベルを確認（複数箇所に表示されるため）
      expect(screen.getAllByText(/研究用/).length).toBeGreaterThan(0);
      expect(screen.getAllByText(/本番用/).length).toBeGreaterThan(0);
    });

    it("カスタムallowlist入力欄が表示される", () => {
      render(
        <FeatureProfileSettings config={mockConfig} onChange={mockOnChange} />
      );

      expect(
        screen.getByText(/カスタム特徴量allowlist/)
      ).toBeInTheDocument();
      expect(screen.getByRole("textbox")).toBeInTheDocument();
    });
  });

  describe("プロファイル選択", () => {
    it("現在のプロファイルが選択されている（production）", () => {
      render(
        <FeatureProfileSettings config={mockConfig} onChange={mockOnChange} />
      );

      const productionRadio = screen.getByRole("radio", {
        name: /本番用/,
      });
      expect(productionRadio).toBeChecked();
    });

    it("現在のプロファイルが選択されている（research）", () => {
      const researchConfig: FeatureEngineeringConfig = {
        profile: "research",
        customAllowlist: null,
      };

      render(
        <FeatureProfileSettings
          config={researchConfig}
          onChange={mockOnChange}
        />
      );

      const researchRadio = screen.getByRole("radio", {
        name: /研究用/,
      });
      expect(researchRadio).toBeChecked();
    });

    it("プロファイル変更が動作する（research → production）", async () => {
      const researchConfig: FeatureEngineeringConfig = {
        profile: "research",
        customAllowlist: null,
      };

      render(
        <FeatureProfileSettings
          config={researchConfig}
          onChange={mockOnChange}
        />
      );

      const productionRadio = screen.getByRole("radio", {
        name: /本番用/,
      });
      fireEvent.click(productionRadio);

      await waitFor(() => {
        expect(mockOnChange).toHaveBeenCalledWith("profile", "production");
      });
    });

    it("プロファイル変更が動作する（production → research）", async () => {
      render(
        <FeatureProfileSettings config={mockConfig} onChange={mockOnChange} />
      );

      const researchRadio = screen.getByRole("radio", {
        name: /研究用/,
      });
      fireEvent.click(researchRadio);

      await waitFor(() => {
        expect(mockOnChange).toHaveBeenCalledWith("profile", "research");
      });
    });
  });

  describe("プロファイル情報表示", () => {
    it("productionプロファイルの説明が表示される", () => {
      render(
        <FeatureProfileSettings config={mockConfig} onChange={mockOnChange} />
      );

      // 複数箇所に表示されるテキストはgetAllByTextを使用
      expect(screen.getAllByText(/本番用（選択された特徴量）/).length).toBeGreaterThan(0);
      expect(screen.getByText(/約40個/)).toBeInTheDocument();
      expect(
        screen.getByText(/本番運用に適しており、計算が高速/)
      ).toBeInTheDocument();
    });

    it("researchプロファイルの説明が表示される", () => {
      const researchConfig: FeatureEngineeringConfig = {
        profile: "research",
        customAllowlist: null,
      };

      render(
        <FeatureProfileSettings
          config={researchConfig}
          onChange={mockOnChange}
        />
      );

      // 複数箇所に表示されるテキストはgetAllByTextを使用
      expect(screen.getAllByText(/研究用（全特徴量）/).length).toBeGreaterThan(0);
      expect(screen.getByText(/約108個/)).toBeInTheDocument();
      expect(
        screen.getByText(/研究・実験用途に適していますが/)
      ).toBeInTheDocument();
    });
  });

  describe("カスタムallowlist入力", () => {
    it("有効なJSON配列入力が処理される", async () => {
      render(
        <FeatureProfileSettings config={mockConfig} onChange={mockOnChange} />
      );

      const textarea = screen.getByRole("textbox");
      const validJson = '["RSI_14", "MACD_Signal", "BB_Position"]';

      fireEvent.change(textarea, { target: { value: validJson } });

      await waitFor(() => {
        expect(mockOnChange).toHaveBeenCalledWith("customAllowlist", [
          "RSI_14",
          "MACD_Signal",
          "BB_Position",
        ]);
      });
    });

    it("空文字列入力はnullに変換される", async () => {
      render(
        <FeatureProfileSettings config={mockConfig} onChange={mockOnChange} />
      );

      const textarea = screen.getByRole("textbox");
      fireEvent.change(textarea, { target: { value: "" } });

      // 空文字列はnullに変換される
      expect(mockOnChange).toHaveBeenCalledWith("customAllowlist", null);
    });

    it("空配列入力はnullに変換される", () => {
      render(
        <FeatureProfileSettings config={mockConfig} onChange={mockOnChange} />
      );

      const textarea = screen.getByRole("textbox");
      
      // 初期値は "[]" なので、まず別の値に変更してから "[]" に戻す
      fireEvent.change(textarea, { target: { value: '["test"]' } });
      
      // 次に空配列 "[]" に変更
      fireEvent.change(textarea, { target: { value: "[]" } });
      
      // 最後の呼び出しでnullが渡されることを確認
      const lastCall = mockOnChange.mock.calls[mockOnChange.mock.calls.length - 1];
      expect(lastCall).toEqual(["customAllowlist", null]);
    });

    it("無効なJSON入力はエラーを表示する", async () => {
      render(
        <FeatureProfileSettings config={mockConfig} onChange={mockOnChange} />
      );

      const textarea = screen.getByRole("textbox");
      const invalidJson = '["RSI", "MACD"'; // 閉じカッコなし

      fireEvent.change(textarea, { target: { value: invalidJson } });

      await waitFor(() => {
        expect(screen.getByText(/無効なJSON形式です/)).toBeInTheDocument();
      });

      // エラー時はonChangeが呼ばれない（配列以外）
      expect(mockOnChange).not.toHaveBeenCalled();
    });

    it("配列以外のJSON入力はエラーを表示する", async () => {
      render(
        <FeatureProfileSettings config={mockConfig} onChange={mockOnChange} />
      );

      const textarea = screen.getByRole("textbox");
      const objectJson = '{"feature": "RSI"}';

      fireEvent.change(textarea, { target: { value: objectJson } });

      await waitFor(() => {
        expect(
          screen.getByText(/配列形式で入力してください/)
        ).toBeInTheDocument();
      });
    });

    it("有効な配列を入力した後、onChangeが呼ばれる", async () => {
      render(
        <FeatureProfileSettings config={mockConfig} onChange={mockOnChange} />
      );

      const textarea = screen.getByRole("textbox");
      const validJson = '["RSI_14", "MACD_Signal", "BB_Position"]';

      fireEvent.change(textarea, { target: { value: validJson } });

      await waitFor(() => {
        expect(mockOnChange).toHaveBeenCalledWith("customAllowlist", [
          "RSI_14",
          "MACD_Signal",
          "BB_Position",
        ]);
      });
    });
  });

  describe("カスタムallowlist表示", () => {
    it("カスタムallowlistが設定されている場合、特徴量数が表示される", () => {
      const configWithAllowlist: FeatureEngineeringConfig = {
        profile: "production",
        customAllowlist: ["RSI_14", "MACD_Signal", "BB_Position"],
      };

      render(
        <FeatureProfileSettings
          config={configWithAllowlist}
          onChange={mockOnChange}
        />
      );

      expect(
        screen.getByText(/3個の特徴量が指定されています/)
      ).toBeInTheDocument();
    });

    it("カスタムallowlistが設定されている場合、警告メッセージが表示される", () => {
      const configWithAllowlist: FeatureEngineeringConfig = {
        profile: "production",
        customAllowlist: ["RSI_14", "MACD_Signal"],
      };

      render(
        <FeatureProfileSettings
          config={configWithAllowlist}
          onChange={mockOnChange}
        />
      );

      expect(
        screen.getByText(/カスタムallowlistが設定されているため/)
      ).toBeInTheDocument();
    });

    it("カスタムallowlistがnullの場合、警告メッセージは表示されない", () => {
      render(
        <FeatureProfileSettings config={mockConfig} onChange={mockOnChange} />
      );

      expect(
        screen.queryByText(/カスタムallowlistが設定されているため/)
      ).not.toBeInTheDocument();
    });
  });

  describe("説明文", () => {
    it("コンポーネントの説明文が表示される", () => {
      render(
        <FeatureProfileSettings config={mockConfig} onChange={mockOnChange} />
      );

      expect(
        screen.getByText(/特徴量エンジニアリング設定は、機械学習モデルで使用/)
      ).toBeInTheDocument();
    });

    it("プロファイル選択の説明が表示される", () => {
      render(
        <FeatureProfileSettings config={mockConfig} onChange={mockOnChange} />
      );

      expect(
        screen.getByText(/全ての特徴量を使用します。研究・実験用途/)
      ).toBeInTheDocument();
      expect(
        screen.getByText(/厳選された特徴量のみを使用します。本番運用/)
      ).toBeInTheDocument();
    });
  });

  describe("エラーハンドリング", () => {
    it("エラー後に有効な入力をすると、エラーメッセージが消える", async () => {
      render(
        <FeatureProfileSettings config={mockConfig} onChange={mockOnChange} />
      );

      const textarea = screen.getByRole("textbox");

      // 無効な入力
      fireEvent.change(textarea, { target: { value: '["invalid' } });
      await waitFor(() => {
        expect(screen.getByText(/無効なJSON形式です/)).toBeInTheDocument();
      });

      // 有効な入力
      fireEvent.change(textarea, { target: { value: '["valid"]' } });
      await waitFor(() => {
        expect(
          screen.queryByText(/無効なJSON形式です/)
        ).not.toBeInTheDocument();
      });
    });

    it("エラー時にも既存のcustomAllowlistは保持される", async () => {
      const configWithAllowlist: FeatureEngineeringConfig = {
        profile: "production",
        customAllowlist: ["RSI_14"],
      };

      render(
        <FeatureProfileSettings
          config={configWithAllowlist}
          onChange={mockOnChange}
        />
      );

      const textarea = screen.getByRole("textbox");
      fireEvent.change(textarea, { target: { value: "invalid json" } });

      await waitFor(() => {
        expect(screen.getByText(/無効なJSON形式です/)).toBeInTheDocument();
      });

      // onChangeが呼ばれていないことを確認（エラー時は更新しない）
      expect(mockOnChange).not.toHaveBeenCalled();
    });
  });

  describe("エッジケース", () => {
    it("空のcustomAllowlistでレンダリングできる", () => {
      const emptyConfig: FeatureEngineeringConfig = {
        profile: "production",
        customAllowlist: [],
      };

      expect(() => {
        render(
          <FeatureProfileSettings
            config={emptyConfig}
            onChange={mockOnChange}
          />
        );
      }).not.toThrow();
    });

    it("大きな配列でも処理できる", async () => {
      render(
        <FeatureProfileSettings config={mockConfig} onChange={mockOnChange} />
      );

      const textarea = screen.getByRole("textbox");
      const largeArray = JSON.stringify(
        Array.from({ length: 100 }, (_, i) => `Feature_${i}`)
      );

      fireEvent.change(textarea, { target: { value: largeArray } });

      await waitFor(() => {
        expect(mockOnChange).toHaveBeenCalledWith(
          "customAllowlist",
          expect.any(Array)
        );
      });

      // onChangeが正しい配列長で呼ばれたことを確認
      const calledArray = mockOnChange.mock.calls[0][1];
      expect(Array.isArray(calledArray)).toBe(true);
      expect(calledArray.length).toBe(100);
    });

    it("空白文字のみの入力はnullに変換される", async () => {
      render(
        <FeatureProfileSettings config={mockConfig} onChange={mockOnChange} />
      );

      const textarea = screen.getByRole("textbox");
      fireEvent.change(textarea, { target: { value: "   " } });

      await waitFor(() => {
        expect(mockOnChange).toHaveBeenCalledWith("customAllowlist", null);
      });
    });

    it("複数行のJSON入力も正しく処理される", async () => {
      render(
        <FeatureProfileSettings config={mockConfig} onChange={mockOnChange} />
      );

      const textarea = screen.getByRole("textbox");
      const multilineJson = `[
        "RSI_14",
        "MACD_Signal",
        "BB_Position"
      ]`;

      fireEvent.change(textarea, { target: { value: multilineJson } });

      await waitFor(() => {
        expect(mockOnChange).toHaveBeenCalledWith("customAllowlist", [
          "RSI_14",
          "MACD_Signal",
          "BB_Position",
        ]);
      });
    });
  });

  describe("初期値", () => {
    it("customAllowlistがnullの場合、textareaは空配列のJSON表示になる", () => {
      render(
        <FeatureProfileSettings config={mockConfig} onChange={mockOnChange} />
      );

      const textarea = screen.getByRole("textbox") as HTMLTextAreaElement;
      expect(textarea.value).toBe("[]");
    });

    it("customAllowlistが設定されている場合、textareaにJSON表示される", () => {
      const configWithAllowlist: FeatureEngineeringConfig = {
        profile: "production",
        customAllowlist: ["RSI_14", "MACD_Signal"],
      };

      render(
        <FeatureProfileSettings
          config={configWithAllowlist}
          onChange={mockOnChange}
        />
      );

      const textarea = screen.getByRole("textbox") as HTMLTextAreaElement;
      expect(JSON.parse(textarea.value)).toEqual([
        "RSI_14",
        "MACD_Signal",
      ]);
    });
  });
});