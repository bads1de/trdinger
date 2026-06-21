import { renderHook, act } from "@testing-library/react";
import { useSetLimit, wrapInArray } from "@/utils/hookUtils";

describe("useSetLimit", () => {
  it("setParamsを呼び出してlimitを設定すること", () => {
    const mockSetParams = jest.fn();
    const { result } = renderHook(() => useSetLimit(mockSetParams));

    act(() => {
      result.current(200);
    });

    expect(mockSetParams).toHaveBeenCalledWith({ limit: 200 });
  });

  it("setParamsを呼び出して異なるlimit値で設定できること", () => {
    const mockSetParams = jest.fn();
    const { result } = renderHook(() => useSetLimit(mockSetParams));

    act(() => {
      result.current(50);
    });

    expect(mockSetParams).toHaveBeenCalledWith({ limit: 50 });
  });

  it("setParamsが変更されても正しく動作すること", () => {
    const mockSetParams1 = jest.fn();
    const mockSetParams2 = jest.fn();
    const { result, rerender } = renderHook(
      (props) => useSetLimit(props.setParams),
      { initialProps: { setParams: mockSetParams1 } }
    );

    act(() => {
      result.current(100);
    });
    expect(mockSetParams1).toHaveBeenCalledWith({ limit: 100 });

    rerender({ setParams: mockSetParams2 });

    act(() => {
      result.current(300);
    });
    expect(mockSetParams2).toHaveBeenCalledWith({ limit: 300 });
  });
});

describe("wrapInArray", () => {
  it("単一オブジェクトを配列にラップすること", () => {
    const obj = { id: 1, name: "test" };
    expect(wrapInArray(obj)).toEqual([{ id: 1, name: "test" }]);
  });

  it("文字列を配列にラップすること", () => {
    expect(wrapInArray("hello")).toEqual(["hello"]);
  });

  it("数値を配列にラップすること", () => {
    expect(wrapInArray(42)).toEqual([42]);
  });

  it("nullを配列にラップすること", () => {
    expect(wrapInArray(null)).toEqual([null]);
  });

  it("undefinedを配列にラップすること", () => {
    expect(wrapInArray(undefined)).toEqual([undefined]);
  });

  it("配列をそのまま返すわけではないこと（ネストされる）", () => {
    const arr = [1, 2, 3];
    const result = wrapInArray(arr);
    expect(result).toEqual([[1, 2, 3]]);
    expect(Array.isArray(result)).toBe(true);
    expect(result.length).toBe(1);
  });
});
