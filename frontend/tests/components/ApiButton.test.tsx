import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import ApiButton from '@/components/button/ApiButton';

describe('ApiButton', () => {
  it('コンポーネントが正常にレンダリングされること', () => {
    render(<ApiButton onClick={() => {}}>テストボタン</ApiButton>);
    expect(screen.getByRole('button', { name: 'テストボタン' })).toBeInTheDocument();
  });

  it('ボタンをクリックするとonClickコールバックが呼び出されること', () => {
    const handleClick = jest.fn();
    render(<ApiButton onClick={handleClick}>クリックしてね</ApiButton>);
    const button = screen.getByRole('button', { name: 'クリックしてね' });
    fireEvent.click(button);
    expect(handleClick).toHaveBeenCalledTimes(1);
  });

  it('disabledプロパティがtrueの場合、onClickコールバックが呼び出されないこと', () => {
    const handleClick = jest.fn();
    render(<ApiButton onClick={handleClick} disabled={true}>無効ボタン</ApiButton>);
    const button = screen.getByRole('button', { name: '無効ボタン' });
    expect(button).toBeDisabled();
    fireEvent.click(button);
    expect(handleClick).not.toHaveBeenCalled();
  });

  it('loadingプロパティがtrueの場合、onClickコールバックが呼び出されないこと', () => {
    const handleClick = jest.fn();
    render(<ApiButton onClick={handleClick} loading={true}>ローディングボタン</ApiButton>);
    // loadingTextのデフォルト値でボタンを取得
    const button = screen.getByRole('button', { name: /読み込み中...|処理中.../ });
    // ローディング中はボタンが無効化されることを期待
    expect(button).toBeDisabled();
    fireEvent.click(button);
    expect(handleClick).not.toHaveBeenCalled();
    // 元のボタンテキストは表示されないことを確認
    expect(screen.queryByText('ローディングボタン')).not.toBeInTheDocument();
  });

  it('loadingプロパティがtrueの場合、loadingTextが表示されること', () => {
    render(<ApiButton onClick={() => {}} loading={true} loadingText="処理中...">ローディングボタン</ApiButton>);
    expect(screen.getByText('処理中...')).toBeInTheDocument();
    // 元のボタンテキストは表示されないことを確認
    expect(screen.queryByText('ローディングボタン')).not.toBeInTheDocument();
  });
});
