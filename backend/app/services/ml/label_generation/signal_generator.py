"""
SignalGenerator - メタラベリング用のシグナル生成クラス

Primary Model として機能するルールベースのシグナルジェネレーター。
ML モデル（Secondary Model）が判定するための「候補イベント」を生成します。
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class SignalGenerator:
    """
    メタラベリング用のシグナル生成クラス

    ボリンジャーバンド、ドンチャンチャネル、出来高急増などの
    テクニカル指標に基づいて、潜在的なトレード候補（イベント）を検出します。

    これらのシグナルは「高Recall」を目指し、多くの候補を広く拾います。
    その後、ML モデル（Secondary Model）が「高Precision」で
    真のトレード機会をフィルタリングします。

    Examples:
        >>> generator = SignalGenerator()
        >>> bb_events = generator.get_bb_breakout_events(ohlcv_df)
        >>> combined_events = generator.get_combined_events(
        ...     ohlcv_df,
        ...     use_bb=True,
        ...     use_volume=True
        ... )
    """

    def get_bb_breakout_events(
        self,
        df: pd.DataFrame,
        window: int = 20,
        dev: float = 2.0,
        price_column: str = "close",
    ) -> pd.DatetimeIndex:
        """
        ボリンジャーバンド ブレイクアウトイベントを検出

        価格が上バンド（+2σ）を上抜けたか、下バンド（-2σ）を下抜けた
        タイミングをブレイクアウトイベントとして検出します。

        Args:
            df: OHLCV DataFrame
            window: 移動平均の期間（デフォルト: 20）
            dev: 標準偏差の倍数（デフォルト: 2.0）
            price_column: 価格カラム名（デフォルト: "close"）

        Returns:
            pd.DatetimeIndex: ブレイクアウトが発生したタイムスタンプ

        Notes:
            - ブレイクアウトは「前の足が範囲内 & 現在の足が範囲外」で判定
            - これにより、バンド外に留まり続けているケースを除外
        """
        if len(df) < window:
            logger.warning(
                f"データ数（{len(df)}）がウィンドウサイズ（{window}）より小さいため、"
                "空のイベントを返します。"
            )
            return pd.DatetimeIndex([])

        prices = df[price_column]

        # ボリンジャーバンド計算
        sma = prices.rolling(window=window, min_periods=window).mean()
        std = prices.rolling(window=window, min_periods=window).std()

        upper_band = sma + (dev * std)
        lower_band = sma - (dev * std)

        # ブレイクアウト条件
        # 上抜け: 価格が上バンドを超える
        upper_breakout = prices > upper_band

        # 下抜け: 価格が下バンドを下回る
        lower_breakout = prices < lower_band

        # 前の足と比較して「新規」ブレイクアウトのみを検出
        # （バンド外に留まり続けているケースを除外）
        new_upper_breakout = upper_breakout & ~upper_breakout.shift(1, fill_value=False)
        new_lower_breakout = lower_breakout & ~lower_breakout.shift(1, fill_value=False)

        # 上抜けまたは下抜けのいずれか
        breakout_mask = new_upper_breakout | new_lower_breakout

        # イベントが発生したタイムスタンプを抽出
        events = df.index[breakout_mask]

        logger.debug(
            f"BBブレイクアウト検出: {len(events)}件 " f"(window={window}, dev={dev})"
        )

        return events

    def get_donchian_breakout_events(
        self,
        df: pd.DataFrame,
        window: int = 20,
        price_column_high: str = "high",
        price_column_low: str = "low",
    ) -> pd.DatetimeIndex:
        """
        ドンチャンチャネル ブレイクアウトイベントを検出

        過去 N 期間の最高値を更新（上抜け）または
        最安値を更新（下抜け）したタイミングをブレイクアウトとして検出します。

        Args:
            df: OHLCV DataFrame
            window: ドンチャンチャネルの期間（デフォルト: 20）
            price_column_high: 高値カラム名（デフォルト: "high"）
            price_column_low: 安値カラム名（デフォルト: "low"）

        Returns:
            pd.DatetimeIndex: ブレイクアウトが発生したタイムスタンプ

        Notes:
            - 高値更新 = 過去 N 期間の最高値を現在の高値が上回る
            - 安値更新 = 過去 N 期間の最安値を現在の安値が下回る
        """
        if len(df) < window:
            logger.warning(
                f"データ数（{len(df)}）がウィンドウサイズ（{window}）より小さいため、"
                "空のイベントを返します。"
            )
            return pd.DatetimeIndex([])

        high_prices = df[price_column_high]
        low_prices = df[price_column_low]

        # 過去 N 期間の最高値と最安値
        rolling_high = high_prices.rolling(window=window, min_periods=window).max()
        rolling_low = low_prices.rolling(window=window, min_periods=window).min()

        # 前の足の最高値・最安値と比較
        prev_rolling_high = rolling_high.shift(1)
        prev_rolling_low = rolling_low.shift(1)

        # ブレイクアウト条件
        # 上ブレイク: 現在の高値が過去の最高値を更新
        upper_breakout = (high_prices > prev_rolling_high) & prev_rolling_high.notna()

        # 下ブレイク: 現在の安値が過去の最安値を更新
        lower_breakout = (low_prices < prev_rolling_low) & prev_rolling_low.notna()

        # いずれかのブレイクアウト
        breakout_mask = upper_breakout | lower_breakout

        # イベントが発生したタイムスタンプを抽出
        events = df.index[breakout_mask]

        logger.debug(f"ドンチャンブレイクアウト検出: {len(events)}件 (window={window})")

        return events

    def get_volume_spike_events(
        self,
        df: pd.DataFrame,
        window: int = 20,
        multiplier: float = 2.0,
        volume_column: str = "volume",
    ) -> pd.DatetimeIndex:
        """
        出来高急増イベントを検出

        現在の出来高が過去 N 期間の平均出来高の X 倍以上になった
        タイミングを急増イベントとして検出します。

        Args:
            df: OHLCV DataFrame
            window: 移動平均の期間（デフォルト: 20）
            multiplier: 平均の何倍で「急増」とみなすか（デフォルト: 2.0）
            volume_column: 出来高カラム名（デフォルト: "volume"）

        Returns:
            pd.DatetimeIndex: 出来高急増が発生したタイムスタンプ

        Notes:
            - 平均計算には現在の足を含まない（shift使用）
            - multiplier=2.0 なら、平均の2倍以上で急増とみなす
        """
        if len(df) < window:
            logger.warning(
                f"データ数（{len(df)}）がウィンドウサイズ（{window}）より小さいため、"
                "空のイベントを返します。"
            )
            return pd.DatetimeIndex([])

        volumes = df[volume_column]

        # 過去 N 期間の平均出来高（現在の足は含まない）
        avg_volume = volumes.rolling(window=window, min_periods=window).mean().shift(1)

        # 急増条件: 現在の出来高が平均の X 倍以上
        spike_mask = (volumes >= avg_volume * multiplier) & avg_volume.notna()

        # イベントが発生したタイムスタンプを抽出
        events = df.index[spike_mask]

        logger.debug(
            f"出来高急増検出: {len(events)}件 "
            f"(window={window}, multiplier={multiplier}x)"
        )

        return events

    def get_combined_events(
        self,
        df: pd.DataFrame,
        use_bb: bool = True,
        use_donchian: bool = True,
        use_volume: bool = True,
        bb_window: int = 20,
        bb_dev: float = 2.0,
        donchian_window: int = 20,
        volume_window: int = 20,
        volume_multiplier: float = 2.0,
    ) -> pd.DatetimeIndex:
        """
        複数のシグナルを組み合わせてイベントを取得

        BB ブレイクアウト、ドンチャンブレイクアウト、出来高急増の
        いずれかが発生したタイミングを統合してイベントとして返します。

        Args:
            df: OHLCV DataFrame
            use_bb: BB ブレイクアウトを使用するか（デフォルト: True）
            use_donchian: ドンチャンブレイクアウトを使用するか（デフォルト: True）
            use_volume: 出来高急増を使用するか（デフォルト: True）
            bb_window: BB の期間
            bb_dev: BB の標準偏差倍数
            donchian_window: ドンチャンチャネルの期間
            volume_window: 出来高平均の期間
            volume_multiplier: 出来高急増の倍数

        Returns:
            pd.DatetimeIndex: すべてのシグナルを統合したイベントのタイムスタンプ

        Notes:
            - 複数のシグナルで同じタイムスタンプが検出された場合は重複除去される
            - いずれのシグナルも使用しない場合は空のインデックスを返す
        """
        all_events = []

        if use_bb:
            bb_events = self.get_bb_breakout_events(df, window=bb_window, dev=bb_dev)
            all_events.append(bb_events)
            logger.debug(f"BB イベント: {len(bb_events)}件")

        if use_donchian:
            donchian_events = self.get_donchian_breakout_events(
                df, window=donchian_window
            )
            all_events.append(donchian_events)
            logger.debug(f"ドンチャン イベント: {len(donchian_events)}件")

        if use_volume:
            volume_events = self.get_volume_spike_events(
                df, window=volume_window, multiplier=volume_multiplier
            )
            all_events.append(volume_events)
            logger.debug(f"出来高急増 イベント: {len(volume_events)}件")

        if not all_events:
            logger.warning("シグナルが選択されていません。空のイベントを返します。")
            return pd.DatetimeIndex([])

        # すべてのイベントを統合（重複除去）
        combined = pd.DatetimeIndex([])
        for events in all_events:
            combined = combined.union(events)

        # ソート
        combined = combined.sort_values()

        logger.info(
            f"統合イベント検出完了: {len(combined)}件 "
            f"(BB={use_bb}, Donchian={use_donchian}, Volume={use_volume})"
        )

        return combined


