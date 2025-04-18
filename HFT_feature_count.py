import pandas as pd
import numpy as np
import pandas_ta as ta

NANOSECOND = 1
MICROSECOND = 1000
MILLISECOND = 1000000
SECOND = 1000000000

from typing import Union


def trades_balance(trades_df: pd.DataFrame, window: Union[str, int]) -> pd.Series:
    sells = trades_df["ask_amount"].rolling(window=window, min_periods=1).sum()
    buys = trades_df["bid_amount"].rolling(window=window, min_periods=1).sum()
    return (sells - buys) / (sells + buys + 1e-8)


def calc_imbalance(lobs, depth):
    """
    Computes the order book imbalance.

    Parameters:
    - lob: pd.DataFrame row containing LOB data.

    Returns:
    - imbalance_value: float
    """
    bid_amount = 0
    ask_amount = 0
    for i in range(depth):
        bid_amount += lobs[f"bids[{i}].amount"]
        ask_amount += lobs[f"asks[{i}].amount"]
    imbalance_value = (bid_amount - ask_amount) / (bid_amount + ask_amount)
    return imbalance_value


def vwap(books_df: pd.DataFrame, lvl_count: int) -> pd.Series:
    """Volume-weighted average price."""
    ask_weighted_price = sum(
        books_df[f"asks[{i}].price"] * books_df[f"asks[{i}].amount"]
        for i in range(lvl_count)
    )
    ask_volume = sum(books_df[f"asks[{i}].amount"] for i in range(lvl_count))

    bid_weighted_price = sum(
        books_df[f"bids[{i}].price"] * books_df[f"bids[{i}].amount"]
        for i in range(lvl_count)
    )
    bid_volume = sum(books_df[f"bids[{i}].amount"] for i in range(lvl_count))

    total_weighted_price = ask_weighted_price + bid_weighted_price
    total_volume = ask_volume + bid_volume

    vwap = total_weighted_price / total_volume

    return vwap / books_df["mid_price"]


def calc_features(
        lobs: pd.DataFrame | None,
        agg_trades: pd.DataFrame | None,
        lobs_embedding: pd.DataFrame | None,
        target_data: pd.DataFrame | None,
    ) -> pd.DataFrame:
        """
        Parameters:
        - lobs: pd.DataFrame of limit orderbooks.
        - agg_trades: pd.DataFrame of aggregated trades.
        - lobs_embedding: pd.DataFrame of embedding over limit orderbooks.
        - target_data: pd.DataFrame with target timestamps.


        Returns:
        - features: pd.DataFrame with features aligned to target_data.index.
        """
        target_data['amount_signed'] = target_data['amount']*target_data['side']
        candles = target_data.resample('250ms').agg({
            'price': [('highPrice', 'max'),
                  ('lowPrice', 'min'),
                  ('openPrice', 'first'),
                  ('closePrice', 'last')],
            'amount': [('volume', 'sum')],
            'amount_signed': [('delta', 'sum')]
        })

        candles.columns = candles.columns.droplevel(0)
        candles.index = candles.index.shift(1)

        candles['closePrice'] = candles['closePrice'].fillna(method='ffill')
        candles['openPrice'] = candles['openPrice'].fillna(candles['closePrice'].shift(1))
        candles['highPrice'] = candles['highPrice'].fillna(candles['closePrice'].shift(1))
        candles['lowPrice'] = candles['lowPrice'].fillna(candles['closePrice'].shift(1))

        candles['closePrice'] = candles['closePrice'].astype(float)
        candles['openPrice'] = candles['openPrice'].astype(float)
        candles['highPrice'] = candles['highPrice'].astype(float)
        candles['lowPrice'] = candles['lowPrice'].astype(float)

        lobs["mid_price"] = (lobs["asks[0].price"] + lobs["bids[0].price"]) / 2

        btcusdt_mid_price = lobs_embedding[lobs_embedding["instrument"] == "BTCUSDT"][
            "mid_price"
        ]
        ethusdt_mid_price = lobs_embedding[lobs_embedding["instrument"] == "ETHUSDT"][
            "mid_price"
        ]

        new_f = pd.DataFrame()

        lobs['log_chg'] = np.log(lobs["mid_price"]).diff()
        new_f['log_chg_100'] = lobs['log_chg'].rolling('100l').sum()
        new_f['log_chg_500'] = lobs['log_chg'].rolling('500l').sum()
        new_f['log_chg_1000'] = lobs['log_chg'].rolling('1s').sum()

        new_f = new_f.apply(lambda x: x.asof(target_data.index) * target_data.side)

        #Теханал
        sol_trades = agg_trades[agg_trades["instrument"] == "SOLUSDT"]
        sol_trades[['bid_max_price', 'ask_max_price', 'bid_min_price', 'ask_min_price', 'bid_mean_price', 'ask_mean_price']] = sol_trades[['bid_max_price', 'ask_max_price', 'bid_min_price', 'ask_min_price', 'bid_mean_price', 'ask_mean_price']].ffill()
        sol_trades[['bid_count', 'ask_count', 'bid_amount', 'ask_amount']] = sol_trades[['bid_count', 'ask_count', 'bid_amount', 'ask_amount']].fillna(0)

        # На основе свечей
        for length in [6,12,16,20,25,30,50,60,80]:
            candles.ta(kind='SMA',append=True,centered=False,close='openPrice',length=length)
            candles[f'f_SMA_{length}'] = np.where(candles['closePrice']>candles[f'SMA_{length}'],1,2)

            candles.ta(kind='ATR',append=True,centered=False,high='highPrice',low='lowPrice',volume='volume',close='closePrice',length=length)
            candles[f'f_ATR_{length}'] = np.where(candles[f'ATRr_{length}']>candles[f'ATRr_{length}'].rolling(window=6).mean(),2,1)
            candles[f'f_ATR_div_{length}'] = candles[f'ATRr_{length}']/candles[f'closePrice']

            candles.ta(kind='RSI',append=True,centered=False,close='closePrice',length=length)
            candles['f_rsi'] = np.where(candles[f'RSI_{length}']<20,1,np.where(candles[f'RSI_{length}']>80,2,1))

            candles.ta(kind='PVT',append=True,centered=False,volume='volume',close='closePrice')
            candles['f_pvt'] = np.where(candles['PVT']>candles['PVT'].rolling(window=length).mean(),2,1)

            candles[f'f_volatility_{length}'] = candles['closePrice'].rolling(window=length).std()



        candles.ta(kind='MACD',append=True,centered=False,high='highPrice',low='lowPrice',volume='volume',close='closePrice',length=6)
        candles['f_macd'] = np.where(candles[f'MACDh_12_26_9']>0,2,1)

        candles['f_ao'] = ta.ao(candles['highPrice'], candles['lowPrice'])
        candles['f_sma_ao'] = ta.sma(candles['f_ao'], length=5)
        candles['f_ac'] = candles['f_ao'] - candles['f_sma_ao']
        candles['f_a_jaw'] = ((candles['highPrice'] + candles['lowPrice'])/2).rolling(window=13).mean().shift(8)
        candles['f_a_teeth'] = ((candles['highPrice'] + candles['lowPrice'])/2).rolling(window=8).mean().shift(5)
        candles['f_a_lips'] = ((candles['highPrice'] + candles['lowPrice'])/2).rolling(window=5).mean().shift(3)
        candles['f_apo'] = ta.apo((candles['highPrice'] + candles['lowPrice'])/2)
        candles = pd.concat([candles, ta.aroon(candles['highPrice'], candles['lowPrice'], length=20)], axis=1)
        candles['f_atr'] = ta.atr(candles['highPrice'], candles['lowPrice'], candles['closePrice'])
        candles = pd.concat([candles, ta.bbands(candles['closePrice'], length=14)], axis=1)
        candles = pd.concat([candles, ta.ichimoku(candles['highPrice'], candles['lowPrice'], candles['closePrice'])[0]], axis=1)
        candles['f_cmo'] = ta.cmo(candles['closePrice'], length=19)
        candles['f_chaikin'] = ta.cmf(candles['highPrice'], candles['lowPrice'], candles['closePrice'], candles['volume'])

        candles['f_BBU_14_2.0'] = np.where(candles['closePrice'] > candles['BBU_14_2.0'], 1, 2)
        candles['f_BBL_14_2.0'] = np.where(candles['closePrice'] > candles['BBL_14_2.0'], 2, 1)
        candles['f_BBM'] = np.where(candles['closePrice'] > candles['BBM_14_2.0'], 2, 1)

        candles.ta(kind='OBV',append=True,centered=False,high='highPrice',low='lowPrice',volume='volume',close='closePrice',length=6)
        candles[f'f_OBV'] = np.where(candles['OBV']<candles['OBV'].rolling(window=12).mean(),1,2)

        candles['f_chaikin_1'] = np.where(candles['highPrice'] > candles['ISA_9'], 2, 1)
        candles['f_chaikin_2'] = np.where(candles['highPrice'] > candles['ISB_26'], 1, 2)
        candles['f_chaikin_3'] = np.where(candles['ITS_9'] > candles['IKS_26'], 1, 2)

        candles['f_ao_1'] = np.where(candles['f_ao'] > 0, 2, 1)
        candles['f_ao2'] = np.where(candles['f_ao'] > candles['f_sma_ao'], 2, 1)

        candles['f_a_1'] = np.where(candles['f_a_lips'] > candles['f_a_teeth'], 2, 1)
        candles['f_a_2'] = np.where(candles['f_a_teeth'] > candles['f_a_jaw'], 2, 1)

        candles['f_apo_1'] = np.where(candles['f_apo'] > 0, 2, 1)

        candles.replace([np.inf, -np.inf], 0, inplace=True)
        candles = candles.fillna(0)
        candles = candles.apply(lambda x: x.asof(target_data.index) * target_data.side)

        candles['uns_f_ATR_div_50'] = candles['f_ATR_div_50'] * target_data.side
        candles['ATRr_50'] = candles['ATRr_50'] * target_data.side
        candles['f_a_jaw'] = candles['f_a_jaw'] * target_data.side
        candles['f_a_teeth'] = candles['f_a_teeth'] * target_data.side
        candles['f_a_lips'] = candles['f_a_lips'] * target_data.side
        candles['uns_SMA_6'] = candles['SMA_6'] * target_data.side
        candles['uns_SMA_50'] = candles['SMA_50'] * target_data.side
        candles['uns_ISB_26'] = candles['ISB_26'] * target_data.side
        candles['uns_f_a_2'] = candles['f_a_2'] * target_data.side
        candles['uns_ITS_9'] = candles['ITS_9'] * target_data.side
        candles['uns_ISA_9'] = candles['ISA_9'] * target_data.side

        # Базовые

        ob_depth_ask = (
            lobs['asks[19].price'] - lobs['asks[0].price']
        ).asof(target_data.index) * target_data.side
        ob_depth_ask.name = "ob_depth_ask"

        ob_depth_bid = (
            lobs['bids[0].price'] - lobs['bids[19].price']
        ).asof(target_data.index) * target_data.side
        ob_depth_bid.name = "ob_depth_bid"

        main_btcusdt_dev = (
            lobs["mid_price"] / (btcusdt_mid_price.asof(lobs.index) + 1e-6)
        ).asof(target_data.index) * target_data.side
        main_btcusdt_dev.name = "main_btcusdt_dev"

        main_ethusdt_dev = (
            lobs["mid_price"] / (ethusdt_mid_price.asof(lobs.index) + 1e-6)
        ).asof(target_data.index) * target_data.side
        main_ethusdt_dev.name = "main_ethusdt_dev"

        distance_to_mid_price = (
            target_data.price / (lobs["mid_price"].asof(target_data.index) + 1e-6) - 1
        ) * target_data.side
        distance_to_mid_price.name = "distance_to_mid_price"
        
        imbalance = pd.DataFrame()
        for i in range(1, 20):
            imbalance[f'imbalance[{i}]'] = (
                calc_imbalance(lobs, i).asof(target_data.index) * target_data.side
            )

        for i in range(1, 10):
            imbalance[f'imbalance[3]_{i}'] = imbalance[f'imbalance[3]'].shift(i)
            imbalance[f'imbalance[5]_{i}'] = imbalance[f'imbalance[5]'].shift(i)
            imbalance[f'imbalance[10]_{i}'] = imbalance[f'imbalance[10]'].shift(i)
            imbalance[f'imbalance[10]_{i}'] = imbalance[f'imbalance[19]'].shift(i)

        vwap_df = pd.DataFrame()
        for i in (1, 3, 5, 10):
            vwap_df[f'vwap{i}'] = vwap(lobs, i).asof(target_data.index) * target_data.side

        solusdt_agg_trades = agg_trades[agg_trades["instrument"] == "SOLUSDT"]
        solusdt_agg_trades[['bid_count', 'ask_count', 'bid_amount', 'ask_amount']] = solusdt_agg_trades[['bid_count', 'ask_count', 'bid_amount', 'ask_amount']].fillna(0)
        solusdt_agg_trades.index = pd.to_datetime(solusdt_agg_trades.index)
        trades_ratio_series = (
            trades_balance(solusdt_agg_trades, "10s").asof(target_data.index)
            * target_data.side
        )
        trades_ratio_series.name = "trades_ratio"

        time = pd.Series(pd.to_datetime(target_data.index).minute + pd.to_datetime(target_data.index).hour*60, index=target_data.index)
        time.name = "time"

        is_weekend = pd.Series(pd.to_datetime(target_data.index).day_of_week.isin([5, 6]), index=target_data.index)
        is_weekend.name = "is_weekend"

        features = pd.concat(
            [
                candles,
                new_f,
                ob_depth_ask,
                ob_depth_bid,
                target_data.amount,
                target_data.amount_signed,
                target_data.side,
                imbalance,
                vwap_df,
                trades_ratio_series,
                distance_to_mid_price,
                main_ethusdt_dev,
                main_btcusdt_dev,
                time,
                is_weekend
            ],
            axis=1,
        )

        features = features.loc[:, ~features.columns.duplicated()]

        # По итогам отбора

        features = features[['side', 'amount', 'log_chg_100', 'ob_depth_ask', 'ob_depth_bid',
       'imbalance[1]', 'imbalance[2]', 'imbalance[3]', 'imbalance[4]',
       'imbalance[5]', 'imbalance[6]', 'imbalance[7]', 'imbalance[8]',
       'imbalance[9]', 'imbalance[10]', 'imbalance[11]', 'imbalance[12]',
       'imbalance[13]', 'imbalance[14]', 'imbalance[15]', 'imbalance[16]',
       'imbalance[17]', 'imbalance[18]', 'imbalance[19]', 'imbalance[3]_1',
       'imbalance[5]_1', 'imbalance[10]_1', 'imbalance[3]_2', 'imbalance[5]_2',
       'imbalance[3]_3', 'imbalance[5]_3', 'imbalance[5]_4', 'imbalance[5]_6',
       'imbalance[5]_9', 'vwap3', 'vwap5', 'vwap10', 'distance_to_mid_price',
       'time', 'is_weekend', 'openPrice', 'SMA_6', 'SMA_50', 'ATRr_50',
       'BBL_14_2.0', 'BBU_14_2.0', 'ISA_9', 'ISB_26', 'ITS_9', 'f_a_2',
       'uns_f_ATR_div_50', 'f_a_teeth', 'f_a_lips', 'uns_SMA_50', 'uns_ISB_26',
       'uns_f_a_2', 'uns_ISA_9']]

        return features