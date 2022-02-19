import numpy as np
import pandas as pd
from fyne import blackscholes

from utils import (get_col, cols_to_args, cols_to_kwargs, resample,
                   is_consecutive_unique, put_call_parity)
from calibration_app import heston_app


BUSINESS_DAYS_IN_YEAR = 252
DATE = pd.to_datetime('2019-10-10')
START_TIME = DATE + pd.to_timedelta('08:00:50')
END_TIME = DATE + pd.to_timedelta('08:01:20')
TIME_STEP = np.timedelta64(1, 's')
CALIBRATION_TIME = DATE + pd.to_timedelta('08:01:00')

PAYOFF_LABEL, EXPIRY_LABEL, STRIKE_LABEL = 'Payoff', 'Expiry', 'Strike'
TIME_LABEL, TIME_TO_EXPIRY_LABEL, JOIN_LABEL = 'Time', 'Time to expiry', 'ID'
ASK_PRICE_LABEL, BID_PRICE_LABEL = 'Ask price', 'Bid price'
ASK_SIZE_LABEL, BID_SIZE_LABEL = 'Ask size', 'Bid size'
FUTURES_INDEX_LABEL = [EXPIRY_LABEL]
OPTIONS_INDEX_LABEL = [PAYOFF_LABEL, EXPIRY_LABEL, STRIKE_LABEL]
CALL_CODE, PUT_CODE = 'Call', 'Put'
MID_PRICE_LABEL = 'Mid price'
UNDERLYING_MID_PRICE_LABEL = 'Underlying mid price'
DISCOUNTED_STRIKE_LABEL = 'Discounted strike'
PUT_FLAG_LABEL = 'Put'

FUTURES_META_KEY = '/futures'
FUTURES_QUOTES_KEY = '/futures_quotes'
OPTIONS_META_KEY = '/options'
OPTIONS_QUOTES_KEY = '/options_quotes'
FUTURES_META_LABEL_PAIRS = [('PRG_CODE', JOIN_LABEL),
                            ('DAEXPIRY', EXPIRY_LABEL)]
FUTURES_QUOTES_LABEL_PAIRS = [('prd', JOIN_LABEL),
                              ('ts_recv', TIME_LABEL),
                              ('bid', BID_PRICE_LABEL),
                              ('ask', ASK_PRICE_LABEL),
                              ('bid_size', BID_SIZE_LABEL),
                              ('ask_size', ASK_SIZE_LABEL)]
OPTIONS_META_LABEL_PAIRS = [('opt_id', JOIN_LABEL),
                            ('callput', PAYOFF_LABEL),
                            ('expiry', EXPIRY_LABEL),
                            ('strike', STRIKE_LABEL),
                            ('ttoexp', TIME_TO_EXPIRY_LABEL)]
OPTIONS_QUOTES_LABEL_PAIRS = [('prd_id', JOIN_LABEL),
                              ('ts_recv', TIME_LABEL),
                              ('bid', BID_PRICE_LABEL),
                              ('ask', ASK_PRICE_LABEL),
                              ('bid_size', BID_SIZE_LABEL),
                              ('ask_size', ASK_SIZE_LABEL)]
PAYOFF_CODES = {'C': CALL_CODE, 'P': PUT_CODE}


def select_rename_labels(data, label_pairs):
    labels_old, labels_new = map(list, zip(*label_pairs))
    data = data[labels_old]
    data.columns = labels_new
    return data


def format_data(meta, quotes, index_labels):
    meta[EXPIRY_LABEL] = (
        meta[EXPIRY_LABEL].astype('datetime64[ns]', copy=False))
    meta[PAYOFF_LABEL] = meta[PAYOFF_LABEL].map(lambda p: PAYOFF_CODES[p])

    quotes = pd.merge(meta[[JOIN_LABEL, *index_labels]], quotes, on=JOIN_LABEL)
    join_values = quotes[JOIN_LABEL].unique()
    quotes.drop(columns=JOIN_LABEL, inplace=True)
    quotes[TIME_LABEL] = (
        quotes[TIME_LABEL].astype('datetime64[ns]', copy=False))
    quotes.set_index([*index_labels, TIME_LABEL], inplace=True)
    quotes[quotes == 0] = np.nan
    quotes.sort_index(inplace=True)

    meta.set_index(JOIN_LABEL, inplace=True)
    meta = meta.loc[join_values]
    meta.reset_index(drop=True, inplace=True)
    meta.set_index(index_labels, inplace=True)
    meta[TIME_TO_EXPIRY_LABEL] /= BUSINESS_DAYS_IN_YEAR
    meta[PUT_FLAG_LABEL] = get_col(meta, PAYOFF_LABEL) == PUT_CODE
    meta.sort_index(inplace=True)
    return quotes, meta


def parse_futures(data):
    meta = select_rename_labels(data[FUTURES_META_KEY],
                                FUTURES_META_LABEL_PAIRS)
    quotes = select_rename_labels(data[FUTURES_QUOTES_KEY],
                                  FUTURES_QUOTES_LABEL_PAIRS)
    return format_data(meta, quotes, FUTURES_INDEX_LABEL)


def parse_options(data):
    meta = select_rename_labels(data[OPTIONS_META_KEY],
                                OPTIONS_META_LABEL_PAIRS)
    quotes = select_rename_labels(data[OPTIONS_QUOTES_KEY],
                                  OPTIONS_QUOTES_LABEL_PAIRS)
    return format_data(meta, quotes, OPTIONS_INDEX_LABEL)


def get_rate(data):
    return data[OPTIONS_META_KEY].prev_rate.median()


def get_gridded_prices(quotes):
    gridded = quotes[[BID_PRICE_LABEL, ASK_PRICE_LABEL]]
    gridded = gridded.groupby([PAYOFF_LABEL, EXPIRY_LABEL, STRIKE_LABEL]
        ).apply(lambda o: o[is_consecutive_unique(o.values)])
    grid = np.arange(START_TIME, END_TIME + TIME_STEP, TIME_STEP)
    gridded = gridded.groupby([PAYOFF_LABEL, EXPIRY_LABEL, STRIKE_LABEL]
                              ).apply(lambda o: resample(o.xs(o.name), grid))
    gridded[MID_PRICE_LABEL] = gridded.mean(axis=1)

    return gridded


def pair_by_payoff(options):
    paired = options.unstack(PAYOFF_LABEL)
    paired.dropna(inplace=True)
    return paired


def bootstrap(quotes, meta, rate):
    gridded_prices = get_gridded_prices(quotes)
    paired_prices = pair_by_payoff(gridded_prices[MID_PRICE_LABEL])

    first_expiry = np.unique(get_col(paired_prices, EXPIRY_LABEL))[0]
    first_expiry_paired_prices = paired_prices.xs(first_expiry,
                                                  level=EXPIRY_LABEL)
    first_expiry_discount = (1 + rate)**meta[TIME_TO_EXPIRY_LABEL].xs(
        first_expiry, level=EXPIRY_LABEL).iloc[0]
    kwargs = cols_to_kwargs(first_expiry_paired_prices, call=CALL_CODE,
                            put=PUT_CODE, strike=STRIKE_LABEL)
    underlying_prices = put_call_parity(discount=first_expiry_discount,
                                        **kwargs)
    underlying_prices = pd.Series(underlying_prices,
        first_expiry_paired_prices.index, name=UNDERLYING_MID_PRICE_LABEL)
    underlying_prices = underlying_prices.groupby(TIME_LABEL).median()

    discounts = paired_prices.groupby([EXPIRY_LABEL, STRIKE_LABEL]).apply(
        lambda p: put_call_parity(underlying=underlying_prices,
            strike=p.name[1], **cols_to_kwargs(p.xs(p.name), call=CALL_CODE,
                                               put=PUT_CODE)))
    discount_curve = discounts.stack(TIME_LABEL).groupby(EXPIRY_LABEL).median()

    meta[DISCOUNTED_STRIKE_LABEL] = (get_col(meta, STRIKE_LABEL)
        *discount_curve.reindex(get_col(meta, EXPIRY_LABEL))).values

    return gridded_prices, underlying_prices, discounts


def get_discount_quantiles(discounts):
    quantiles = pd.Index([0.025, 0.5, 0.0975], name='Quantile')
    discounts_quantiles = discounts.groupby(EXPIRY_LABEL).apply(
        lambda e: pd.Series(e.quantile(quantiles), quantiles))
    return discounts_quantiles.unstack('Quantile')


def get_implied_vols(prices, underlying_prices, meta):
    kwargs = cols_to_kwargs(meta, strike=DISCOUNTED_STRIKE_LABEL,
                            expiry=TIME_TO_EXPIRY_LABEL, put=PUT_FLAG_LABEL)
    underlying_prices = underlying_prices.reindex(prices.columns)
    return prices.apply(lambda o: blackscholes.implied_vol(
        underlying_prices.xs(o.name), option_price=o,
        assert_no_arbitrage=False, **kwargs))


def filter_liquid(underlying_price, options_prices):
    expiries = np.unique(get_col(options_prices, EXPIRY_LABEL))[1:4]
    filtered = options_prices.reindex(expiries, level=EXPIRY_LABEL)
    strikes = get_col(filtered, STRIKE_LABEL)
    put = get_col(filtered, PAYOFF_LABEL) == PUT_CODE
    filtered = filtered[((0.8*underlying_price <= strikes) & (strikes <= underlying_price) & put) | ((underlying_price <= strikes) &
 (strikes <= 1.2*underlying_price) & ~put)]
    return filtered

if __name__ == '__main__':
    data = pd.HDFStore('test.h5')
    rate = get_rate(data)
    quotes, meta = parse_options(data)
    gridded_prices, underlying_prices, discounts = bootstrap(quotes, meta, rate)
    filtered = filter_liquid(underlying_prices.xs(CALIBRATION_TIME),
                             gridded_prices.xs(CALIBRATION_TIME,
                                               level=TIME_LABEL))

    kwargs = cols_to_kwargs(
        meta.loc[filtered.index], strikes=DISCOUNTED_STRIKE_LABEL,
        expiries=TIME_TO_EXPIRY_LABEL, put=PUT_FLAG_LABEL)
    params = heston_app(underlying_prices.xs(CALIBRATION_TIME),
                        option_prices=filtered[MID_PRICE_LABEL], **kwargs)
