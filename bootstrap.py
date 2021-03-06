import pandas as pd
import numpy as np
import statsmodels.api as sm
import xarray as xr


INDEX_DTYPE = np.dtype(
    [
        ('expiry', 'M8[D]'),
        ('strike', 'i8'),
        ('payoff', 'U1'),
        ('time', 'm8[ns]'),
    ]
)

def format_gridded_quotes(tick_quotes, time_granularity, start_time):
    index_coarse = tick_quotes['index'].copy()
    index_coarse['time'] = np.ceil(
        index_coarse['time'] / time_granularity
    ) * time_granularity
    mask_last = np.ones(len(index_coarse), dtype=np.bool_)
    mask_last[:-1] = index_coarse[1:] != index_coarse[:-1]
    n_rows = np.sum(mask_last)
    gridded_quotes_dtype = np.dtype(
        [
            ('index', INDEX_DTYPE, n_rows),
            ('bid', 'f8', n_rows),
            ('ask', 'f8', n_rows),
            ('date', 'M8[D]'),
            ('lazymaker_hash', 'U40'),
        ]
    )
    gridded_quotes_arr = np.array(
        (
            index_coarse[mask_last],
            tick_quotes['bid'][mask_last],
            tick_quotes['ask'][mask_last],
            tick_quotes['date'],
            tick_quotes['lazymaker_hash']
        ),
        dtype=gridded_quotes_dtype,
    )
    variables = dict(
        bid=('index', gridded_quotes_arr['bid']),
        ask=('index', gridded_quotes_arr['ask'])
    )
    option_specs = gridded_quotes_arr['index'][['expiry', 'strike', 'payoff']]
    uniq_mask = np.ones(len(option_specs), dtype=np.bool_)
    uniq_mask[1:] = option_specs[1:] != option_specs[:-1]
    option_ids = np.cumsum(uniq_mask)
    uniq_specs = option_specs[uniq_mask]
    coords = dict(
        option_id=('index', option_ids),
        time=('index', gridded_quotes_arr['index']['time']),
        date=((), gridded_quotes_arr['date']),
    )
    expiries = uniq_specs['expiry']
    date = tick_quotes['date'].astype('M8[D]')
    years_to_expiry = np.busday_count(date, expiries) / 252
    gridded_quotes = (
        xr.Dataset(variables, coords)
        .set_index(index=['option_id', 'time'])
        .unstack('index')
        .ffill('time')
        .sel(time=slice(start_time, None))
        .reset_index('option_id', drop=True)
        .assign_coords(
            expiry=('option_id', uniq_specs['expiry']),
            strike=('option_id', uniq_specs['strike']),
            payoff=('option_id', uniq_specs['payoff']),
            years_to_expiry=('option_id', years_to_expiry),
        )
    )
    return gridded_quotes


def pca_on_midprice(prices):
    mid_prices = prices.set_index({'option_id': ['expiry', 'strike']}).mid
    mid_changes = mid_prices.diff('time').fillna(0)
    mid_z = (mid_changes - mid_changes.mean('time'))/mid_changes.std('time')
    corr_mat = np.cov(mid_z.fillna(0))
    eigvals, eigvecs = np.linalg.eigh(corr_mat)
    eigvals, eigvecs = eigvals[::-1], eigvecs[:, ::-1]
    explained = pd.Index(eigvals/np.sum(eigvals), name='explained')
    weights = xr.DataArray(eigvecs, coords=[mid_prices.option_id, explained])
    weights.name = 'weights'

    return weights.to_dataset().reset_index('option_id')


def recover_underlying(parity, weights):
    mid_prices = parity.set_index({'option_id': ['expiry', 'strike']}).mid
    weights = weights.set_index({'option_id': ['expiry', 'strike']}).weights
    weights = weights.isel(explained=0).to_series()
    total_weight = np.sum(weights.values)

    def aggregate_mids(mids):
        mask = ~np.isnan(mids)
        slice_weights = weights[mask]
        slice_total_weight = np.sum(slice_weights.values)
        return (
            mids[mask].dot(slice_weights) / slice_total_weight
            if slice_total_weight / total_weight > 0.95
            else np.nan
        )

    mid_std = mid_prices.diff('time').std('time')
    mid_normalised = (mid_prices - mid_prices.mean('time'))/mid_std
    reconstructed = mid_normalised.to_pandas().apply(aggregate_mids)
    reconstructed.name = 'underlying'
    return reconstructed.to_xarray().to_dataset()


def pair_puts_and_calls(quotes):
    quotes = quotes.set_index(option_id=['expiry', 'strike'])
    calls = quotes.sel(option_id=quotes.payoff == 'C')
    puts = quotes.sel(option_id=quotes.payoff == 'P')
    parity = {'bid': calls.bid - puts.ask, 'ask': calls.ask - puts.bid}
    parity = xr.Dataset(parity).dropna('option_id', how='all')
    parity['mid'] = (parity.bid + parity.ask) / 2
    return parity.reset_index('option_id')


def forward_bond_regression_map(parity, underlying, expiry):
    parity = parity.set_index(option_id='strike').rename(option_id='strike')
    parity['half_spread'] = (parity['ask'] - parity['bid']) / 2
    reg_data = parity.reset_coords(drop=True
                    ).assign_coords({'underlying': underlying}
                    ).to_dataframe().dropna().reset_index('strike')

    if len(reg_data) < 2:
        return np.nan*underlying.assign_coords(bond=np.nan, max_spread=np.nan,
                                               in_spread=np.nan)

    reg_data = sm.add_constant(reg_data)
    reg_data = reg_data[['const', 'strike', 'mid', 'underlying']
                        ]/reg_data.half_spread.values[:, None]
    fit = sm.OLS(reg_data['mid'], reg_data.drop('mid', axis=1)).fit()
    forward = fit.params['const'] + fit.params['underlying']*underlying
    bond = -fit.params['strike']

    synthetic_forwards = xr.concat([forward - bond*strike
                                    for strike in parity.strike], parity.strike)
    relative_spreads = (synthetic_forwards - parity.mid)/parity.half_spread
    max_spread = relative_spreads.max().data
    in_spread = (relative_spreads <= 1).sum().data/relative_spreads.count().data
    forward = forward.where(~(relative_spreads > 1).any('strike'))
    coords = dict(
        bond=bond,
        expiry=expiry,
        max_spread=max_spread,
        in_spread=in_spread,
    )
    return forward.assign_coords(coords)


def forward_bond_regression(parity, underlying):
    forwards_bonds_lst = [
        forward_bond_regression_map(
            parity.sel(option_id=parity.expiry == expiry),
            underlying.underlying,
            expiry
        ) for expiry in np.unique(parity.expiry)
    ]
    forwards_bonds = xr.concat(forwards_bonds_lst, 'expiry')
    forwards_bonds.name = 'forward'
    return forwards_bonds.to_dataset().reset_coords('bond')
