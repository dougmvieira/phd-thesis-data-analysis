import numpy as np
import pandas as pd
import xarray as xr


def get_summary_stats_for_date(tick_quotes: np.ndarray) -> pd.DataFrame:
    specs = tick_quotes['index'][['expiry', 'strike', 'payoff']]
    specs_uniq = np.concatenate([specs[:1], specs[1:][specs[1:] != specs[:-1]]])
    np.testing.assert_equal(specs_uniq, np.unique(specs_uniq))

    expiries, n_options = np.unique(specs_uniq['expiry'], return_counts=True)
    expiries_uniq, n_rows = np.unique(specs['expiry'], return_counts=True)
    expiries_fmt = pd.to_datetime(expiries).strftime('%Y-%m-%d')
    np.testing.assert_array_equal(expiries, expiries_uniq)
    np.testing.assert_equal(np.sum(n_rows), len(tick_quotes['index']))

    return xr.Dataset(
        dict(
            n_options=('expiry', n_options),
            n_rows=('expiry', n_rows),
        ),
        dict(
            expiry=('expiry', expiries_fmt),
            date=tick_quotes['date'],
        )
    )


def get_summary_stats(*tick_quotes_lst: np.ndarray):
    summaries = xr.concat(
        [get_summary_stats_for_date(tick_quotes) for tick_quotes in tick_quotes_lst],
        'date',
        fill_value=0,
    )
    return xr.Dataset(
        dict(
            n_options_1st_quartile=summaries.n_options.quantile(0.25, 'date'),
            n_options_median=summaries.n_options.median('date'),
            n_options_3rd_quartile=summaries.n_options.quantile(0.25, 'date'),
            n_rows_1st_quartile=summaries.n_rows.quantile(0.25, 'date'),
            n_rows_median=summaries.n_rows.median('date'),
            n_rows_3rd_quartile=summaries.n_rows.quantile(0.25, 'date'),
        ),
    ).drop('quantile').to_dataframe()
