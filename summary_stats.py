import numpy as np
import pandas as pd
import xarray as xr


def get_summary_stats_for_date(tick_quotes: np.ndarray) -> xr.Dataset:
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
            n_options=('expiry', [*n_options, np.sum(n_options)]),
            n_rows=('expiry', [*n_rows, np.sum(n_rows)]),
        ),
        dict(
            expiry=('expiry', [*expiries_fmt, 'total']),
            date=tick_quotes['date'],
        )
    )


def get_summary_stats(*tick_quotes_lst: np.ndarray) -> pd.DataFrame:
    summaries = xr.concat(
        [get_summary_stats_for_date(tick_quotes) for tick_quotes in tick_quotes_lst],
        'date',
        fill_value=0,
    )
    return xr.Dataset(
        dict(
            n_options_min=summaries.n_options.min('date'),
            n_options_median=summaries.n_options.median('date'),
            n_options_mean=summaries.n_options.mean('date'),
            n_options_max=summaries.n_options.max('date'),
            n_rows_min=summaries.n_rows.min('date'),
            n_rows_median=summaries.n_rows.median('date'),
            n_rows_mean=summaries.n_rows.mean('date'),
            n_rows_max=summaries.n_rows.max('date'),
        ),
    ).to_dataframe()


def tabulate_summary_stats(*tick_quotes_lst: np.ndarray):
    return (
        get_summary_stats(*tick_quotes_lst)
        .rename(
            columns=dict(
                n_options_min="Active options\n(min)",
                n_options_median="Active options\n(median)",
                n_options_mean="Active options\n(mean)",
                n_options_max="Active options\n(max)",
                n_rows_min="Mid-price changes\n(min)",
                n_rows_median="Mid-price changes\n(median)",
                n_rows_mean="Mid-price changes\n(mean)",
                n_rows_max="Mid-price changes\n(max)",
            )
        )
        .to_latex()
    )
