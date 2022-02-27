import numpy as np
import pandas as pd


def get_summary_stats_for_date(tick_quotes: np.ndarray) -> pd.DataFrame:
    specs = tick_quotes['index'][['expiry', 'strike', 'payoff']]
    specs_uniq = np.concatenate([specs[:1], specs[1:][specs[1:] != specs[:-1]]])
    np.testing.assert_equal(specs_uniq, np.unique(specs_uniq))

    expiries, n_options = np.unique(specs_uniq['expiry'], return_counts=True)
    expiries_uniq, n_rows = np.unique(specs['expiry'], return_counts=True)
    expiries_fmt = pd.to_datetime(expiries).strftime('%Y-%m-%d')
    np.testing.assert_array_equal(expiries, expiries_uniq)
    np.testing.assert_equal(np.sum(n_rows), len(tick_quotes['index']))

    summary = pd.DataFrame(
        dict(expiry=expiries_fmt, n_options=n_options, n_rows=n_rows)
    ).set_index('expiry')
    summary.loc['total'] = summary.sum(axis=0)
    summary['date'] = tick_quotes['date']

    return summary


def get_summary_stats(*tick_quotes_lst: np.ndarray):
    summary = pd.concat(
        [get_summary_stats_for_date(tick_quotes) for tick_quotes in tick_quotes_lst]
    )
    return summary.groupby('expiry').mean()
