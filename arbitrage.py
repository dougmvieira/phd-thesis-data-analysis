import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import cm
from matplotlib.ticker import PercentFormatter
from scipy.optimize import linprog

from utils import A4_HEIGHT, A4_WIDTH


def compute_no_arb_bounds_map(parity_bid, parity_ask, strikes):
    bid_mask, ask_mask = ~np.isnan(parity_bid), ~np.isnan(parity_ask)
    parity_bid, parity_ask = parity_bid[bid_mask], parity_ask[ask_mask]
    strikes_bid, strikes_ask = map(np.array, [strikes[bid_mask],
                                              strikes[ask_mask]])

    b_ub = np.concatenate((parity_ask, -parity_bid))
    A_ub = np.block([[ np.ones((len(parity_ask), 1)), -strikes_ask[:, None]],
                     [-np.ones((len(parity_bid), 1)),  strikes_bid[:, None]]])
    bid = linprog(np.array([ 1, 0]), A_ub, b_ub).x
    ask = linprog(np.array([-1, 0]), A_ub, b_ub).x

    coords = [('side', ['ask', 'bid']), ('asset', ['forward', 'bond'])]
    return xr.DataArray([bid, ask], coords)


def compute_no_arb_bounds(parity, time):
    parity_slice = parity.sel(time=time)
    no_arb_bounds = parity_slice.groupby('expiry', restore_coord_dims=True).map(
        lambda e: compute_no_arb_bounds_map(e.bid, e.ask, e.strike))
    return no_arb_bounds.to_dataset(name='bounds')


def plot_parity_bounds(no_arb_bounds, parity, time, n_expiries):
    no_arb_bounds = no_arb_bounds.bounds
    parity = parity.set_index({'option_id': ['expiry', 'strike']})
    expiries = np.unique(parity.expiry)

    parity_slice = parity.sel(time=time)
    strike_max = np.max(parity_slice.strike.values)
    parity_bid = parity_slice.bid.reset_coords(drop=True
                            ).to_series().dropna().unstack('expiry')
    parity_ask = parity_slice.ask.reset_coords(drop=True
                            ).to_series().dropna().unstack('expiry')
    parity_bid += parity_bid.index.values[:, None]
    parity_ask += parity_ask.index.values[:, None]

    forward_bounds = no_arb_bounds.sel(asset='forward').to_dataset('side'
                                 ).reset_coords(drop=True).to_dataframe()
    bond_bounds = no_arb_bounds.sel(asset='bond').to_dataset('side'
                              ).reset_coords(drop=True).to_dataframe()
    parity_bounds = pd.concat([forward_bounds,
                               forward_bounds - (bond_bounds - 1)*strike_max],
                              keys=[0, strike_max], names=['strike'])

    colours = cm.coolwarm(np.linspace(0., 1., len(expiries)))

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(A4_WIDTH, A4_HEIGHT))
    for ax in axes:
        parity_bid.plot(marker='^', linewidth=0, ax=ax, color=colours,
                        markersize=3)
        parity_ask.plot(marker='v', linewidth=0, ax=ax, color=colours,
                        markersize=3, legend=False)
        ylim = ax.get_ylim()
        for (_, g), c in zip(parity_bounds.groupby('expiry'), colours):
            g = g.droplevel('expiry')
            ax.fill_between(g.index, g.bid, g.ask, color=c, alpha=.2)
        ax.set_ylim(ylim)
        ax.set_ylabel(r'price minus strike')

    axes[0].get_legend().remove()
    axes[0].set_ylim((parity_bid[expiries[n_expiries - 1]].min(), None))

    return fig


def plot_forward_bounds(no_arb_bounds):
    no_arb_bounds = no_arb_bounds.bounds
    forward_bounds = no_arb_bounds.sel(asset='forward').to_dataset('side'
                                 ).reset_coords(drop=True).to_dataframe()

    fig, ax = plt.subplots(figsize=(A4_WIDTH, A4_HEIGHT/2))
    ax.fill_between(forward_bounds.index, forward_bounds.bid,
                    forward_bounds.ask, step='mid')
    ax.set_xlabel('expiry')
    ax.set_xlabel('forward price')
    return fig


def plot_no_arb_yield_curve(no_arb_bounds, parity):
    no_arb_bounds = no_arb_bounds.bounds
    parity = parity.set_index({'option_id': ['expiry', 'strike']})

    bond_bounds = no_arb_bounds.sel(asset='bond').to_dataset('side'
                              ).reset_coords(drop=True).to_dataframe()
    years_to_expiry = parity.years_to_expiry.groupby('expiry').first(
                           ).to_series()
    yield_curve = bond_bounds.apply(lambda side: side**(1/years_to_expiry) - 1)

    fig, ax = plt.subplots(figsize=(A4_WIDTH, A4_HEIGHT/2))
    ax.fill_between(yield_curve.index, 100*yield_curve.bid,
                    100*yield_curve.ask, step='mid')
    ax.set_xlabel('expiry')
    ax.set_xlabel('interest rate')
    ax.yaxis.set_major_formatter(PercentFormatter())

    return fig


def plot_no_arb_dividend_curve(no_arb_bounds, parity):
    no_arb_bounds = no_arb_bounds.bounds
    parity = parity.set_index({'option_id': ['expiry', 'strike']})

    forward_bounds = no_arb_bounds.sel(asset='forward').to_dataset('side'
                                 ).reset_coords(drop=True).to_dataframe()
    acc_dividends = forward_bounds.iloc[1:]/forward_bounds.iloc[0, ::-1].values
    years_to_expiry = parity.years_to_expiry.groupby('expiry').first(
                           ).to_series()
    years_to_expiry = years_to_expiry.iloc[1:] - years_to_expiry.iloc[0]
    dividend_curve = acc_dividends.apply(
        lambda side: side**(-1/years_to_expiry) - 1)

    fig, ax = plt.subplots(figsize=(A4_WIDTH, A4_HEIGHT/2))
    ax.fill_between(dividend_curve.index, 100*dividend_curve.bid,
                    100*dividend_curve.ask, step='mid')
    ax.set_xlabel('expiry')
    ax.set_xlabel('dividend yield')
    ax.yaxis.set_major_formatter(PercentFormatter())

    return fig
