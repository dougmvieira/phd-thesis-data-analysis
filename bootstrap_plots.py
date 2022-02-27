import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from fyne import blackscholes, heston
from matplotlib.ticker import PercentFormatter
from matplotlib.dates import DateFormatter

from numpy_tricks import aggregate, groupby
from utils import A4_HEIGHT, A4_WIDTH


def plot_pca(weights):
    weights = weights.set_index({'option_id': ['expiry', 'strike']}).weights
    explained_variance = pd.Series(weights.explained[:10],
                                   pd.Index(range(1, 11), name='component'))
    weights = weights.isel(explained=0)
    rel_weights = (weights/weights.sum()).to_series().unstack('expiry')
    rel_weights.columns = pd.Index(rel_weights.columns.date, name='expiry')

    fig, axes = plt.subplots(2, 1, figsize=(A4_WIDTH, A4_HEIGHT))
    rel_weights.plot(marker='o', linewidth=0, cmap='jet', ax=axes[0])
    explained_variance.plot(kind='bar', ax=axes[1])
    axes[0].set_ylabel('relative weights')
    axes[1].set_ylabel('explained variance ratio')

    return fig


def compare_with_itm(underlying, quotes, strike):
    underlying = underlying.underlying

    is_strike = quotes.strike == strike
    is_expiry = quotes.expiry == quotes.expiry.isel(option_id=0)
    is_payoff = quotes.payoff == 'C'
    itm_quotes = quotes.sel(option_id=is_strike & is_expiry & is_payoff)
    itm_quotes = itm_quotes.isel(option_id=0)
    itm_mid = (itm_quotes.bid + itm_quotes.ask)/2
    itm_std = itm_mid.diff('time').std()
    itm_normalised = (itm_mid - itm_mid.mean('time'))/itm_std

    normalised_prices = pd.concat(
        [underlying.to_series(), itm_normalised.to_series()],
        keys=['underlying', 'itm_option'], axis=1)
    normalised_prices.index += quotes.date.values

    fig, ax = plt.subplots(figsize=(A4_WIDTH, A4_HEIGHT/2))
    normalised_prices.plot(ax=ax)
    ax.set_ylabel('standardised price')
    ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))

    return fig


def plot_forwards(forwards_bonds):
    fig, ax = plt.subplots(figsize=(A4_WIDTH, A4_HEIGHT/2))
    forwards = forwards_bonds.forward.to_series().unstack('expiry')
    forwards.index += forwards_bonds.date.values
    forwards.columns = pd.Index(forwards.columns.date, name='expiry')
    forwards.plot(ax=ax)
    ax.set_ylabel('forward price')
    ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    return fig


def plot_interest_rates(parity, forwards_bonds):
    years_to_expiry = parity.years_to_expiry.groupby('expiry').first()
    interest_rates = forwards_bonds.bond**(-1/years_to_expiry) - 1

    fig, ax = plt.subplots(figsize=(A4_WIDTH, A4_HEIGHT/2))
    (100*interest_rates).to_series().plot(ax=ax)
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.set_ylabel('Interest rate')

    return fig


def tabulate_bootstrap_spreads(forwards_bonds):
    spreads = forwards_bonds.reset_coords(['max_spread', 'in_spread'])
    spreads = spreads[['max_spread', 'in_spread']].reset_coords(drop=True)
    spreads = spreads.rename_vars({'max_spread': 'Maximum spread',
                                   'in_spread': 'Estimates within spreads'})
    spreads = spreads.to_dataframe().applymap(lambda x: '{:.3%}'.format(x))
    return spreads.to_latex()


def compute_heston_ivs_slice(ivs, forwards_bonds, heston_params):
    min_strike, max_strike = ivs.strike.min(), ivs.strike.max()
    strikes = np.linspace(min_strike, max_strike)
    bond = forwards_bonds.bond.data
    forward = forwards_bonds.forward.data
    expiry = ivs.years_to_expiry.isel(moneyness=0).data
    vol, kappa = heston_params.vol, heston_params.kappa
    theta, nu, rho = heston_params.theta, heston_params.nu, heston_params.rho
    heston_calls = heston.formula(forward, bond*strikes, expiry, vol, kappa,
                                  theta, nu, rho)
    heston_ivs = blackscholes.implied_vol(forward, bond*strikes, expiry,
                                          heston_calls)
    moneyness = np.log(strikes / forward) / np.sqrt(expiry)
    return pd.Series(heston_ivs, moneyness)


def heston_calibration_plot(ivs, forwards_bonds, heston_params, time):
    expiries = np.unique(ivs.expiry)[2:5]
    ivs_slice = ivs.sel(time=time, option_id=ivs.expiry.isin(expiries))
    forwards = forwards_bonds.forward.sel(time=time, expiry=ivs_slice.expiry)
    moneyness = (
        np.log(ivs_slice.discounted_strike / forwards)
        / np.sqrt(ivs_slice.years_to_expiry)
    )
    otm_mask = (
        ((ivs_slice.payoff.values == 'C') & (moneyness.values > 0))
        | ((ivs_slice.payoff.values == 'P') & (moneyness.values < 0))
    )
    moneyness_mask = (-0.5 <= moneyness.values) & (moneyness.values <= 0.5)
    ivs_slice = (
        ivs_slice
        .assign_coords(moneyness=moneyness)
        .sel(option_id=otm_mask & moneyness_mask)
        .set_index(option_id=['expiry', 'moneyness'])
    )

    heston_params_slice = heston_params.sel(time=time)
    forwards_bonds_slice = forwards_bonds.sel(time=time)

    fig, axes = plt.subplots(3, 1, figsize=(A4_WIDTH, A4_HEIGHT), sharex=True)
    for e, ax in zip(np.unique(ivs_slice.expiry), axes):
        compute_heston_ivs_slice(ivs_slice.sel(expiry=e),
                                 forwards_bonds_slice.sel(expiry=e),
                                 heston_params_slice).plot(ax=ax)
        ivs_slice.ask.sel(expiry=e).to_series(
            ).plot(marker='v', linewidth=0, c='tab:orange', ax=ax)
        ivs_slice.bid.sel(expiry=e).to_series(
            ).plot(marker='^', linewidth=0, c='tab:orange', ax=ax)
        ax.set_title('expiry: {}'.format(np.datetime_as_string(e, unit='D')))
        ax.set_ylabel('implied volatility')
        ax.set_xlim(-0.5, 0.5)
        fig.tight_layout()

    return fig


def heston_calibration_table(heston_params, time):
    params_names = dict(vol='$V_0$', kappa=r'$\kappa$', theta=r'$\theta$',
                        nu=r'$\nu$', rho=r'$\rho$')
    params_table = heston_params.reset_coords('date', drop=True)
    params_table = params_table.rename(params_names).to_dataframe().loc[time]
    return params_table.to_latex(escape=False, header=False)
