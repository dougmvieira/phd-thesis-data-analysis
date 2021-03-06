import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from fyne import heston
from matplotlib.dates import DateFormatter

from utils import A4_HEIGHT, A4_WIDTH


def heston_volatility_plot(heston_params, sample_stds):
    fig, ax = plt.subplots(figsize=(A4_WIDTH, A4_HEIGHT/2))
    if np.all(heston_params.vol.isnull().data):
        return fig

    heston_time = heston_params.date + heston_params.time
    sample_time = heston_params.date + sample_stds.time
    heston_vol = np.sqrt(heston_params.vol)
    ax.plot(heston_time, heston_vol, color='tab:blue', label='heston')
    ax.fill_between(
        sample_time,
        sample_stds.lower,
        sample_stds.upper,
        color='tab:orange',
        alpha=0.3,
        step='mid',
    )
    ax.plot(
        sample_time,
        sample_stds.estimate,
        color='tab:orange',
        label='sample',
        drawstyle='steps-mid',
    )
    ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    ax.set_title('')
    ax.legend()
    ax.set_ylabel('volatility')
    return fig


def heston_volatility_multiday_plot(heston_params_lst, sample_stds_lst):
    fig, axes = plt.subplots(
        5, 4, sharex=True, sharey=True, figsize=(A4_WIDTH, A4_HEIGHT)
    )

    for heston_params, sample_stds, ax in zip(
        heston_params_lst, sample_stds_lst, axes.ravel()
    ):
        heston_time = np.datetime64(0, 'ns') + heston_params.time
        sample_time = np.datetime64(0, 'ns') + sample_stds.time
        heston_vol = np.sqrt(heston_params.vol)
        ax.plot(heston_time, heston_vol, color='tab:blue', label='heston')
        ax.fill_between(
            sample_time,
            sample_stds.lower,
            sample_stds.upper,
            color='tab:orange',
            alpha=0.3,
            step='mid',
        )
        ax.plot(
            sample_time,
            sample_stds.estimate,
            color='tab:orange',
            label='sample',
            drawstyle='steps-mid',
        )
        date_str = np.datetime_as_string(heston_params.date.values, unit='D')
        ax.set_title(date_str)
    axes[0, -1].legend()
    for ax in axes[-1, :]:
        ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        for tick in ax.get_xticklabels():
            tick.set_rotation(90)
    for ax in axes[:, 0]:
        ax.set_ylabel('volatility')
    fig.subplots_adjust(wspace=0)
    fig.tight_layout()
    return fig


def heston_daily_volatility_plot(heston_params, daily_stds):
    heston_params = [hp for hp in heston_params
                     if not np.all(hp.vol.isnull().data)]
    heston_params = xr.combine_nested(heston_params, concat_dim='date')
    heston_vols = np.sqrt(heston_params.vol.mean('time'))

    fig, ax = plt.subplots(figsize=(A4_WIDTH, A4_HEIGHT/2))
    heston_vols.to_series().plot(ax=ax, label='heston', color='tab:blue')
    daily_stds.estimate.to_series().plot(label='sample', ax=ax,
                                         color='tab:orange')
    ax.fill_between(daily_stds.date.data, daily_stds.lower.data,
                    daily_stds.upper.data, color='tab:orange', alpha=0.3)

    ax.legend()
    ax.set_ylabel('volatility')
    plt.tight_layout()
    return fig


def greeks_plot(greeks, heston_params, forwards_bonds, greek_name, payoff):
    fig, axes = plt.subplots(3, 1, figsize=(A4_WIDTH, A4_HEIGHT))
    if np.all(heston_params.vol.isnull().data):
        return fig

    forwards = forwards_bonds.forward.sel(expiry=greeks.expiry).mean('time')
    moneyness = (
        np.log(greeks.discounted_strike / forwards)
        / np.sqrt(greeks.years_to_expiry)
    )
    greeks = (
        greeks[greek_name]
        .assign_coords(moneyness=moneyness)
        .set_index({'option_id': ['payoff', 'expiry', 'moneyness']})
        .sel(payoff=payoff)
    )
    greeks = greeks.sel(
        option_id=(-0.5 <= greeks.moneyness) & (greeks.moneyness <= 0.5)
    )
    expiries = np.unique(greeks.expiry)
    for e, ax in zip(expiries[2:5], axes):
        forwards_slice = forwards_bonds.forward.sel(expiry=e)
        greeks_slice = greeks.sel(option_id=greeks.expiry == e).sel(expiry=e)

        greeks_emp = greeks_slice.reset_coords(drop=True)
        greeks_emp = greeks_emp.to_series().unstack('confidence')
        greeks_emp.columns = ['{:.1%}'.format(p) for p in greeks_emp.columns]

        underlying = forwards_slice.mean('time').data
        strike = greeks_slice.discounted_strike.data
        expiry = greeks_slice.years_to_expiry.data
        vol = heston_params.vol.mean('time').data
        kappa = heston_params.kappa.data
        theta = heston_params.theta.data
        nu = heston_params.nu.data
        rho = heston_params.rho.data
        if greek_name == 'delta':
            greek_heston = heston.delta(underlying, strike, expiry, vol, kappa,
                                        theta, nu, rho, put=payoff=='P')
        elif greek_name == 'vega':
            greek_heston = heston.vega(underlying, strike, expiry, vol, kappa,
                                       theta, nu, rho)
        else:
            greek_heston = 0.
        greek_heston = pd.Series(greek_heston, greeks_emp.index,
                                 name='hypothetical')

        greeks_emp_ub = greeks_emp.iloc[:, 1]
        greeks_emp_lb = greeks_emp.iloc[:, 0]
        hi_qtile = greeks_emp_ub.quantile(0.9)
        lo_qtile = greeks_emp_lb.quantile(0.1)
        centre = (hi_qtile + lo_qtile) / 2
        ub = centre + 2 * (hi_qtile - centre)
        lb = centre + 2 * (lo_qtile - centre)
        greeks_emp_ub.plot(marker='v', linewidth=0, ax=ax)
        greeks_emp_lb.plot(marker='^', linewidth=0, ax=ax)
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(lb, ub)
        greek_heston.plot(ax=ax)
        ax.set_title('expiry: {}'.format(np.datetime_as_string(e, unit='D')))
        ax.set_ylabel(f'Heston {greek_name}')
        ax.legend()

    fig.tight_layout()
    return fig


def greeks_r2_plot(greeks, forwards_bonds):
    fig, axes = plt.subplots(3, 1, figsize=(A4_WIDTH, A4_HEIGHT))
    try:
        greeks = greeks['delta']
    except KeyError:
        return fig

    forwards = forwards_bonds.forward.sel(expiry=greeks.expiry).mean('time')
    moneyness = (
        np.log(greeks.discounted_strike / forwards)
        / np.sqrt(greeks.years_to_expiry)
    )
    greeks = (
        greeks
        .assign_coords(moneyness=moneyness)
        .set_index({'option_id': ['payoff', 'expiry', 'moneyness']})
    )
    greeks = greeks.sel(
        option_id=(-0.5 <= greeks.moneyness) & (greeks.moneyness <= 0.5)
    )
    expiries = np.unique(greeks.expiry)
    for e, ax in zip(expiries[2:5], axes):
        r2 = greeks.r2.sel(expiry=e).to_series().unstack('payoff')
        r2 = r2.rename(columns=dict(C='call', P='put'))
        r2.plot(ax=ax, marker='o', linewidth=0)

        ax.set_title('expiry: {}'.format(np.datetime_as_string(e, unit='D')))
        ax.set_ylabel('$R^2$')
        ax.legend()

    fig.tight_layout()
    return fig


def greeks_sr2_plot(greeks, forwards_bonds):
    fig, axes = plt.subplots(3, 1, figsize=(A4_WIDTH, A4_HEIGHT))
    try:
        greeks = greeks['delta']
    except KeyError:
        return fig

    forwards = forwards_bonds.forward.sel(expiry=greeks.expiry).mean('time')
    moneyness = (
        np.log(greeks.discounted_strike / forwards)
        / np.sqrt(greeks.years_to_expiry)
    )
    greeks = (
        greeks
        .assign_coords(moneyness=moneyness)
        .set_index({'option_id': ['payoff', 'expiry', 'moneyness']})
    )
    greeks = greeks.sel(
        option_id=(-0.5 <= greeks.moneyness) & (greeks.moneyness <= 0.5)
    )
    expiries = np.unique(greeks.expiry)
    for e, ax in zip(expiries[2:5], axes):
        sr2 = greeks.sr2.sel(expiry=e).to_series().unstack('payoff')
        sr2 = sr2.rename(columns=dict(C='call', P='put'))
        sr2.plot(ax=ax, marker='o', linewidth=0)

        ax.set_title('expiry: {}'.format(np.datetime_as_string(e, unit='D')))
        ax.set_ylabel('$sR^2$')
        ax.legend()

    fig.tight_layout()
    return fig
