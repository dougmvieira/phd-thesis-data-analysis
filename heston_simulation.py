from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from math import exp, sqrt

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numba import njit
from scipy import stats
from tqdm import tqdm
from fyne import heston


@njit
def simulate_heston(spot_init, vol_init, mu, kappa, theta, nu, rho, step, z_scores):
    n_samples, dim = z_scores.shape
    assert dim == 2
    z0_scale_spot = sqrt(step)
    z0_scale_vol = sqrt(step) * rho
    z1_scale_vol = sqrt(step * (1 - rho ** 2))
    spot = np.full(n_samples + 1, spot_init, dtype=np.float64)
    vol = np.full(n_samples + 1, vol_init, dtype=np.float64)
    for i in range(n_samples):
        vol[i + 1] = vol[i]
        vol[i + 1] += kappa * (theta - vol[i]) * step
        vol[i + 1] += nu * sqrt(vol[i]) * (z0_scale_vol * z_scores[i, 0] + z1_scale_vol * z_scores[i, 1])
        if vol[i + 1] < 0:
            vol[i + 1] *= -1
        spot[i + 1] = spot[i]
        spot[i + 1] *= exp((mu - 1 / 2) * vol[i] * step + sqrt(vol[i]) * z0_scale_spot * z_scores[i, 0])
    return spot, vol


def plot_heston_across_timescales(forwards_bonds, heston_params):
    kappa = heston_params.kappa.item()
    theta = heston_params.theta.item()
    nu = heston_params.nu.item()
    rho = heston_params.rho.item()
    spot_init = forwards_bonds.forward.dropna('time').isel(expiry=0, time=0).item()
    vol_init = heston_params.vol.dropna('time').isel(time=0).item()
    mu = 0
    n_samples = 100_000
    np.random.seed(42)
    z_scores = np.random.randn(n_samples, 2)
    fig, axes = plt.subplots(2, 2, sharex='col')
    for unit, (ax_spot, ax_vol) in zip(['hours', 'years'], axes.T):
        step = (1 / 252 if unit == 'hours' else 1) / n_samples
        spot, vol = simulate_heston(spot_init, vol_init, mu, kappa, theta, nu, rho, step, z_scores)
        timestamps = (8 if unit == 'hours' else 10) * np.arange(n_samples + 1) / (n_samples + 1)
        ax_spot.plot(timestamps, spot)
        ax_vol.plot(timestamps, vol)
        ax_vol.set_xlabel(unit)
    axes[0, 0].set_title('Short timescale')
    axes[0, 1].set_title('Large timescale')
    axes[0, 0].set_ylabel('Price')
    axes[1, 0].set_ylabel('Variance')
    return fig


def plot_tricking_normality_tests(heston_params):
    kappa = heston_params.kappa.item()
    theta = heston_params.theta.item()
    nu = heston_params.nu.item()
    rho = heston_params.rho.item()
    spot_init = 1.
    vol_init = theta
    mu = 0
    n_samples = 2_000
    fig, axes = plt.subplots(2, 2, sharey=True)
    np.random.seed(42)
    steps = np.exp(np.linspace(np.log(1), np.log(1_000), 100))
    colors = mpl.cm.get_cmap('viridis')(np.linspace(0, 1, 100))
    spot_p_values_list = []
    vol_p_values_list = []
    for step, color in zip(steps, colors):
        z_scores = np.random.randn(n_samples, 2)
        annualized_step = step * (np.timedelta64(1, 's') / np.timedelta64(8 * 252, 'h'))
        spot, vol = simulate_heston(
            spot_init, vol_init, mu, kappa, theta, nu, rho, annualized_step, z_scores
        )
        for p_values_list, process, ax in [(spot_p_values_list, spot, axes[0, 0]), (vol_p_values_list, vol, axes[0, 1])]:
            diffs = np.diff(process)
            _, kolmogorov_smirnov = stats.ks_1samp(diffs, stats.norm.cdf)
            _, shapiro = stats.shapiro(diffs)
            _, jarque_bera = stats.jarque_bera(diffs)
            _, dagostino_pearson = stats.normaltest(diffs)
            p_values_list.append(
                {
                    'Kolmogorov-Smirnov': kolmogorov_smirnov,
                    'Shapiro': shapiro,
                    'Jarque-Bera': jarque_bera,
                    'D\'Agostino-Pearson': dagostino_pearson
                }
            )
            ax.set_xlabel('Z-scores')
            ax.set_xlim((-3, 3))
            pd.Series(diffs / np.std(diffs)).value_counts(normalize=True).sort_index().cumsum().plot(ax=ax, color=color)
    for p_values_list, ax in [
        (spot_p_values_list, axes[1, 0]),
        (vol_p_values_list, axes[1, 1])
    ]:
        p_values = pd.DataFrame(p_values_list, steps)
        p_values.plot(ax=ax, logx=True, marker='o', alpha=.5, linewidth=0)
        ax.set_xlabel('Time step (seconds)')
    divider = make_axes_locatable(axes[0, 1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(mpl.cm.ScalarMappable(mpl.colors.Normalize(0, 3), 'viridis'), cax=cax, format=lambda x, _: f'$10^{{{x}}}$', label='time step (seconds)')
    axes[0, 0].set_title('Price differences')
    axes[0, 1].set_title('Variance differences')
    axes[0, 0].set_ylabel('CDF')
    axes[1, 0].set_ylabel('P-values')
    return fig


def generate_quotes(ivs, forwards_bonds, heston_params, tick_size, n_workers):
    ivs = ivs.load()
    valid_options = (ivs.bid.isnull() | ivs.ask.isnull()).sum('time') < len(ivs.time) / 2
    ivs = ivs.sel(option_id=valid_options)
    n_samples = len(ivs.time) - 1
    step = 1 / (n_samples * 252)
    np.random.seed(42)
    z_scores = np.random.randn(n_samples, 2)
    kappa = heston_params.kappa.item()
    theta = heston_params.theta.item()
    nu = heston_params.nu.item()
    rho = heston_params.rho.item()
    spot_init = forwards_bonds.forward.dropna('time').isel(expiry=0, time=0).item()
    vol_init = heston_params.vol.dropna('time').isel(time=0).item()
    mu = 0
    spot, vol = simulate_heston(spot_init, vol_init, mu, kappa, theta, nu, rho, step, z_scores)
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = executor.map(
            heston.formula,
            spot,
            repeat(ivs.strike.values),
            repeat(ivs.years_to_expiry.values),
            vol,
            repeat(kappa),
            repeat(theta),
            repeat(nu),
            repeat(rho),
            repeat(ivs.payoff.values == 'P'),
            repeat(False),
        )
        option_prices = np.stack(list(tqdm(futures, total=len(spot))))
    bids_arr = tick_size * np.floor(option_prices / tick_size)
    asks_arr = tick_size * np.ceil(option_prices / tick_size)
    quotes = xr.Dataset(
        dict(
            bid=(('option_id', 'time'), bids_arr.T),
            ask=(('option_id', 'time'), asks_arr.T),
            spot=('time', spot),
            vol=('time', vol),
        ),
        ivs.coords,
        ivs.dims,
    )
    return quotes


# TODO:
# - Get calibrated Heston params from data
# - Get strikes and expiries from data
# - Simulate Heston for the day
# - Construct rounded forward and option prices
# - Plot Epps effect between ATM option and forward price
# - Calibrate Heston model from simulated option prices and forwards
# - Compute sample stds from underlying

# TODO plots:
# - Heston simulation for a day vs Heston for 10 years
# - Normality test p-values as time deltas get small
# - Epps effect between ATM option and rounded underlying price
# - Heston vols vs std vols
# - Greeks plots
