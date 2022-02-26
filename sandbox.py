from dataclasses import dataclass
from hashlib import sha1
from typing import Any, Callable

import numpy as np
import xarray as xr
from lazymaker import lazymake, add_dummy_args, add_side_effects

from bootstrap import (format_gridded_quotes, pair_puts_and_calls,
                       pca_on_midprice, recover_underlying,
                       forward_bond_regression)
from bootstrap_plots import (compare_with_itm, plot_pca, plot_forwards,
                             plot_interest_rates, tabulate_bootstrap_spreads,
                             heston_calibration_plot, heston_calibration_table)
from arbitrage import (compute_no_arb_bounds, plot_parity_bounds,
                       plot_forward_bounds, plot_no_arb_yield_curve,
                       plot_no_arb_dividend_curve)
from heston_calibration import (compute_ivs, calibrate_heston,
                                calibrate_heston_with_app)
from volatility import compute_sample_stds, greeks_regression
from volatility_plots import (heston_volatility_plot, greeks_plot,
                              greeks_r2_plot, greeks_sr2_plot)
from utils import read_source, hash_dataset


cache_filename = 'cache.json'
tick_quotes_filename = 'tick_quotes.npz'
quotes_filename = 'quotes.nc'
underlying_filename = 'recovered_underlying.nc'
parity_filename = 'parity.nc'
weights_filename = 'underlying_weights.nc'
no_arb_bounds_filename = 'no_arb_bounds.nc'
forwards_bonds_filename = 'forwards_bonds.nc'
ivs_filename = 'ivs.nc'
params_guess_filename = 'params_guess.nc'
heston_params_filename = 'heston_params.nc'
sample_stds_filename = 'sample_stds.nc'
greeks_filename = 'greeks.nc'
results_dir = 'results/'

second = np.timedelta64(1, 's').astype('m8[ns]')
start_time = np.timedelta64(8, 'h').astype('m8[ns]')
itm_strike = 3000
midday = '12:00:00'
n_workers = 12


@dataclass(frozen=True)
class HashedStruct:
    struct: np.void

    @property
    def lazymaker_hash(self) -> str:
        return str(self.struct['lazymaker_hash'])


def make_struct(
    name: str, compute: Callable[..., np.void], *args: Any
) -> np.void:
    def read(name: str) -> HashedStruct:
        return HashedStruct(np.lib.format.open_memmap(name, mode='r'))

    def compute_and_persist(*args: Any) -> HashedStruct:
        args = [
            arg.struct if isinstance(arg, HashedStruct) else arg for arg in args
        ]
        struct = compute(*args[:-1])
        struct['lazymaker_hash'] = ''
        struct['lazymaker_hash'] = sha1(struct.tobytes()).hexdigest()
        np.save(name, struct, allow_pickle=False)
        return HashedStruct(struct)

    return lazymake(cache_filename, name, compute_and_persist, args, read).struct


def make_dataset(name, compute, *args):
    def compute_and_persist(*args):
        args = [
            arg.struct if isinstance(arg, HashedStruct) else arg for arg in args
        ]
        dataset = compute(*args[:-1])
        hashed = hash_dataset(dataset)
        hashed.to_netcdf(name)
        return hashed

    return lazymake(cache_filename, name, compute_and_persist, args,
                    xr.open_dataset)


def make_plot(name, compute, *args):
    def read(name):
        pass

    compute = add_dummy_args(compute, 1)
    compute_and_persist = add_side_effects(compute, lambda f: f.savefig(name))

    return lazymake(cache_filename, name, compute_and_persist, args, read)


def make_table(name, compute, *args):
    def read(name):
        pass

    def persist(name, contents):
        with open(name, 'w') as f:
            f.write(contents)

    compute = add_dummy_args(compute, 1)
    compute_and_persist = add_side_effects(compute, lambda c: persist(name, c))

    return lazymake(cache_filename, name, compute_and_persist, args, read)


print('Gridding quotes')
bootstrap_source = read_source('bootstrap.py')
tick_quotes = HashedStruct(
    np.lib.format.open_memmap(tick_quotes_filename, mode='r')
)
quotes = make_dataset(quotes_filename, format_gridded_quotes, tick_quotes,
                      second, start_time, bootstrap_source)


print('Bootstrapping')
parity = make_dataset(parity_filename, pair_puts_and_calls, quotes,
                      bootstrap_source)
weights = make_dataset(weights_filename, pca_on_midprice, parity,
                       bootstrap_source)
underlying = make_dataset(underlying_filename, recover_underlying, parity,
                          weights, bootstrap_source)
forwards_bonds = make_dataset(forwards_bonds_filename, forward_bond_regression,
                              parity, underlying, bootstrap_source)

print('No-arbitrage bounds')
arbitrage_source = read_source('arbitrage.py')
no_arb_bounds = make_dataset(no_arb_bounds_filename, compute_no_arb_bounds,
                             parity, midday, arbitrage_source)

print('Heston calibration')
heston_calibration_source = read_source('heston_calibration.py')
ivs = make_dataset(ivs_filename, compute_ivs, quotes, forwards_bonds,
                   n_workers, heston_calibration_source)
params_guess = make_dataset(params_guess_filename, calibrate_heston_with_app,
                            ivs, forwards_bonds, midday,
                            heston_calibration_source)
heston_params = make_dataset(heston_params_filename, calibrate_heston, ivs,
                             forwards_bonds, midday, params_guess, n_workers,
                             heston_calibration_source)

print('Volatility analysis')
volatility_source = read_source('volatility.py')
sample_stds = make_dataset(sample_stds_filename, compute_sample_stds,
                           forwards_bonds, volatility_source)
greeks = make_dataset(greeks_filename, greeks_regression, quotes,
                      forwards_bonds, heston_params, volatility_source)

print('Plots and tables')
bootstrap_plots_source = ''.join([read_source('bootstrap.py'),
                                  read_source('heston_calibration.py'),
                                  read_source('bootstrap_plots.py')])
volatility_plots_source = ''.join([read_source('volatility.py'),
                                   read_source('volatility_plots.py')])
make_plot(results_dir + 'recover_comp.png', compare_with_itm, underlying,
          quotes, itm_strike, bootstrap_plots_source)
make_plot(results_dir + 'recover_pca.png', plot_pca, weights,
          bootstrap_plots_source)
make_plot(results_dir + 'forwards.png', plot_forwards, forwards_bonds,
          bootstrap_plots_source)
make_plot(results_dir + 'interest_rates.png', plot_interest_rates, parity,
          forwards_bonds, bootstrap_plots_source)
make_table(results_dir + 'bootstrap_spreads.tex', tabulate_bootstrap_spreads,
           forwards_bonds, bootstrap_plots_source)
make_plot(results_dir + 'heston_calibration.png', heston_calibration_plot, ivs,
          forwards_bonds, heston_params, midday, bootstrap_plots_source)
make_table(results_dir + 'heston_params.tex', heston_calibration_table,
           heston_params, midday, bootstrap_plots_source)
make_plot(results_dir + 'synthetic_forwards_noarb_bounds.png',
          plot_parity_bounds, no_arb_bounds, parity, midday, 5,
          arbitrage_source)
make_plot(results_dir + 'noarb_intervals_forward.png', plot_forward_bounds,
          no_arb_bounds, arbitrage_source)
make_plot(results_dir + 'noarb_intervals_yield.png', plot_no_arb_yield_curve,
          no_arb_bounds, parity, arbitrage_source)
make_plot(results_dir + 'noarb_intervals_dividend.png',
          plot_no_arb_dividend_curve, no_arb_bounds, parity, arbitrage_source)
make_plot(results_dir + 'heston_volatility.png', heston_volatility_plot,
          heston_params, sample_stds, volatility_plots_source)
make_plot(results_dir + 'calls_deltas.png', greeks_plot, greeks, heston_params,
          forwards_bonds, 'delta', 'C', volatility_plots_source)
make_plot(results_dir + 'calls_vegas.png', greeks_plot, greeks, heston_params,
          forwards_bonds, 'vega', 'C', volatility_plots_source)
make_plot(results_dir + 'calls_gammas.png', greeks_plot, greeks, heston_params,
          forwards_bonds, 'gamma', 'C', volatility_plots_source)
make_plot(results_dir + 'calls_consts.png', greeks_plot, greeks, heston_params,
          forwards_bonds, 'const', 'C', volatility_plots_source)
make_plot(results_dir + 'puts_deltas.png', greeks_plot, greeks, heston_params,
          forwards_bonds, 'delta', 'P', volatility_plots_source)
make_plot(results_dir + 'puts_vegas.png', greeks_plot, greeks, heston_params,
          forwards_bonds, 'vega', 'P', volatility_plots_source)
make_plot(results_dir + 'puts_gammas.png', greeks_plot, greeks, heston_params,
          forwards_bonds, 'gamma', 'P', volatility_plots_source)
make_plot(results_dir + 'puts_consts.png', greeks_plot, greeks, heston_params,
          forwards_bonds, 'const', 'P', volatility_plots_source)
make_plot(results_dir + 'greeks_r2.png', greeks_r2_plot, greeks,
          volatility_plots_source)
make_plot(results_dir + 'greeks_sr2.png', greeks_sr2_plot, greeks,
          volatility_plots_source)
