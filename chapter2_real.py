from dataclasses import dataclass
from hashlib import sha1
from typing import Any, Callable

import numpy as np
import xarray as xr
from lazymaker import lazymake, add_dummy_args, add_side_effects
from tqdm import tqdm

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
results_folder = 'chapter2_real/'

second = np.timedelta64(1, 's').astype('m8[ns]')
start_time = np.timedelta64(8, 'h').astype('m8[ns]')
itm_strike = 3000
midday = '12:00:00'
n_expiries = 5
n_workers = 12
mid, first, last = 0, 1, -1
data_folders = [
    '20211203/',
    '20211122/',
    '20211123/',
    '20211124/',
    '20211125/',
    '20211126/',
    '20211129/',
    '20211130/',
    '20211201/',
    '20211202/',
    '20211206/',
    '20211207/',
    '20211208/',
    '20211209/',
    '20211210/',
    '20211213/',
    '20211214/',
    '20211215/',
    '20211216/',
]


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
quotes_lst = []
for folder in tqdm(data_folders):
    tick_quotes = HashedStruct(np.load(folder + tick_quotes_filename))
    quotes = make_dataset(
        folder + quotes_filename,
        format_gridded_quotes,
        tick_quotes,
        second,
        start_time,
        bootstrap_source,
    )
    quotes_lst.append(quotes)


print('Bootstrapping')
parity_lst = []
weights_lst = []
underlying_lst = []
forwards_bonds_lst = []
for folder, quotes in zip(tqdm(data_folders), quotes_lst):
    parity = make_dataset(
        folder + parity_filename,
        pair_puts_and_calls,
        quotes,
        bootstrap_source,
    )
    weights = make_dataset(
        folder + weights_filename,
        pca_on_midprice,
        parity,
        bootstrap_source,
    )
    underlying = make_dataset(
        folder + underlying_filename,
        recover_underlying,
        parity,
        weights,
        bootstrap_source,
    )
    forwards_bonds = make_dataset(
        folder + forwards_bonds_filename,
        forward_bond_regression,
        parity,
        underlying,
        bootstrap_source,
    )
    parity_lst.append(parity)
    weights_lst.append(weights)
    underlying_lst.append(underlying)
    forwards_bonds_lst.append(forwards_bonds)

print('No-arbitrage bounds')
arbitrage_source = read_source('arbitrage.py')
no_arb_bounds = make_dataset(
    data_folders[mid] + no_arb_bounds_filename,
    compute_no_arb_bounds,
    parity_lst[mid],
    midday,
    arbitrage_source,
)

print('Computing implied vols')
heston_calibration_source = read_source('heston_calibration.py')
ivs_lst = []
for folder, quotes, forwards_bonds in zip(
    tqdm(data_folders), quotes_lst, forwards_bonds_lst
):
    ivs = make_dataset(
        folder + ivs_filename,
        compute_ivs,
        quotes,
        forwards_bonds,
        n_workers,
        heston_calibration_source,
    )
    ivs_lst.append(ivs)

print('Calibrating Heston')
params_guess = make_dataset(
    data_folders[mid] + params_guess_filename,
    calibrate_heston_with_app,
    ivs_lst[mid],
    forwards_bonds_lst[mid],
    midday,
    heston_calibration_source,
)
heston_params_lst = []
for folder, ivs, forwards_bonds in zip(
    data_folders, ivs_lst, forwards_bonds_lst
):
    heston_params = make_dataset(
        folder + heston_params_filename,
        calibrate_heston,
        ivs,
        forwards_bonds,
        midday,
        params_guess,
        n_workers,
        False,
        folder,
        heston_calibration_source,
    )
    heston_params_lst.append(heston_params)

print('Volatility analysis')
volatility_source = read_source('volatility.py')
sample_stds_lst = []
greeks_lst = []
for folder, quotes, forwards_bonds, heston_params in zip(
    tqdm(data_folders), quotes_lst, forwards_bonds_lst, heston_params_lst
):
    sample_stds = make_dataset(
        folder + sample_stds_filename,
        compute_sample_stds,
        forwards_bonds,
        volatility_source
    )
    greeks = make_dataset(
        folder + greeks_filename,
        greeks_regression,
        quotes,
        forwards_bonds,
        heston_params,
        volatility_source
    )
    sample_stds_lst.append(sample_stds)
    greeks_lst.append(greeks)

print('Plots and tables')
bootstrap_plots_source = ''.join([read_source('bootstrap.py'),
                                  read_source('heston_calibration.py'),
                                  read_source('bootstrap_plots.py')])
volatility_plots_source = ''.join([read_source('volatility.py'),
                                   read_source('volatility_plots.py')])
make_plot(
    results_folder + 'recover_comp.png',
    compare_with_itm,
    underlying_lst[mid],
    quotes_lst[mid],
    itm_strike,
    bootstrap_plots_source
)
make_plot(
    results_folder + 'recover_pca.png',
    plot_pca,
    weights_lst[mid],
    bootstrap_plots_source,
)
make_plot(
    results_folder + 'forwards.png',
    plot_forwards,
    forwards_bonds_lst[mid],
    bootstrap_plots_source,
)
make_plot(
    results_folder + 'interest_rates.png',
    plot_interest_rates,
    parity_lst[mid],
    forwards_bonds_lst[mid],
    bootstrap_plots_source,
)
make_table(
    results_folder + 'bootstrap_spreads.tex',
    tabulate_bootstrap_spreads,
    forwards_bonds_lst[mid],
    bootstrap_plots_source,
)
make_plot(
    results_folder + 'heston_calibration_mid.png',
    heston_calibration_plot,
    ivs_lst[mid],
    forwards_bonds_lst[mid],
    heston_params_lst[mid],
    midday,
    bootstrap_plots_source
)
make_plot(
    results_folder + 'heston_calibration_first.png',
    heston_calibration_plot,
    ivs_lst[first],
    forwards_bonds_lst[first],
    heston_params_lst[first],
    midday,
    bootstrap_plots_source
)
make_plot(
    results_folder + 'heston_calibration_last.png',
    heston_calibration_plot,
    ivs_lst[last],
    forwards_bonds_lst[last],
    heston_params_lst[last],
    midday,
    bootstrap_plots_source
)
make_table(
    results_folder + 'heston_params.tex',
    heston_calibration_table,
    heston_params_lst[mid],
    midday,
    bootstrap_plots_source
)
make_plot(
    results_folder + 'synthetic_forwards_noarb_bounds.png',
    plot_parity_bounds,
    no_arb_bounds,
    parity_lst[mid],
    midday,
    n_expiries,
    arbitrage_source,
)
make_plot(
    results_folder + 'noarb_intervals_forward.png',
    plot_forward_bounds,
    no_arb_bounds,
    arbitrage_source
)
make_plot(
    results_folder + 'noarb_intervals_yield.png',
    plot_no_arb_yield_curve,
    no_arb_bounds,
    parity_lst[mid],
    arbitrage_source
)
make_plot(
    results_folder + 'noarb_intervals_dividend.png',
    plot_no_arb_dividend_curve,
    no_arb_bounds,
    parity_lst[mid],
    arbitrage_source
)
make_plot(
    results_folder + 'heston_volatility.png',
    heston_volatility_plot,
    heston_params_lst[mid],
    sample_stds_lst[mid],
    volatility_plots_source
)
for payoff in ['C', 'P']:
    for greek in ['delta', 'vega', 'gamma', 'const']:
        payoff_name = 'calls' if payoff == 'C' else 'puts'
        plot_filename = f'{payoff_name}_{greek}.png'
        make_plot(
            results_folder + plot_filename,
            greeks_plot,
            greeks_lst[mid],
            heston_params_lst[mid],
            forwards_bonds_lst[mid],
            greek,
            payoff,
            volatility_plots_source
        )
make_plot(
    results_folder + 'greeks_r2.png',
    greeks_r2_plot,
    greeks_lst[mid],
    forwards_bonds_lst[mid],
    volatility_plots_source
)
make_plot(
    results_folder + 'greeks_sr2.png',
    greeks_sr2_plot,
    greeks_lst[mid],
    forwards_bonds_lst[mid],
    volatility_plots_source
)
