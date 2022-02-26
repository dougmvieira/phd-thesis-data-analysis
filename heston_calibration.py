from concurrent.futures import ProcessPoolExecutor
from functools import partial
from multiprocessing import Pool

import numpy as np
import xarray as xr
from fyne import blackscholes, heston
import py_lets_be_rational
from tqdm import tqdm

from calibration_app import heston_app
from utils import combine_dataarrays


@np.vectorize
def implied_vol(forward, strike, expiry, option_price, put_flag):
    f = py_lets_be_rational.implied_volatility_from_a_transformed_rational_guess
    try:
        iv = f(option_price, forward, strike, expiry, -1 if put_flag else 1)
    except py_lets_be_rational.exceptions.VolatilityValueException:
        iv = np.nan
    return iv


def format_serialised(quotes, forwards_bonds):
    quotes = quotes.set_index(option_id=['payoff', 'expiry', 'strike'])
    quotes = quotes.sortby(['payoff', 'expiry', 'strike'])
    discounted_strikes = forwards_bonds.bond.reindex(
        expiry=quotes.expiry.values).values*quotes.strike
    quotes = quotes.assign_coords(discounted_strike=discounted_strikes)
    quotes = quotes.where(quotes.expiry != quotes.date, drop=True)
    return quotes.where(~quotes.discounted_strike.isnull(), drop=True)


def calibration_selection(ivs, forwards):
    moneyness = combine_dataarrays(
        lambda s, f: np.where(s.payoff.data == 'C', s - f, f - s),
        'expiry', ivs.strike, forwards)
    return ivs.where(moneyness > 0, drop=True)


def calibrate_vol_map(ivs_slice, forwards, vol_guess, kappa, theta, nu, rho):
    ivs_slice = ivs_slice.dropna('option_id')

    if len(ivs_slice.mid) > 1:
        forwards_slice = forwards.sel(time=ivs_slice.time)
        ivs_slice = calibration_selection(ivs_slice, forwards_slice)
        forwards_slice = forwards_slice.sel(expiry=ivs_slice.expiry.values)

        calls = blackscholes.formula(
            forwards_slice.values, ivs_slice.discounted_strike,
            ivs_slice.years_to_expiry, ivs_slice.mid)

        try:
            vol = heston.calibration_vol(
                forwards_slice.values, ivs_slice.discounted_strike.values,
                ivs_slice.years_to_expiry.values, calls, kappa, theta, nu, rho,
                vol_guess=vol_guess, weights=1/ivs_slice.spread)
        except ValueError:
            print('WARNING: Runtime error in Heston vol calibration.')
            vol = np.nan

    else:
        vol = np.nan

    return vol


def calibration_preprocessing(ivs, forwards_bonds, time):
    ivs = ivs.set_index({'option_id': ['payoff', 'expiry', 'strike']})
    ivs['mid'] = (ivs.ask + ivs.bid) / 2
    ivs['spread'] = ivs.ask - ivs.bid
    forwards = forwards_bonds.forward

    expiries = np.unique(ivs.expiry)
    ivs = ivs.sel(option_id=ivs.expiry.isin(expiries[2:5]))
    forwards = forwards.sel(expiry=expiries[2:5])

    ivs_slice_raw = ivs.sel(time=time)
    forwards_slice = forwards.sel(time=time)
    deltas = blackscholes.delta(
        forwards_slice.sel(expiry=ivs_slice_raw.expiry.values).values,
        ivs_slice_raw.discounted_strike, ivs_slice_raw.years_to_expiry,
        ivs_slice_raw.mid)
    ivs = ivs.sel(option_id=(0.05 <= deltas) & (deltas < 0.95))


    ivs_slice = calibration_selection(
        ivs.sel(time=time).dropna('option_id'), forwards_slice)
    calls = blackscholes.formula(
        forwards_slice.sel(expiry=ivs_slice.expiry.values).values,
        ivs_slice.discounted_strike, ivs_slice.years_to_expiry, ivs_slice.mid)

    return ivs, forwards_slice, ivs_slice, calls


def calibrate_heston_with_app(ivs, forwards_bonds, time):
    _, forwards_slice, ivs_slice, calls = calibration_preprocessing(
        ivs.load(), forwards_bonds, time)

    names = ['vol', 'kappa', 'theta', 'nu', 'rho']
    params_guess = heston_app(
        forwards_slice.sel(expiry=ivs_slice.expiry.values).values,
        ivs_slice.discounted_strike.values, ivs_slice.years_to_expiry.values, calls, False,
        weights=1/ivs_slice.spread, enforce_feller_cond=True)
    params_guess = xr.DataArray(list(params_guess), dict(name=names), 'name')
    return xr.Dataset({'param': params_guess})


def calibrate_heston(
    ivs,
    forwards_bonds,
    time,
    params_guess,
    n_workers,
    recalibrate_crosssection=True,
    tqdm_desc=None,
):
    params_guess = params_guess.param
    ivs, forwards_slice, ivs_slice, calls = calibration_preprocessing(
        ivs.load(), forwards_bonds.load(), time)

    vol, kappa, theta, nu, rho = (
        heston.calibration_crosssectional(
            forwards_slice.sel(expiry=ivs_slice.expiry.values).values,
            ivs_slice.discounted_strike, ivs_slice.years_to_expiry,
            calls,
            params_guess.values,
            weights=1/ivs_slice.spread,
            enforce_feller_cond=True,
        ) if recalibrate_crosssection else params_guess.values
    )

    calibrate_smile = partial(calibrate_vol_map,
                              forwards=forwards_bonds.forward, vol_guess=vol,
                              kappa=kappa, theta=theta, nu=nu, rho=rho)

    desc = tqdm_desc or 'Heston vol calibration'
    with Pool(n_workers) as p:
        vols_iter = p.imap(calibrate_smile,
                           (smile for _, smile in ivs.groupby('time')))
        vols = np.array(list(tqdm(vols_iter, desc=desc, total=len(ivs.time))))

    vols = xr.DataArray(vols, dict(time=ivs.time, date=ivs.date), 'time')

    heston_params = vols.to_dataset(name='vol').assign_coords(
        dict(kappa=kappa, theta=theta, nu=nu, rho=rho))

    return heston_params


def compute_ivs_map(quotes, forward):
    return implied_vol(forward.expand_dims('option_id'),
                       quotes.discounted_strike.values[:, None],
                       quotes.years_to_expiry.values[:, None],
                       quotes,
                       quotes.payoff.values[:, None] == 'P')


def compute_ivs(quotes, forwards_bonds, n_workers):
    quotes = format_serialised(quotes, forwards_bonds.load())
    forwards = forwards_bonds.forward
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        ivs_ask = combine_dataarrays(compute_ivs_map, 'expiry', quotes.ask,
                                     forwards, executor.map, name='ask')
        ivs_bid = combine_dataarrays(compute_ivs_map, 'expiry', quotes.bid,
                                     forwards, executor.map, name='bid')

    return xr.merge([ivs_bid, ivs_ask]).reset_index('option_id')
