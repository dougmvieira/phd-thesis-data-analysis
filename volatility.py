import numpy as np
import xarray as xr
import statsmodels.api as sm
from arch.bootstrap import optimal_block_length, StationaryBootstrap
from scipy.stats import norm

from heston_calibration import format_serialised
from utils import equal_split


n_splits = 20
coarse_interval = 300
days_in_year = 252


def integrated_variance(log_returns, alpha=0.05):
    log_returns = log_returns.dropna('time').data
    estimate = np.sum(log_returns**2)
    quarticity = np.sum(log_returns**4)
    half_band = norm.ppf(1 - alpha/2)*np.sqrt(2*quarticity/3)
    ivs = xr.DataArray([estimate, estimate - half_band, estimate + half_band],
                       {'end': ['estimate', 'lower', 'upper']}, 'end')
    return days_in_year*ivs


def sample_stds_map(log_returns):
    log_returns = log_returns.dropna('time').data
    estimate = np.sqrt(np.sum(log_returns**2))

    block_size = optimal_block_length(log_returns).loc[0, 'stationary']
    conf_int = StationaryBootstrap(block_size, log_returns)
    conf_int = conf_int.conf_int(lambda x: np.sqrt(np.sum(x**2)))[:, 0]
    stds = xr.DataArray([estimate, *conf_int],
                        {'end': ['estimate', 'lower', 'upper']}, 'end')
    return np.sqrt(days_in_year*n_splits)*stds


def compute_daily_stds(forwards_bonds):
    forwards_bonds = xr.combine_nested(list(forwards_bonds), concat_dim='date')
    log_returns = np.log(forwards_bonds.forward.isel(expiry=0))
    log_returns = log_returns.isel(time=slice(None, None, coarse_interval))
    log_returns = log_returns.diff('time')
    stds = np.sqrt(log_returns.groupby('date').map(integrated_variance))
    return stds.to_dataset('end')


def compute_sample_stds(forwards_bonds):
    log_returns = np.log(forwards_bonds.forward.isel(expiry=0)).diff('time')

    stds = equal_split(log_returns, 'time', n_splits).map(sample_stds_map)
    time = xr.DataArray([interval.mid for interval in stds.time_bins.data],
                        dims='time_bins')
    stds = stds.assign_coords(time=time)
    stds = stds.set_index({'time_bins': 'time'}).rename({'time_bins': 'time'})
    return stds.to_dataset('end')


def greeks_regression_map(mids_slice, forwards, heston_vols):
    forwards_slice = forwards.sel(expiry=mids_slice.expiry)
    regression_data = xr.Dataset(
        dict(mid=mids_slice, forward=forwards_slice, vol=heston_vols))
    regression_data = regression_data.isel(time=slice(None, None, 300))
    regression_data = regression_data.diff('time').dropna('time')
    regression_data['forward_squared'] = regression_data.forward ** 2
    if len(regression_data.mid) > 1:
        exog = regression_data.mid.to_series()
        endog = regression_data.drop('mid')
        endog = endog.reset_coords(drop=True).to_dataframe()
        endog = sm.add_constant(endog)
        fit = sm.OLS(exog, endog).fit()
        fit_novol = sm.OLS(exog, endog.drop('vol', axis=1)).fit()
        greeks = fit.conf_int().values
        r2 = fit.rsquared
        sr2 = r2 - fit_novol.rsquared
    else:
        greeks = np.nan*np.zeros((4, 2))
        r2 = sr2 = np.nan

    coords = dict(greek=['const', 'delta', 'vega', 'gamma'],
                  confidence=[0.025, 0.975])
    greeks = xr.DataArray(greeks, coords, ('greek', 'confidence'))
    greeks = greeks.assign_coords(mids_slice.strike.coords)
    return greeks.assign_coords(dict(r2=r2, sr2=sr2))


def greeks_regression(quotes, forwards_bonds, heston_params):
    if np.all(heston_params.vol.isnull().data):
        return heston_params.drop('vol')
    quotes = format_serialised(quotes, forwards_bonds)
    heston_vols = heston_params.vol
    forwards = forwards_bonds.forward
    mids = (quotes.ask + quotes.bid)/2
    mids = mids.dropna('option_id', how='all').reset_index('option_id')
    greeks = mids.groupby('option_id').map(
        lambda m: greeks_regression_map(m, forwards, heston_vols))
    return greeks.to_dataset('greek')
