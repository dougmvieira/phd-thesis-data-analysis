import tkinter as tk
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from fyne import blackscholes, heston

from utils import cols_to_args


class CustomRangeScale(tk.Frame):
    def __init__(self, parent, label, floor, ceil, var):
        super().__init__(parent)

        self._ceil = tk.StringVar(value=str(ceil))
        self._floor = tk.StringVar(value=str(floor))
        self._ceil.trace_add('write', self._update)
        self._floor.trace_add('write', self._update)

        tk.Label(self, text=label).pack()
        tk.Entry(self, textvariable=self._ceil, width=6).pack()
        tk.Scale(self, variable=var, name='scale').pack()
        tk.Entry(self, textvariable=self._floor, width=6).pack()

        self._update()

    def _update(self, *args):
        try:
            to = float(self._floor.get())
            from_ = float(self._ceil.get())
        except ValueError:
            return

        scale = self.children['scale']
        scale.configure(to=to, from_=from_, resolution=(to - from_)/20)


class App(tk.Tk):
    def __init__(self, fig, plot_update, calibrate, labels, defaults, floors,
                 ceils):
        super().__init__()
        self.params = [tk.DoubleVar() for _ in labels]

        self._canvas = FigureCanvasTkAgg(fig, self)
        controls = tk.Frame(self)

        self._canvas.get_tk_widget().pack()
        controls.pack()

        def vars_plot_update(*args):
            plot_update(*map(tk.DoubleVar.get, self.params))
            self._canvas.draw()

        for label, default, floor, ceil, param in zip(labels, defaults, floors,
                                                      ceils, self.params):
            scale = CustomRangeScale(controls, label, floor, ceil, param)
            scale.pack(side=tk.LEFT)

            param.set(default)
            param.trace_add('write', vars_plot_update)

        def vars_calibrate():
            new_values = calibrate(*map(tk.DoubleVar.get, self.params))
            for param, new_value in zip(self.params, new_values):
                param.set(new_value)

        button = tk.Button(controls, text='Calibrate', command=vars_calibrate)
        button.pack(side=tk.LEFT)

        vars_plot_update()

    def run(self):
        self.mainloop()
        return map(tk.DoubleVar.get, self.params)


def heston_smile(y, ax):
    underlying_price = 100
    expiry = 0.2

    def closure(vol, kappa, theta, nu, rho):
        ax.cla()
        option_prices = heston.formula(underlying_price, y.index, expiry, vol,
                                       kappa, theta, nu, rho)
        y.loc[:] = blackscholes.implied_vol(underlying_price, y.index, expiry,
                                            option_prices)
        y.plot(ax=ax)

    return closure


def heston_fit(vols_data, vols_model, underlying_price, ax):
    def closure(vol, kappa, theta, nu, rho):
        ax.cla()
        strikes = vols_model.index.get_level_values('Strike')
        expiries = vols_model.index.get_level_values('Expiry')
        prices_model = heston.formula(underlying_price, strikes, expiries, vol,
                                      kappa, theta, nu, rho)
        vols_model.loc[:] = blackscholes.implied_vol(underlying_price, strikes,
                                                     expiries, prices_model,
                                                     assert_no_arbitrage=False)
        for expiry, color in zip(np.unique(expiries),
                                 plt.get_cmap('tab10').colors):
            vols_data.xs(expiry).plot(ax=ax, c=color, marker='o', linewidth=0)
            vols_model.xs(expiry).plot(ax=ax, color=color)

    return closure


def heston_calibration(underlying_price, strikes, expiries, option_prices,
                       put, **kwargs):
    def closure(*initial_guess):
        return heston.calibration_crosssectional(
            underlying_price, strikes, expiries, option_prices, initial_guess,
            put, **kwargs)

    return closure


def heston_app(underlying_price, strikes, expiries, option_prices, put,
               **kwargs):
    backend = mpl.get_backend()
    mpl.use('agg')

    fig, ax = plt.subplots()

    index = pd.MultiIndex.from_arrays([expiries, strikes],
                                      names=['Expiry', 'Strike'])
    vols_data = blackscholes.implied_vol(underlying_price, strikes, expiries,
                                         option_prices, put)
    vols_data = pd.Series(vols_data, index, name='Data')
    underlying_unique = pd.Series(underlying_price,
                                  pd.Index(expiries, name='Expiry')
                         ).groupby('Expiry').first()

    strike_grid = np.linspace(np.min(strikes), np.max(strikes), 100)
    index = pd.MultiIndex.from_product([np.unique(expiries), strike_grid],
                                       names=['Expiry', 'Strike'])
    vols_model = pd.Series(0, index, name='Model')
    underlying_b = underlying_unique.reindex(index.get_level_values('Expiry'))

    labels = ['vol', 'kappa', 'theta', 'nu', 'rho']
    defaults = [0.1, 7.2, 0.05, 1.25, -0.54]
    floors = [0.0, 1., 0.0, 0.0, -1.]
    ceils = [1.0, 10., 1.0, 5.0, 1.]
    plot_update = heston_fit(vols_data, vols_model, underlying_b, ax)
    calibrate = heston_calibration(underlying_price, strikes, expiries,
                                   option_prices, put, **kwargs)
    app = App(fig, plot_update, calibrate, labels, defaults, floors, ceils)

    params = app.run()
    mpl.use(backend)

    return params


def open_app():
    backend = mpl.get_backend()
    mpl.use('agg')

    fig, ax = plt.subplots()
    x = pd.Index(np.linspace(80, 120, 100), name='Strike')
    y = pd.Series(0, x)

    labels = ['vol', 'kappa', 'theta', 'nu', 'rho']
    defaults = [0.1, 7.2, 0.05, 1.25, -0.54]
    floors = [0.0, 1., 0.0, 0.0, -1.]
    ceils = [1.0, 10., 1.0, 5.0, 1.]
    app = App(fig, heston_smile(y, ax), str, labels, defaults, floors,
              ceils)

    power, scale = app.run()
    mpl.use(backend)

    return power, scale
