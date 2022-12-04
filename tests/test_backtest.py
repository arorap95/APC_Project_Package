from golden_fred import fred_backtest
from golden_fred import get_fred

import pandas as pd
import numpy as np
from urllib import request
import datetime
import warnings
from typing import Union, Tuple
import sklearn as sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import sys
import pytest
import math
from scipy import sparse
import cvxpy as cp


# create data to test all functions
def create_input():
    data = get_fred.GetFred()
    df = data.get_fred_md()

    return df


df = create_input()

# create fixture that generates instance of the Backtest class
@pytest.fixture
def c():
    def _c(df, start_date=None, end_date=None, rebalancing="monthly", handle_missing=1):
        return fred_backtest.FredBacktest(
            df, start_date, end_date, rebalancing, handle_missing
        )

    yield _c


#test whether all rebalancing techniques function
@pytest.mark.parametrize("rebalancing", ["monthly", "quarterly", "annually"])
def test_init(c, rebalancing):
    myobject = c(df, rebalancing=rebalancing)
    assert myobject.inputdata.equals(df)

#test both handle_missing parameters function
@pytest.mark.parametrize("handle_missing", [0, 1])
def test_fill(c, handle_missing):
    myobject = c(df, handle_missing=handle_missing)
    myobject.fred_compute_backtest(
        ["RPI", "W875RX1", "IPDCONGD"], [0.2, 0.5, 0.3], [0, 0, 0]
    )

    assert myobject.inputdata.notnull().values.any()


# test that start and end dates work as expected, including edge cases for beginning and ending dates
@pytest.mark.parametrize(
    "start_date",
    [
        datetime.datetime(1959, 1, 1),
        datetime.datetime(1959, 2, 1),
        datetime.datetime(2022, 8, 1),
    ],
)
def test_dates(c, start_date):
    myobject = c(df, start_date=start_date)
    assert myobject.startdate == start_date

    myobject._fillmissing()
    returns = myobject.inputdata.pct_change()[1:]

    myobject.fred_compute_backtest(["RPI"], [1.0], [0])
    assert math.isclose(
        myobject.stats.loc["Average Annual Return (%)"],
        returns.loc[start_date:, "RPI"].mean() * 12,
        abs_tol=1e-3,
    )
    assert math.isclose(
        myobject.stats.loc["Annualized Vol (%)"],
        returns.loc[start_date:, "RPI"].std() * np.sqrt(12),
        abs_tol=1e-3,
    )


@pytest.mark.parametrize(
    "end_date", [datetime.datetime(1959, 3, 1), datetime.datetime(2022, 8, 1)]
)
def test_dates(c, end_date):
    myobject = c(df, end_date=end_date)
    assert myobject.enddate == end_date

    myobject._fillmissing()
    returns = myobject.inputdata.pct_change()[1:]

    myobject.fred_compute_backtest(["RPI"], [1.0], [0])
    assert math.isclose(
        myobject.stats.loc["Average Annual Return (%)"],
        returns.loc[:end_date, "RPI"].mean() * 12,
        abs_tol=1e-3,
    )
    assert math.isclose(
        myobject.stats.loc["Annualized Vol (%)"],
        returns.loc[:end_date, "RPI"].std() * np.sqrt(12),
        abs_tol=1e-3,
    )


# test that statistics work as expected with basic test cases
def test_backtest(c):
    myobject = c(df)
    myobject._fillmissing()
    returns = myobject.inputdata.pct_change()[1:]

    myobject.fred_compute_backtest(
        ["RPI", "W875RX1", "IPDCONGD"], [0.2, 0.5, 0.3], [0, 0, 0]
    )

    assert myobject.stats.loc["Annualized Turnover (%)"] == 0
    assert (
        myobject.stats.loc["Max DD from Base"] >= myobject.stats.loc["Max Drawdown (%)"]
    )
    assert math.isclose(
        (returns[["RPI", "W875RX1", "IPDCONGD"]] * [0.2, 0.5, 0.3]).mean().sum() * 12,
        myobject.stats.loc["Average Annual Return (%)"],
        abs_tol=1e-3,
    )
    assert math.isclose(
        (returns[["RPI", "W875RX1", "IPDCONGD"]] * [0.2, 0.5, 0.3]).sum(axis=1).std()
        * np.sqrt(12),
        myobject.stats.loc["Annualized Vol (%)"],
        abs_tol=1e-3,
    )

#test regime filtering outputs are as expected
def test_regime(c):
    myobject = c(df)
    myobject._fillmissing()

    myobject.regime_filtering(["RPI", "W875RX1"])

    for col in myobject.regimes.columns:
        assert myobject.regimes[col].all() in [-1, 0, 1]
