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


# create data to test all functions
def create_input():
    data = get_fred.GetFred()
    df = data.get_fred_md()

    return df


df = create_input()

# create fixture that generates instance of the class
@pytest.fixture
def c():
    def _c(df, start_date=None, end_date=None, rebalancing="monthly", handle_missing=1):
        return fred_backtest.FredBacktest(
            df, start_date, end_date, rebalancing, handle_missing
        )

    yield _c


@pytest.mark.parametrize("rebalancing", ["monthly", "quarterly", "annually"])
def test_init(c, rebalancing):
    myobject = c(df, rebalancing=rebalancing)
    assert myobject.inputdata.equals(df)


def test_fill(c):
    myobject = c(df)
    myobject.fred_compute_backtest(
        ["RPI", "W875RX1", "IPDCONGD"], [0.2, 0.5, 0.3], [0, 0, 0]
    )

    assert myobject.inputdata.notnull().values.any()


def test_backtest(c):
    myobject = c(df)
    myobject.fred_compute_backtest(
        ["RPI", "W875RX1", "IPDCONGD"], [0.2, 0.5, 0.3], [0, 0, 0]
    )

    assert myobject.stats.loc["Annualized Turnover (%)"] == 0
    assert (
        myobject.stats.loc["Max DD from Base"] >= myobject.stats.loc["Max Drawdown (%)"]
    )
