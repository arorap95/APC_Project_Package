from goldenfred import fred_covariance
import pytest
import random
import numpy as np
import datetime
import pandas as pd


# create data to test all functions
def create_input():
    x = [
        datetime.datetime(2022, 11, 6),
        datetime.datetime(2022, 11, 7),
        datetime.datetime(2022, 11, 8),
        datetime.datetime(2022, 11, 9),
        datetime.datetime(2022, 11, 10),
        datetime.datetime(2022, 11, 11),
    ]

    y = [np.nan, np.nan, 12, 100, 8, 1]
    z = [np.nan, np.nan, 200, 2, 11, 2]

    df = pd.DataFrame({"date": x, "column1": y, "column2": z})
    df.set_index("date", inplace=True)

    return df


df = create_input()

# create fixture that generates instance of the class
@pytest.fixture
def c():
    def _c(df):
        return fred_covariance.CovarianceFred(df)

    yield _c


def test_init(c):
    myobject = c(df)
    assert myobject.originaldata.equals(df)


def test_threshold_covaraince(c):
    myobject = c(df)
    threshold = np.random.random()
    covmtx = myobject.threshold_covariance(correlationthreshold=threshold)
    assert covmtx[covmtx > 0].min().min() >= threshold


def test_positivedefinite(c):
    myobject = c(df)
    threshold = np.random.random()
    covmtx = myobject.positive_semidefinite_method1()
    np.linalg.cholesky(covmtx)
