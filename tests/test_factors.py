import golden_fred
from golden_fred import fred_factors
import pytest
import random
import numpy as np
import datetime
import pandas as pd
from numpy.linalg import eigh


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
    def _c(
        df,
        standardization=2,
        maxfactors=None,
        factorselection=None,
        removeoutliers=None,
        handle_missing=1,
    ):
        return fred_factors.FredFactors(
            df,
            standardization,
            maxfactors,
            factorselection,
            removeoutliers,
            handle_missing,
        )

    yield _c


# test __init function
@pytest.mark.parametrize("standardization", [0, 1, 2])
def test_init(c, standardization):
    myobject = c(df, standardization=standardization)
    assert myobject.standardization == standardization
    assert myobject.currentdata.equals(df)


# test _removeoutliers function by comparing object output with outliers of sample data
@pytest.mark.parametrize("removeoutliers", [True, False])
def test_outliers(c, removeoutliers):
    myobject = c(df, removeoutliers=removeoutliers)

    myobject._removeoutliers()

    if removeoutliers == False:
        assert myobject.currentdata.equals(df)

    elif removeoutliers == True:
        lb = df.quantile(0.25)
        ub = df.quantile(0.75)
        iqr = ub - lb
        med = df.quantile(0.5)
        df_new = df[(df > med - (10 * iqr)) & (df < med + (10 * iqr))]
        assert df_new.equals(myobject.currentdata)


# test _fillmissing function
@pytest.mark.parametrize("handle_missing", [0, 1])
def test_handlemissing(c, handle_missing):
    myobject = c(df, removeoutliers=True)
    myobject._removeoutliers()
    myobject._fillmissing()

    assert myobject.currentdata.notnull().values.any()


# test _standardize function
@pytest.mark.parametrize("standardization", [0, 1, 2])
def test_standardization(c, standardization):
    myobject = c(df, standardization=standardization)
    myobject._removeoutliers()
    myobject._fillmissing()

    scalar = myobject._standardize()
    scaleddata = pd.DataFrame(scalar.fit_transform(myobject.currentdata))

    if standardization == 0:
        assert np.array_equal(
            myobject.currentdata.iloc[:, 0].values, scaleddata.iloc[:, 0].values
        )

    # test demeaning - sum ~0
    elif standardization == 1:
        means = scaleddata.sum()
        assert (means.gt(-1e-10) & means.lt(1e-10)).all()

    # test demeaning and descaling - sum~0 and std~1
    elif standardization == 2:
        means = scaleddata.sum()
        assert (means.gt(-1e-10) & means.lt(1e-10)).all()

        stds = scaleddata.std()
        assert (stds.gt(0.98) & stds.lt(1.2)).all()


# test _runPCAalgorithm by comparing eigenvalues of sample data and eigenvalues of the object instance returned
def test_PCA(c):
    myobject = c(df)
    myobject._removeoutliers()
    myobject._fillmissing()
    myobject._runPCAalgorithm()

    # test _PCA manually
    myobject1 = c(df)
    myobject1._removeoutliers()
    myobject1._fillmissing()
    scalar = myobject1._standardize()
    scaleddata = scalar.fit_transform(myobject1.currentdata)
    cov_mat = np.cov(scaleddata, rowvar=False)
    w, v = eigh(cov_mat)

    # compare numpy eigenvalues with object.eigenvalues returned to user
    assert (np.round(sorted(w), 3) == np.round(sorted(myobject.eigenvalues), 3)).all()


@pytest.mark.parametrize("factorselection", [{0: 3}, {1: 80}, {2: 0}])
def test_optimal_factor(c, factorselection):
    myobject = c(df, factorselection=factorselection)
    myobject._removeoutliers()
    myobject._fillmissing()
    myobject._runPCAalgorithm()
    myobject._selectoptimalfactors()

    # test _selectoptimalfactors manually
    myobject1 = c(df, factorselection=factorselection)
    myobject1._removeoutliers()
    myobject1._fillmissing()
    scalar = myobject1._standardize()
    scaleddata = scalar.fit_transform(myobject1.currentdata)
    cov_mat = np.cov(scaleddata, rowvar=False)
    w, v = eigh(cov_mat)

    explained_variance_ratio = w / sum(w)

    selection = list(myobject1.factorselection.keys())[0]
    target = myobject1.factorselection[selection]

    if selection == 0:
        additional_explained_variance = (
            np.round((explained_variance_ratio), decimals=4) * 100
        )
        assert (
            myobject.optimalfactors
            == np.argmax(additional_explained_variance < target) + 1
        )

    elif selection == 1:
        cumvar = np.cumsum(np.round(explained_variance_ratio, decimals=4) * 100)
        assert myobject.optimalfactors == np.argmax(cumvar > target) + 1

    elif selection == 2:
        additional_explained_variance = (
            np.round((explained_variance_ratio), decimals=4) * 100
        )
        drops = additional_explained_variance[:-1] / additional_explained_variance[1:]
        assert myobject.optimalfactors == (-drops).argsort()[target] + 1
