from golden_fred import get_fred
import pytest
from typing import Union
import datetime
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal

sample_data = pd.Series(np.random.randint(0, 100, 100))


@pytest.fixture
def c():
    def _c(
        transform: bool = False,
        start_date: Union[datetime.datetime, None] = None,
        end_date: Union[datetime.datetime, None] = None,
        vintage: str = "current",
    ):
        return get_fred.GetFred(transform, start_date, end_date, vintage)

    yield _c


@pytest.mark.parametrize(
    "start_date",
    [datetime.date(2020, 3, 1), datetime.date(2020, 4, 1), datetime.date(2020, 5, 1)],
)
def test_init(c, start_date):
    myobject = c(start_date=start_date)
    assert myobject.start_date == start_date
    assert myobject.transform == False  # this should also be the default
    assert myobject.vintage == "current"  # this default should never change


@pytest.mark.parametrize("transf_code", [1, 2, 3, 4, 5, 6, 7])
def test_stationarity(c, transf_code):
    myobject = c()
    if transf_code == 1:
        manual_transf = sample_data
        assert manual_transf.equals(myobject.stationarity_functions[1](sample_data))
    elif transf_code == 2:
        manual_transf = sample_data.diff()
        assert manual_transf.equals(myobject.stationarity_functions[2](sample_data))
    elif transf_code == 3:
        manual_transf = sample_data.diff().diff()
        assert manual_transf.equals(myobject.stationarity_functions[3](sample_data))
    elif transf_code == 4:
        manual_transf = sample_data.diff().diff()
        assert manual_transf.equals(myobject.stationarity_functions[4](sample_data))
    elif transf_code == 5:
        manual_transf = np.log(sample_data).diff()
        assert manual_transf.equals(myobject.stationarity_functions[5](sample_data))
    elif transf_code == 6:
        manual_transf = np.log(sample_data).diff()
        assert manual_transf.equals(myobject.stationarity_functions[6](sample_data))
    elif transf_code == 7:
        manual_transf = np.log(sample_data).diff()
        assert manual_transf.equals(myobject.stationarity_functions[7](sample_data))


@pytest.mark.parametrize("vintage", ["2015-06", "2017-02", "2019-04"])
def test_vintage_start(c, vintage):
    """
    Ensure the last date is not after the vintage date. Ensure the first date is not before
    the start date of the data.
    """
    myobject = c(vintage=vintage)
    df = myobject.get_fred_md()
    year, mon = vintage.split("-")
    assert df.index.max() <= pd.Timestamp(datetime.date(int(year), int(mon), 1))
    assert df.index.min() >= pd.Timestamp("1959-01-01")  # default start date of FRED-MD


@pytest.mark.parametrize("vintage", ["random", "2017-13", "2300-04"])
def test_vintage_format(c, vintage):
    """
    Ensure that putting in invalid `vintage` format throws an error.
    """
    with pytest.raises(Exception):
        df = c(vintage=vintage)


@pytest.mark.parametrize(
    "fred_qd_group", [[1, 2], [4, 7], [5, 8, 9], [10, 11, 12], None]
)
def test_duplicates(c, fred_qd_group):
    """
    The purpose of this function is to test that combine_fred leads to NO exact
    duplicate columns. This is very important, as it is tricky to ensure that the
    two are completely separate between MD and QD.
    """
    myobject = c()
    df = myobject.combine_fred(fred_qd_group=fred_qd_group)
    assert df.columns.duplicated().sum() == 0
    assert df.duplicated().sum() == 0


# not parametrizing due to different object dictionary names
def test_appendix_cols(c):
    """
    The purpose of this function is to ensure that the groups saved in the class object
    correspond to the NUMBER of groups in the appendix for both QD and MD. If it breaks,
    this is a good time to check the documentation and add any new groups.
    There are also some general inconsistencies in conventions between the
    two appendices.
    """
    myobject = c()
    appendix_md = myobject.get_appendix(freq="monthly")
    appendix_qd = myobject.get_appendix(freq="quarterly")
    unique_groups_md = appendix_md["group"].nunique()
    unique_groups_qd = appendix_qd["Group"].nunique()
    assert unique_groups_md == len(myobject.group_lookup["FRED-MD"])
    assert unique_groups_qd == len(myobject.group_lookup["FRED-QD"])


def test_interp_freq(c):
    """
    Ensure that the interpolation frequency stays at the beginning of the month
    in the index.
    """
    myobject = c()
    fred_qd = myobject.get_fred_qd(interpolate_to_monthly=True)
    assert fred_qd.index.freq == "MS"  # testing frequency


@pytest.mark.parametrize(
    "date",
    [
        pd.Timestamp("1959-06-01"),
        pd.Timestamp("1959-03-01"),
        pd.Timestamp("1971-02-01"),
        pd.Timestamp("1999-12-01"),
        pd.Timestamp("2022-01-01"),
    ],
)
def test_interp_freq(c, date):
    """
    Test that the interpolation happens BEFORE the stationarity.
    This would be a big mistake if not done: it could lead to countless econometric
    issues, and it is easily done by flipping the order of the ``get()`` class.
    """
    myobject_stationary = c(transform=True)
    myobject_levels = c(transform=False)
    qd_stationary = c(interpolate_to_monthly=True)
    qd_levels = c(interpolate_to_monthly=False)
    qd_levels_interp = qd_levels.resample("MS").interpolate()
    transf_codes = myobject_levels._clean_qd(
        myobject_levels._get_file(freq="quarterly")
    )[1]
    qd_levels_interp_transform = myobject_levels._stationarize(
        qd_levels_interp, transf_codes
    )
    assert (qd_levels_interp_transform.loc[date] == qd_stationary.loc[date]).all()
    assert_frame_equal(qd_levels_interp_transform.head(1), qd_stationary.head(1))
    assert_frame_equal(qd_levels_interp_transform.tail(1), qd_stationary.tail(1))


@pytest.mark.parametrize("group_no", [[1], [2], [3], [1, 2], [8], [4, 5, 6, 7]])
def test_md_group_varcount(c, group_no):
    """
    Determine the counts of variables from pulling the full dataset for a given
    group number is equal to the number of variables in the appendix for that group.
    Important to not lose variables.
    """

    myobject = c()
    group_df = myobject.get_fred_md(group_no=group_no)
    col_counts = len(group_df.columns)
    appendix_df = myobject.get_appendix(freq="monthly")
    appendix_df_filtered = appendix_df[appendix_df["group"].isin(group_no)]
    assert col_counts == len(appendix_df_filtered)


@pytest.mark.parametrize(
    "group_no", [[1], [2], [3], [1, 2], [8], [4, 5, 6, 7], [11, 12]]
)
def test_qd_group_varcount(c, group_no):
    """
    Determine the counts of variables from pulling the full dataset for a given
    group number is equal to the number of variables in the appendix for that group.
    Important to not lose variables.
    """

    myobject = c()
    group_df = myobject.get_fred_qd(group_no=group_no)
    col_counts = len(group_df.columns)
    appendix_df = myobject.get_appendix(freq="quarterly")
    appendix_df_filtered = appendix_df[appendix_df["Group"].isin(group_no)]
    assert col_counts == len(appendix_df_filtered)


def test_descriptions_md(c):
    """
    If you replace variable names with descriptions, then there should
    no longer be variable names and only descriptions.
    """
    myobject = c()
    df = myobject.get_fred_md(use_descriptions=True)
    appendix_df = myobject.get_appendix(freq="monthly")
    # NOTE: you may notice NAs in the above; this comes directly
    # from the appendix and is not incorrect
    vars = appendix_df["fred"].to_list()
    assert (
        sum([v in df.columns for v in vars]) == 4
    )  # there are 4 that should match: the S&P ones


def test_descriptions_qd(c):
    """
    If you replace variable names with descriptions, then there should
    no longer be variable names and only descriptions. This is specific to QD.
    """
    myobject = c()
    df = myobject.get_fred_qd(use_descriptions=True)
    appendix_df = myobject.get_appendix(freq="quarterly")
    vars = appendix_df["FRED MNEMONIC"].to_list()
    assert sum([v in df.columns for v in vars]) == 0  # QD uses longer descriptions
