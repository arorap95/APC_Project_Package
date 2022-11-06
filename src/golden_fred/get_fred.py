import pandas as pd
import numpy as np
from urllib import request
import datetime
import warnings
from typing import Union, Tuple


class GetFred:
    def __init__(
        self,
        transform: bool = True,
        start_date: Union[datetime.datetime, None] = None,
        end_date: Union[datetime.datetime, None] = None,
        vintage: str = "current",
    ):
        """
        Pull FRED-MD or FRED-QD data. Returns a Pandas DataFrame of golden copy FRED data.
        :param transform: transform to stationarity
        :param start_date: start date for data
        :param end_date: end date for data
        :param vintage: which version of the file to look at; 'current' uses the latest one

        Main functions are:
        :param get_fred_md(): pulls FRED MD data
        :param get_fred_qd(): pulls FRED QD data
        """
        self.transform = transform
        self.start_date = start_date
        self.end_date = end_date
        self.vintage = vintage  # if not default "current" MUST be YYYY-MM
        self._check_vintage()
        self.url_format = "https://files.stlouisfed.org/files/htdocs/fred-md/"
        self.stationarity_functions = {
            1: lambda l: l,
            2: lambda l: l.diff(),
            3: lambda l: l.diff().diff(),
            4: lambda l: np.log(l),
            5: lambda l: np.log(l).diff(),
            6: lambda l: np.log(l).diff().diff(),
            7: lambda l: (l / l.shift(1) - 1).diff(),
        }

    def get_fred_md(self) -> pd.DataFrame:
        """
        Returns FRED-MD data per class parameters specified by user.
        :return: Pandas DataFrame
        """
        raw_df = self._get_file(freq="monthly")
        df, transf_codes = self._clean_md(raw_df)
        if self.transform:
            df = self._stationarize(df, transf_codes)
        df = self._filter_dates(df)
        return df

    def get_fred_qd(
        self, interpolate_to_monthly: bool = False, return_factors: bool = False
    ) -> Union[Tuple[pd.DataFrame, pd.Series], pd.DataFrame]:
        """
        Returns FRED-QD data per class parameters specified by user.
        :param interpolate_to_monthly: interpolate quarterly results to monthly to match FRED-MD
        :param return_factors: return variable factors as a separate object
        :return: Pandas DataFrame
        """
        raw_df = self._get_file(freq="quarterly")
        df, transf_codes, factors = self._clean_qd(raw_df)
        if self.transform:
            df = self._stationarize(df, transf_codes)
        df = self._filter_dates(df)
        if interpolate_to_monthly:
            df = df.resample("MS").interpolate()
        if return_factors:
            return df, factors
        else:
            return df

    def _stationarize(self, df: pd.DataFrame, transf_codes: pd.Series) -> pd.DataFrame:
        df_trans = []
        for s in range(df.shape[1]):  # perform transformations
            s_trans = self.stationarity_functions[transf_codes[s]](df.iloc[:, s])
            df_trans.append(s_trans)
        df_trans = pd.DataFrame(df_trans).T.dropna(how="all")
        return df_trans

    def _clean_qd(
        self, raw_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        factors = raw_df.iloc[0, 1:]
        transf_codes = raw_df.iloc[1, 1:]
        df = raw_df.iloc[2:, 0:]
        df = self._clean_df(df)
        return df, transf_codes, factors

    def _clean_md(self, raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Return a cleaned dataframe with transformation codes taken out and a nice
        indexed date. Applicable to FRED-MD since QD has "factors".
        :param raw_df: Pandas Dataframe based on the raw pull of FRED data
        :return: a cleaned Pandas DataFrame
        """
        transf_codes = raw_df.iloc[0, 1:]
        df = raw_df.iloc[1:, 0:]
        df = self._clean_df(df)
        return df, transf_codes

    def _clean_df(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        df = raw_df.dropna(how="all")  # drop any rows where ALL are NaN
        df = df.rename(columns={"sasdate": "date"})
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        return df

    def _get_file(self, freq: str = "monthly") -> pd.DataFrame:
        """
        Pull the source file from the FRED-MD site into Pandas DataFrame.
        :param freq: 'monthly' or 'quarterly'
        :return: Pandas DataFrame with raw results
        """
        url = f"{self.url_format}/{freq}/{self.vintage}.csv"
        df = pd.read_csv(url)
        return df

    def _filter_dates(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        if self.start_date:
            df = df.loc[self.start_date :]
        if self.end_date:
            df = df.loc[self.end_date :]
        return df

    def _check_vintage(self):
        """
        A verification function to ensure a proper vintage is being fed into the class.
        It will throw an exception if any issues
        :return: a warning if vintage != default or Exception if vintage isn't proper.
        """
        if self.vintage == "current":
            pass
        else:
            warnings.warn(
                f"""It is advised to use the default vintage: current.
                          If requesting a historical vintage, use format YYYY-MM.
                          Oldest vintage is 2015-01."""
            )
            if "-" not in self.vintage:
                raise Exception(
                    f"Incorrect vintage format: {self.vintage}. Correct format: YYYY-MM"
                )
            year, mon = self.vintage.split("-")
            if (year < 2015) or (year > datetime.date.today().year):
                raise Exception(f"Invalid year: {year}. Format YYYY-MM.")
            if (mon > 12) or (mon < 1):
                raise Exception(f"Invalid month: {mon}. Format YYYY-MM")
            if (year == datetime.date.today().year) and (
                mon > datetime.date.today().month
            ):
                raise Exception(
                    f"Vintage {self.vintage} too far into the future. Request vintage=current for latest data"
                )
