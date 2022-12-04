import pandas as pd
import numpy as np
from urllib import request
from zipfile import ZipFile
from io import BytesIO
import datetime
import warnings
from typing import Union, Tuple, List, Optional


class GetFred:
    def __init__(
        self,
        transform: bool = False,
        start_date: Union[datetime.datetime, None] = None,
        end_date: Union[datetime.datetime, None] = None,
        vintage: str = "current",
    ):
        """
        Pull FRED-MD or FRED-QD data. Returns a Pandas DataFrame of golden copy FRED data.
        :param transform: stationarize using FRED-recommended transformations
        :param start_date: start date for data
        :param end_date: end date for data
        :param vintage: which version of the file to look at; 'current' uses the latest one

        Main functions are:
        :param get_fred_md(): pulls FRED MD data
        :param get_fred_qd(): pulls FRED QD data
        :param combine_fred(): combines FRED MD and FRED QD data using fuzzy match technique
        :param get_appendix(): pulls appendix from the site with metadata; pulls only updated version
        """
        self.transform = transform
        self.start_date = start_date
        self.end_date = end_date
        self._check_dates()
        self.vintage = vintage  # if not default "current" MUST be YYYY-MM
        self._check_vintage()
        self.url_format = "https://files.stlouisfed.org/files/htdocs/fred-md/"
        # The functions for stationarity are taken from the appendix PDF on the website
        self.stationarity_functions = {
            1: lambda l: l,
            2: lambda l: l.diff(),
            3: lambda l: l.diff().diff(),
            4: lambda l: np.log(l),
            5: lambda l: np.log(l).diff(),
            6: lambda l: np.log(l).diff().diff(),
            7: lambda l: (l / l.shift(1) - 1).diff(),
        }
        # These lookups from group numbers to names are taken from the PDF as well;
        # they are not available in an easy look-up format anywhere. This is why
        # manual definition is needed.
        self.group_lookup = {
            "FRED-MD": {
                1: "Output and Income",
                2: "Labor Market",
                3: "Housing",
                4: "Consumption, Orders, and Inventories",
                5: "Money and Credit",
                6: "Interest and Exchange Rates",
                7: "Prices",
                8: "Stock Market",
            },
            "FRED-QD": {  # note that these do not correspond to MD
                1: "National Income and Product Accounts (NIPA)",
                2: "Industrial Production",
                3: "Employment and Unemployment",
                4: "Housing",
                5: "Inventories, Orders, and Sales",
                6: "Prices",
                7: "Earnings and Productivity",
                8: "Interest Rates",
                9: "Money and Credit",
                10: "Household Balance Sheets",
                11: "Exchange Rates",
                12: "Other",
                13: "Stock Markets",
                14: "Non-Household Balance Sheets",
            },
        }

    def get_fred_md(
        self, group_no: Optional[List[int]] = None, use_descriptions: bool = False
    ) -> pd.DataFrame:
        """
        Returns FRED-MD data per class parameters specified by user.
        :param: group_no (list or None): indicates a specific group or groups to filter for
        :param: use_descriptions: replace series names with description names
        (not recommended as latter is long, but is less cryptic and hence more useful for plotting)
        :return: Pandas DataFrame
        """
        raw_df = self._get_file(freq="monthly")
        df, transf_codes = self._clean_md(raw_df)
        if self.transform:
            df = self._stationarize(df, transf_codes)
        if group_no:
            lookup = self.get_appendix(freq="monthly")
            group_names = {
                k: v for k, v in self.group_lookup["FRED-MD"].items() if k in group_no
            }
            warnings.warn(
                f"""Filtering for group(s) {group_names} as specified by user..."""
            )
            vars = lookup.loc[lookup["group"].isin(group_no), "fred"].to_list()
            df = df.iloc[:, df.columns.str.upper().isin([v.upper() for v in vars])]
        df = self._filter_dates(df)
        if use_descriptions:
            lookup = self.get_appendix(freq="monthly")
            var2desc = dict(zip(lookup["fred"], lookup["gsi:description"]))
            df = df.rename(mapper={k: v for k, v in var2desc.items()}, axis=1)
        return df

    def get_fred_qd(
        self,
        group_no: Optional[List[int]] = None,
        interpolate_to_monthly: bool = False,
        return_factors: bool = False,
        use_descriptions: bool = False,
    ) -> Union[Tuple[pd.DataFrame, pd.Series], pd.DataFrame]:
        """
        Returns FRED-QD data per class parameters specified by user.
        :param interpolate_to_monthly: interpolate quarterly results to monthly to match FRED-MD
        :param return_factors: return variable factors as a separate object
        :param use_descriptions: replace variable names with descriptions.
        Warning: this is particularly long for FRED-QD...
        :return: Pandas DataFrame
        """
        raw_df = self._get_file(freq="quarterly")
        df, transf_codes, factors = self._clean_qd(raw_df)
        if interpolate_to_monthly:  # include BEFORE stationarizing
            df = df.resample("MS").interpolate()
        if self.transform:
            df = self._stationarize(df, transf_codes)
        if group_no:
            warnings.warn(
                f"""Recall that FRED-QD has different groups than FRED-MD. See get_appendix(freq='quarterly') for details)"""
            )
            lookup = self.get_appendix(freq="quarterly")
            group_names = {
                k: v for k, v in self.group_lookup["FRED-QD"].items() if k in group_no
            }
            warnings.warn(
                f"""Filtering for group(s) {group_names} as specified by user..."""
            )
            vars = lookup.loc[lookup["Group"].isin(group_no), "FRED MNEMONIC"].to_list()
            df = df.iloc[:, df.columns.str.upper().isin([v.upper() for v in vars])]
        df = self._filter_dates(df)
        if use_descriptions:
            warnings.warn(
                f"""Descriptions for FRED-QD variable names are particularly long.
                              Just sayin..."""
            )
            lookup = self.get_appendix(freq="quarterly")
            var2desc = dict(zip(lookup["FRED MNEMONIC"], lookup["DESCRIPTION"]))
            df = df.rename(mapper={k: v for k, v in var2desc.items()}, axis=1)
        if return_factors:
            return df, factors
        else:
            return df

    def combine_fred(
        self,
        interpolate: bool = True,
        fred_md_group: Optional[List[int]] = None,
        fred_qd_group: Optional[List[int]] = None,
        use_descriptions=False,
    ) -> pd.DataFrame:
        """
        Returns a combined monthly-quarterly panel (with the default of interpolating quarterly to monthly)
        based on user-specific groups, or a custom methodology. Interpolates by default.
        :param fred_md_group: group numbers from FRED-MD to include
        :param fred_qd_group: group numbers from FRED-QD to include
        :param use_descriptions: replace variable names with descriptions.
        Warning: this is particularly long for FRED-QD... and perhaps not super informative
        :return: a monthly pandas DataFrame with the series together
        """
        warnings.warn(
            f"""FRED-MD and FRED-QD have duplicates (or transformed versions of the same series).
                          combine_fred() removes only obvious duplicates from FRED-QD (prioritizing FRED-MD)."""
        )
        if not fred_md_group and not fred_qd_group:
            warnings.warn(
                f"""No groups for FRED-MD or FRED-QD defined. Taking all variables from both and taking out only obvious duplicates.
                              Check output to ensure it is as you wish."""
            )
        md_df = self.get_fred_md(group_no=fred_md_group)
        qd_df = self.get_fred_qd(
            group_no=fred_qd_group, interpolate_to_monthly=interpolate
        )
        df = self._find_duplicates(fred_md_df=md_df, fred_qd_df=qd_df)
        if (
            use_descriptions
        ):  # note: it is best to do this AFTER combining vars because lot of repetition in descriptions
            warnings.warn(
                f"""Descriptions for FRED-QD variable names are particularly long.
                              Just sayin..."""
            )
            lookup_md = self.get_appendix(freq="monthly")
            lookup_qd = self.get_appendix(freq="quarterly")
            var2desc = dict(zip(lookup_md["fred"], lookup_md["gsi:description"]))
            var2desc_qd = dict(
                zip(lookup_qd["FRED MNEMONIC"], lookup_qd["DESCRIPTION"])
            )
            var2desc_qd = {
                k: v
                for k, v in var2desc_qd.items()
                if k.toupper() in df.columns.toupper()
            }  # you do not want duplicates back here
            var2desc.update(var2desc_qd)
            df = df.rename(mapper={k: v for k, v in var2desc.items()}, axis=1)
        return df

    def get_appendix(
        self, freq: str = "monthly", add_group_names: bool = True
    ) -> pd.DataFrame:
        """
        This is useful for getting group lookups; you get a direct lookup from variable names to groups this way,
        and Stock-Watson lookup information.
        :param freq - 'monthly' or 'quarterly'
        :param add_group_names - whether to add the 'Group_Name' column in addition to the column that gives group numbers
        :return: appendix Pandas Dataframe
        """
        clean_freq = freq[0].upper()
        z = request.urlopen(
            f"https://files.stlouisfed.org/files/htdocs/uploads/FRED-{clean_freq}D%20Appendix.zip"
        )
        myzip = ZipFile(BytesIO(z.read()))
        df = pd.read_csv(
            myzip.open(
                f"FRED-{clean_freq}D Appendix/FRED-{clean_freq}D_updated_appendix.csv"
            ),
            encoding="cp1252",
        )  # utf-8 fails cuz of quotes
        if add_group_names:
            try:
                df["Group_Name"] = df["group"].map(
                    self.group_lookup[f"FRED-{clean_freq}D"]
                )
            except KeyError:  # inconsistency in cases between MD and QD appendices
                df["Group_Name"] = df["Group"].map(
                    self.group_lookup[f"FRED-{clean_freq}D"]
                )
        return df

    def _stationarize(self, df: pd.DataFrame, transf_codes: pd.Series) -> pd.DataFrame:
        """
        Uses the FRED-recommended transformation codes to transform each column to stationarity.
        See self.stationarity_functions for the actual transformations.
        :param df: input raw Pandas DataFrame
        :param transf_codes: the code corresponding to the proper function to stationarize a given series
        :return: stationarized Pandas DataFrame
        """
        df_trans = []
        for s in range(df.shape[1]):  # perform transformations
            s_trans = self.stationarity_functions[transf_codes[s]](df.iloc[:, s])
            df_trans.append(s_trans)
        df_trans = pd.DataFrame(df_trans).T.dropna(how="all")
        return df_trans

    def _clean_qd(
        self, raw_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Return a cleaned dataframe with transformation codes and factors taken out and a nice
        indexed date; return transformation codes and factors separately.
        Applicable to FRED-QD since QD has "factors" in addition to transformation codes.
        :param raw_df: Pandas Dataframe based on the raw pull of FRED-QD data
        :return: a cleaned Pandas DataFrame, Pandas Series of transformation codes, Pandas Series of factors
        """
        factors = raw_df.iloc[0, 1:]
        transf_codes = raw_df.iloc[1, 1:]
        df = raw_df.iloc[2:, 0:]
        df = self._clean_df(df)
        return df, transf_codes, factors

    def _clean_md(self, raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Return a cleaned dataframe with transformation codes taken out and a nice
        indexed date and returns transformation codes separately. Applicable to FRED-MD since QD has "factors".
        :param raw_df: Pandas Dataframe based on the raw pull of FRED-MD data
        :return: a cleaned Pandas DataFrame, Pandas Series of transformation codes
        """
        transf_codes = raw_df.iloc[0, 1:]
        df = raw_df.iloc[1:, 0:]
        df = self._clean_df(df)
        return df, transf_codes

    def _clean_df(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        General dataframe cleaner across MD and QD.
        :param raw_df: Pandas DataFrame to be cleaned
        :return: cleaned Pandas DataFrame
        """
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

    def _find_duplicates(self, *, fred_md_df, fred_qd_df) -> pd.DataFrame:
        """
        This is a helper function to remove duplicate columns between FRED-MD and FRED-QD.

        It is important to note that it is not a simple matter of duplication: QD has transformed
        MD columns, and vice versa. This method removes it using a simple duplication test as well
        as a "fuzzy match" that keeps MD columns wherever a transformed column occurs in either.
        Transformations are usually indicated by the suffix 'x'.

        :param fred_md_df: cleaned FRED-MD DataFrame
        :param fred_qd_df: cleaned FRED-QD DataFrame
        :return: combined FRED DataFrame without duplicates
        """

        warnings.warn(
            f"""Removing FRED-QD duplicates and fuzzy duplicates from FRED-QD before combining.
                          See documentation for more info."""
        )

        # initial pass: get EXACT duplicates in FRED-QD from FRED-MD
        exact_duplicate_cols = fred_qd_df.columns[
            fred_qd_df.columns.isin(fred_md_df.columns)
        ].to_list()
        # fuzzy match pass:
        # remove 'x' from columns of both types and prioritize both originals and transformations in FRED-MD
        fuzzymatch_qd_cols = [s[:-1] for s in fred_qd_df.columns if s[-1] == "x"]
        fuzzymatch_md_cols = [s[:-1] for s in fred_md_df.columns if s[-1] == "x"]
        ignore_fuzzymatch_qd = [
            col + "x" for col in fuzzymatch_qd_cols if col in fred_md_df.columns
        ]
        keep_fuzzymatch_md = [
            col for col in fuzzymatch_md_cols if col in fred_qd_df.columns
        ]
        ignore_cols = exact_duplicate_cols + ignore_fuzzymatch_qd + keep_fuzzymatch_md
        # combine all together, again keepign everything that was in FRED MD and ignoring duplicates from FRED-QD
        combined_df = pd.concat(
            [fred_md_df, fred_qd_df.drop(columns=ignore_cols)], axis=1
        )
        return combined_df

    def _filter_dates(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        If ``start_date`` and/or ``end_date`` specified will filter the dataframe
        to be within these dates. Otherwise returns the same dataframe.

        If the end date is chosen to be early, some columns could have all NA;
        the function leaves them in and warns the user.

        :param df: Pandas DataFrame with Datetime Index
        :return: Pandas DataFrame
        """
        if self.start_date:
            df = df.loc[self.start_date :]
        if self.end_date:
            df = df.loc[: self.end_date]
        all_zero_cols = df.columns[df.isnull().all()].to_list()
        if all_zero_cols:
            warnings.warn(
                f"""The following columns have only NaN values: {','.join(all_zero_cols)}.
                              See FRED-MD/QD documentation for more details; some series begin later than others."""
            )
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
            try:
                year, mon = self.vintage.split("-")
            except ValueError:
                raise Exception(f"Invalid vintage: {self.vintage}. Format YYYY-MM")
            if (int(year) < 2015) or (int(year) > datetime.date.today().year):
                raise Exception(f"Invalid year: {year}. Format YYYY-MM.")
            if (int(mon) > 12) or (int(mon) < 1):
                raise Exception(f"Invalid month: {mon}. Format YYYY-MM")
            if (int(year) == datetime.date.today().year) and (
                int(mon) > datetime.date.today().month
            ):
                raise Exception(
                    f"Vintage {self.vintage} too far into the future. Request vintage=current for latest data"
                )
                
    def _check_dates(self):
        """
        A verification function to check if start and end dates are legitimate
        :return: warnings for erroneous dates
        """
        if self.start_date is None:
            pass
        elif self.start_date <= datetime.date(1959, 1, 1):
            warnings.warn(
                f"FRED-MD and QD data not available before 1959. Removing start_date filter..."
            )
            self.start_date = None
        if self.end_date is None:
            pass
        elif self.end_date >= datetime.date.today():
            warnings.warn(
                f"Specified end_date {self.end_date} greater than current date. Removing end_date filter..."
            )
            self.end_date = None
