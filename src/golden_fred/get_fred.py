import pandas as pd
import numpy as np
from urllib import request
from datetime import date
import warnings
import os


class GetFred:
    """
    Pull FRED-MD or FRED-QD data
    """

    def __init__(self, vintage="current"):
        self.vintage = vintage  # if not default "current" MUST be YYYY-MM
        self._check_vintage()
        self.url_format = "https://files.stlouisfed.org/files/htdocs/fred-md/"

    def get_fredmd(self):
        df = self._get_file(freq="monthly")

    def _get_file(self, freq="monthly"):
        """
        Pull the source file from the FRED-MD site into Pandas DataFrame.
        :param freq: 'monthly' or 'quarterly'
        :return: Pandas DataFrame with raw results
        """
        if freq not in ["monthly", "quarterly"]:
            raise Exception(f"Invalid frequency: {freq}. Must be monthly or quarterly.")

        url = f"{self.url_format}/{freq}/{self.vintage}.csv"
        response = request.urlopen(url)
        csv = response.read()

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
            if (year < 2015) or (year > date.today().year):
                raise Exception(f"Invalid year: {year}. Format YYYY-MM.")
            if (mon > 12) or (mon < 1):
                raise Exception(f"Invalid month: {mon}. Format YYYY-MM")
            if (year == date.today().year) and (mon > date.today().month):
                raise Exception(
                    f"Vintage {self.vintage} too far into the future. Request vintage=current for latest data"
                )
