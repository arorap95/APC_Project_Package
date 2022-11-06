import pandas as pd
import numpy as np
from urllib import request
from warnings import warn
import os


class GetFred:
    """
    Pull FRED-MD or FRED-QD data
    """

    def __init__(self, vintage="current"):
        self.vintage = vintage  # if not default "current" MUST be YYYY-MM
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
