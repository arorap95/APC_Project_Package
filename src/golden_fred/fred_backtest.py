import pandas as pd
import numpy as np
import datetime
import warnings
import sys
from typing import Union, Tuple


class FredBacktest:
    def __init__(
        self,
        data: pd.DataFrame,
        start_date: Union[datetime.datetime, None] = None,
        end_date: Union[datetime.datetime, None] = None,
        rebalancing: str = "monthly",
        handle_missing: int = 1,
    ) -> None:
        """
        Run a backtest on a strategy of FRED factors with weights and customized inputs provided by the user.
        Report all backtest statistics of the strategy

        :param data: cleaned Fred Data Monthly Data outputted from GetFred()
        :param start_date: start date for backtest
        :param end_date: end date for backtest
        :param rebalancing: rebalancing technique for backtest. Options are: monthly, quarterly, yearly, ntz (trade zones)
        :param handle_missing: is an integer in [0,1] representing:
        0: Forward Fill followed by Backward Fill missing values
        1: Fill missing values with mean of respective series

        Main functions are:
        :param fred_compute_backtest(): runs backtest and outputs statistics
        """

        # Check that parameters are set correctly
        # -------------------------------------------------------------------------------------------------------
        assert (
            data.index[1].month - data.index[0].month == 1
        ), "Must provide monthly Fred-MD data"

        assert handle_missing in [
            0,
            1,
        ], "Handle Missing parameter must be an integer in [0,1]"
        assert rebalancing in [
            "monthly",
            "quarterly",
            "annually",
        ], "Rebalancing input not recognized"
        assert isinstance(
            data.index, pd.DatetimeIndex
        ), "Index of input data must be a datetime index"

        if start_date is None:
            start_date = data.index[0]

        if end_date is None:
            end_date = data.index[-1]

        assert start_date in data.index, "Start date not available in the data index"
        assert end_date in data.index, "End date not available in the data index"
        # -------------------------------------------------------------------------------------------------------

        self.rebalancing = rebalancing
        self.startdate = start_date
        self.enddate = end_date
        self.inputdata = data.loc[start_date:end_date]
        self.handle_missing = handle_missing

    def fred_compute_backtest(
        self, factors: np.array, initialweights: np.array, Tcosts: np.array
    ) -> pd.Series:

        """
        Wrapper function called by the user to run a historical backtest of a strategy consisting of the factors

        :param factors: array of data column names corresponding to the factor in the strategy
        :param weights: array of weights corresponding to each of the factors in the strategy
        :T-cost: array of T-costs corresponding to T-costs of trading each of the factors
        :return: pandas Series of statistics from the backtest
        """

        # Check that parameters are set correctly
        # -------------------------------------------------------------------------------------------------------

        assert len(factors) >= 1, "Must have at least 1 factor for a backtest"
        assert len(initialweights) == len(
            factors
        ), "Length of weights must match length of factors"
        assert len(Tcosts) == len(
            factors
        ), "Length of T-costs must match length of factors"

        for factor in factors:
            assert (
                factor in self.inputdata.columns.values
            ), f"{factor} does not exist in the data column names"

        assert sum(initialweights) == 1, "Sum of weights does not equal 1"

        for Tcost in Tcosts:
            assert Tcost >= 0 and Tcost <= 1, "Tcost must be between 0-1 for each asset"
        # -------------------------------------------------------------------------------------------------------

        initialweightsmap = {}
        Tcostsmap = {}

        for i, factor in enumerate(factors):
            initialweightsmap[factor] = initialweights[i]
            Tcostsmap[factor] = Tcosts[i]

            self.factors = factors
            self.initialweights = initialweightsmap
            self.Tcosts = Tcostsmap

        self._fillmissing()
        # compute returns
        self.returndata = self.returndata.pct_change()
        self._run_backtest()
        self._compute_stats()

        return self.stats

    def _is_rebal(self, date):
        """Uses index to check whether specified date is a rebalancing date
        :param: self.rebalancing
        return: boolean indicating whether the index corresponds to a rebalancing trigger"""

        if self.rebalancing == "monthly":
            return True

        elif self.rebalancing == "quarterly" and date % 3 == 0:
            return True

        elif self.rebalancing == "annually" and date % 12 == 0:
            return True

        return False

    def _fillmissing(self):
        """
        Fill missing values
        :param: handle_missing
        0: Forward Fill followed by Backward Fill missing values
        1: Fill missing values with mean of respective series
        :return: self.inputdata without missing values
        """

        if self.handle_missing == 0:
            self.inputdata = self.inputdata.ffill().bfill()

        elif self.handle_missing == 1:
            self.inputdata = self.inputdata.fillna(self.inputdata.mean())

    def _run_backtest(self):
        """Runs the backtest based on user specified weights and other input parameters
        :params: self.inputdata, self.rebalancing, self.initialweights, self.Tcosts
        :return: self.cumulativevalue, self.turnover

        Steps:
        For each date, compute the pre-rebalance dates basd on the corresponding returns
        Use the rebalancing trigger to convert pre-rebalance weights to post-rebalance weights
        If rebalancing occured, apply T costs to the turnover amounts
        Obtain final post-rebalance weights for the date
        """

        postrebal = {}
        prerebal = {}
        cumulativevalue = {}
        turnover = {}

        for asset in self.initialweights.keys():
            postrebal[asset] = {}
            prerebal[asset] = {}

        # set starting weights
        for asset in postrebal.keys():
            postrebal[asset][self.startdate] = self.initialweights[asset]

        # set initial starting value as 1
        cumulativevalue[self.startdate] = 1
        turnover[self.startdate] = 0

        alldates = self.inputdata.index

        for i in range(len(alldates) - 1):
            cumulativevalue[alldates[i + 1]] = 0
            turnover[alldates[i + 1]] = 0

            # apply returns
            for asset in prerebal.keys():
                prerebal[asset][alldates[i + 1]] = postrebal[asset][alldates[i]] * (
                    1 + self.returndata.loc[alldates[i]][asset]
                )
                cumulativevalue[alldates[i + 1]] += prerebal[asset][alldates[i + 1]]

            # apply rebalancing and turnover
            if self._is_rebal(i + 1):
                for asset in postrebal.keys():
                    postrebal[asset][alldates[i + 1]] = (
                        cumulativevalue[alldates[i + 1]] * self.initialweights[asset]
                    )
                    currentturnover = (
                        abs(
                            postrebal[asset][alldates[i + 1]]
                            - prerebal[asset][alldates[i + 1]]
                        )
                        * self.Tcosts[asset]
                    )
                    postrebal[asset][alldates[i + 1]] = (
                        postrebal[asset][alldates[i + 1]] - currentturnover
                    )
                    turnover[alldates[i + 1]] = (
                        turnover[alldates[i + 1]] + currentturnover
                    )

                turnover[alldates[i + 1]] = (
                    turnover[alldates[i + 1]] / cumulativevalue[alldates[i + 1]]
                )

            else:
                for asset in postrebal.keys():
                    postrebal[asset][alldates[i + 1]] = prerebal[asset][alldates[i + 1]]
                    turnover[alldates[i + 1]] = 0

            self.cumulativevalues = pd.Series(cumulativevalue)
            self.turnover = pd.Series(turnover)

    def _compute_stats(self):
        """Computes all relevant statistics from the backtest results
        :param: self.cumulativevalues and self.turnover
        :return: self.stats"""

        stats = {}

        # risk metrics:
        returns = self.cumulativevalues.pct_change()[1:]

        annualizedvol = np.std(returns) * np.sqrt(12)
        stats["Annualized Vol (%)"] = annualizedvol

        Roll_Max = self.cumulativevalues.cummax()
        Drawdown = self.cumulativevalues / Roll_Max - 1.0
        Max_Drawdown = Drawdown.cummin()[-1]
        stats["Max Drawdown (%)"] = Max_Drawdown
        stats["Max DD from Base"] = min(self.cumulativevalues) - 1
        var_95 = np.percentile(returns, 5)
        var_95 = var_95 * np.sqrt(12)
        stats["95% VaR"] = var_95

        # return metrics
        returns = self.cumulativevalues.pct_change()[1:]

        cumulativereturn = ((1 + returns).cumprod() - 1)[-1]
        stats["Cumulative Return"] = cumulativereturn
        N = len(returns) / 12
        annualizedreturn = (1 + cumulativereturn) ** (1 / N) - 1
        stats["Annualized Return (%)"] = annualizedreturn

        averageannualreturn = np.mean(returns) * 12
        stats["Average Annual Return (%)"] = averageannualreturn

        stats["Annualized Turnover (%)"] = np.mean(self.turnover) * 12

        sharpe = (averageannualreturn - 0) / annualizedvol
        stats["Annualized Sharpe"] = round(sharpe, 2)

        self.stats = pd.Series(stats)
