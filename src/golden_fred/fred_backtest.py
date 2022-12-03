import pandas as pd
import numpy as np
import datetime
import warnings
import sys
from typing import Union, Tuple
import datetime
import warnings
from typing import Union, Tuple
from scipy import sparse
import cvxpy as cp


class FredBacktest:
    def __init__(
        self,
        data: pd.DataFrame,
        start_date: Union[datetime.datetime, None] = None,
        end_date: Union[datetime.datetime, None] = None,
        rebalancing: str = "monthly",
        handle_missing: int = 1,
    ):
        """

        :param data: cleaned Fred Data Monthly Data outputted from GetFred()
        :param start_date: start date for backtest
        :param end_date: end date for backtest
        :param rebalancing: rebalancing technique for backtest. Options are: monthly, quarterly, yearly, ntz (trade zones)
        :param handle_missing: is an integer in [0,1] representing:
        0: Forward Fill followed by Backward Fill missing values
        1: Fill missing values with mean of respective series

        Main functions are:
        :fred_compute_backtest(): runs a backtest on a portfolio of FRED factors with weights and T costs provided by the user. Reports all backtest statistics of the strategy
        :regime_filtering(): runs a L1 trend filtering algorithm on each column of data. Reports historical regimes of contraction, expansion and transition.
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
        ], f"Rebalancing input {rebalancing=} not one of 'monthly', 'quarterly', 'annually'"
        assert isinstance(
            data.index, pd.DatetimeIndex
        ), "Index of input data must be a datetime index"

        if start_date is None:
            start_date = data.index[0]

        if end_date is None:
            end_date = data.index[-1]

        assert end_date > start_date, "End date must be after the start date"

        assert start_date in data.index, "Start date not available in the data index"
        assert end_date in data.index, "End date not available in the data index"
        # -------------------------------------------------------------------------------------------------------

        self.rebalancing = rebalancing
        self.startdate = start_date
        self.enddate = end_date
        self.inputdata = data.loc[start_date:end_date]
        self.handle_missing = handle_missing

        assert (
            len(self.inputdata) > 2
        ), "Must have at least 3 data points to compute statistics"

    def fred_compute_backtest(
        self, factors: np.array, initialweights: np.array, Tcosts: np.array
    ) -> pd.Series:

        """
        Wrapper function called by the user to run a historical backtest of a strategy consisting of the factors

        :param factors: array of data column names corresponding to the factors in the strategy
        :param weights: array of weights corresponding to each of the factors in the strategy
        :T-cost: array of T-costs corresponding to T-costs of trading each of the factors
        :return: pandas Series of statistics from the historical backtest
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
        self.factors = np.array(factors)
        self.initialweights = np.array(initialweights)
        self.Tcosts = np.array(Tcosts)

        self._fillmissing()

        # compute returns
        self.returndata = self.inputdata.pct_change().fillna(0)
        self._run_backtest()
        self._compute_stats()

        return self.stats

    def regime_filtering(
        self, columns: np.array, lambda_param: np.array = None
    ) -> pd.DataFrame:

        """
        Wrapper function called by the user to extract historical regimes.
        Runs a L1 trend-filtering algorithm

        :param columns: array of column names for regime filtering
        :param lambda_param: array of lambda regularization parameteres for L1 trend filtering
        : return Pandas DataFrame: containing a historical time series of [-1,1] for each column, corresponding to
        [contraction, expansion] regimes respectively
        """

        # Check that parameters are set correctly
        # -------------------------------------------------------------------------------------------------------
        if lambda_param is None:
            lambda_param = [10000] * len(columns)

        assert (
            len(columns) >= 1
        ), "Must have at least 1 column name for a historical regime backtest"
        assert len(lambda_param) == len(
            columns
        ), "Length of lambda parameters must match length of columns"

        for column in columns:
            assert (
                column in self.inputdata.columns.values
            ), f"{column} does not exist in the data column names"

        # -------------------------------------------------------------------------------------------------------
        self.columns = np.array(columns)
        self.lambdas = np.array(lambda_param)
        self._fillmissing()
        self._run_regimefiltering()

        return self.regimes

    def _is_rebal(self, date) -> bool:
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
        For each date, compute the pre-rebalance dates based on the corresponding returns
        Use the rebalancing trigger to convert pre-rebalance weights to post-rebalance weights
        If rebalancing occured, apply T costs to the turnover amounts
        Obtain final post-rebalance weights for the date
        """

        postrebal = {}
        prerebal = {}
        cumulativevalue = {}
        turnover = {}

        # set starting weights
        postrebal[self.startdate] = self.initialweights

        # set initial starting value as 1
        cumulativevalue[self.startdate] = 1
        turnover[self.startdate] = 0

        alldates = self.inputdata.index

        for i in range(len(alldates) - 1):
            cumulativevalue[alldates[i + 1]] = 0
            turnover[alldates[i + 1]] = 0

            # apply returns
            prerebal[alldates[i + 1]] = postrebal[alldates[i]] * (
                1 + self.returndata.loc[alldates[i + 1]][self.factors]
            )
            cumulativevalue[alldates[i + 1]] = np.sum(prerebal[alldates[i + 1]])

            # apply rebalancing and turnover
            if self._is_rebal(i + 1):
                postrebal[alldates[i + 1]] = (
                    cumulativevalue[alldates[i + 1]] * self.initialweights
                )
                currentturnover = (
                    abs(postrebal[alldates[i + 1]] - prerebal[alldates[i + 1]])
                    * self.Tcosts
                )
                postrebal[alldates[i + 1]] = (
                    postrebal[alldates[i + 1]] - currentturnover
                )
                turnover[alldates[i + 1]] = (
                    np.sum(currentturnover) / cumulativevalue[alldates[i + 1]]
                )

            else:
                postrebal[alldates[i + 1]] = prerebal[alldates[i + 1]]
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

        annualizedvol = returns.std() * np.sqrt(12)
        stats["Annualized Vol (%)"] = annualizedvol

        roll_max = self.cumulativevalues.cummax()
        drawdown = self.cumulativevalues / roll_max - 1.0
        max_drawdown = drawdown.cummin()[-1]
        stats["Max drawdown (%)"] = max_drawdown
        stats["Max DD from Base"] = min(self.cumulativevalues) - 1
        var_95 = np.percentile(returns, 5) * np.sqrt(12)
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


    def _run_regimefiltering(self):
        """
        Runs the L1 trend filtering algorithm to identify historical regimes of contraction (-1), and expansion (+1)
        Uses a piecewise linear function
        Lambda regularization parameter is specified by the user
        """
        regimes = None

        for i, column in enumerate(self.columns):
            # Set up L1 regime filter algorithm using a difference matrix
            n = len(self.inputdata[column])
            one_vec = np.ones((1, n))
            D = sparse.spdiags(
                np.vstack((one_vec, -2 * one_vec, one_vec)), range(3), m=n - 2, n=n
            ).toarray()  # spdiags(data, diags_to_set, m, n

            # run the L1 optimization using cvxpy
            y = self.inputdata[column].values
            lambd = self.lambdas[i]
            x = cp.Variable(n)
            obj = cp.Minimize(0.5 * cp.sum_squares(y - x) + lambd * cp.norm(D @ x, 1))
            prob = cp.Problem(obj)
            prob.solve()

            # identify +1, 0, and -1 regimes as regions of positive and negative slopes, and 0 as transition (if any)
            # piece-wise linear function
            slopes = np.diff(x.value)
            result = np.where(slopes < 0, -1, slopes)
            result = np.where(result > 0, 1, result)
            result = np.where(result == 0, 0, result)

            if regimes is None:
                regimes = result
            else:
                regimes = np.concatenate(([regimes], [result]), axis=0)

        regimes = pd.DataFrame(regimes.T, index=self.inputdata.index.values[1:])
        regimes.columns = self.columns
        self.regimes = regimes
