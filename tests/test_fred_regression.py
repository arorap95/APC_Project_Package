import pytest
import unittest
import numpy as np
import pandas as pd
import fred_regression
import get_fred
from golden_fred import fred_regression
from golden_fred import get_fred
import datetime
from parameterized import parameterized


class test_AR_Model(unittest.TestCase):
    def c(
        self,
        data=None,
        max_lag=1,
        start_date=pd.to_datetime("2020-01"),
        end_date=pd.to_datetime("2020-05"),
        dependent_variable_name="please_define",
        window_size=2,
        handle_missing=0,
    ):

        AR_model = fred_regression.AR_Model

        return AR_model(
            data=data,
            max_lag=max_lag,
            start_date=start_date,
            end_date=end_date,
            dependent_variable_name=dependent_variable_name,
            window_size=window_size,
            handle_missing=handle_missing,
        )

    def create_input(self, use_Fred_data=False, test_missing=False):

        if test_missing:
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

            df = pd.DataFrame({"Date": x, "column1": y, "column2": z})
            df.set_index("Date", inplace=True)

            return df

        elif use_Fred_data:
            data = get_fred.GetFred()
            df = data.get_fred_md()

        else:

            df = pd.DataFrame(
                columns=["Date", "col1", "col2", "result"],
                data=[
                    [pd.to_datetime("2010-01"), 1, 2, 4],
                    [pd.to_datetime("2010-02"), 2, 2, 6],
                    [pd.to_datetime("2010-03"), 3, 1, 7],
                    [pd.to_datetime("2010-04"), 1, 0, 2],
                    [pd.to_datetime("2010-05"), 5, 2, 10],
                    [pd.to_datetime("2010-06"), 4, 2, 10],
                ],
            )
        return df

    def test_class_initialisation(self):
        df = self.create_input()
        model = self.c(
            data=df,
            max_lag=3,
            start_date=min(df.Date),
            end_date=max(df.Date),
            dependent_variable_name="result",
        )

        self.assertEqual(model.start_date, pd.to_datetime("2010-01"))
        self.assertEqual(model.end_date, pd.to_datetime("2010-06"))
        self.assertEqual(model.dependent_variable_name, "result")
        self.assertEqual(model.lag_patience, 5)
        self.assertEqual(model.window_size, 2)
        self.assertEqual(model.handle_missing, 0)
        self.assertEqual(model.max_lag, 3)

    def test_features_and_target(self):

        df = self.create_input()

        model = self.c(dependent_variable_name="result", data=df)

        model.features_and_target()

        self.assertEqual(len(model.target), 6)
        self.assertEqual(len(model.features.axes[1]), 3)

    def test_get_error(self):
        model = self.c()
        err = model.get_error(np.array([1, 2]), np.array([0, 1]))
        self.assertEqual(err, 1)

    @parameterized.expand([["0", 0], ["1", 1]])
    def test_handle_missing(self, name, val):
        df = self.create_input(test_missing=True)
        model = self.c(data=df, handle_missing=val)

        model._fill_missing_data()
        assert df.notnull().values.any()

    def test_fit(self):
        df = self.create_input(use_Fred_data=True)
        start = pd.to_datetime("2010-03")
        end = pd.to_datetime("2011-03")
        model = self.c(
            dependent_variable_name="CPIAUCSL",
            data=df,
            start_date=start,
            end_date=end,
            max_lag=5,
            window_size=100,
        )
        model.fit()
        nMonths = round((end - start) / np.timedelta64(1, "M"))
        self.assertEqual(len(model.in_sample_error), nMonths)
        self.assertEqual(len(model.out_of_sample_error), nMonths)
        self.assertEqual(len(model.lag_from_ar_model), nMonths)
        self.assertEqual(len(model.dates_tested), nMonths)
        self.assertEqual(len(model.predicted), nMonths)
        self.assertEqual(len(model.targets), nMonths)


class test_Regularised_Regression_Model(unittest.TestCase):
    def c(
        self,
        data=None,
        regularisation_type="please_specify",
        start_date=pd.to_datetime("2020-01"),
        end_date=pd.to_datetime("2020-05"),
        dependent_variable_name="please_define",
        window_size=2,
        handle_missing=0,
        lambdas=[0.01],
        model_lags=[1, 2],
    ):

        Regularised_Regression_Model = fred_regression.Regularised_Regression_Model

        return Regularised_Regression_Model(
            data=data,
            start_date=start_date,
            end_date=end_date,
            dependent_variable_name=dependent_variable_name,
            window_size=window_size,
            handle_missing=handle_missing,
            regularisation_type=regularisation_type,
            model_lags=model_lags,
            lambdas=lambdas,
        )

    def create_input(self, use_Fred_data=False, test_missing=False):

        if test_missing:
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

            df = pd.DataFrame({"Date": x, "column1": y, "column2": z})
            df.set_index("Date", inplace=True)

            return df

        elif use_Fred_data:
            data = get_fred.GetFred()
            df = data.get_fred_md()

        else:

            df = pd.DataFrame(
                columns=["Date", "col1", "col2", "result"],
                data=[
                    [pd.to_datetime("2010-01"), 1, 2, 4],
                    [pd.to_datetime("2010-02"), 2, 2, 6],
                    [pd.to_datetime("2010-03"), 3, 1, 7],
                    [pd.to_datetime("2010-04"), 1, 0, 2],
                    [pd.to_datetime("2010-05"), 5, 2, 10],
                    [pd.to_datetime("2010-06"), 4, 2, 10],
                ],
            )
        return df

    @parameterized.expand([["ridge", "Ridge"], ["lasso", "Lasso"]])
    def test_class_initialisation(self, name, reg_type):
        df = self.create_input()
        model = self.c(
            data=df,
            start_date=min(df.Date),
            end_date=max(df.Date),
            dependent_variable_name="result",
            regularisation_type=reg_type,
        )

        self.assertEqual(model.start_date, pd.to_datetime("2010-01"))
        self.assertEqual(model.end_date, pd.to_datetime("2010-06"))
        self.assertEqual(model.dependent_variable_name, "result")
        self.assertEqual(model.window_size, 2)
        self.assertEqual(model.handle_missing, 0)
        self.assertEqual(model.regularisation_type, reg_type)
        self.assertEqual(len(model.model_lags), 2)
        self.assertEqual(len(model.lambdas), 1)

    def test_features_and_target(self):

        df = self.create_input()

        model = self.c(dependent_variable_name="result", data=df)

        model.features_and_target()

        self.assertEqual(len(model.target), 6)
        self.assertEqual(len(model.features.axes[1]), 3)

    def test_get_error(self):
        model = self.c()
        err = model.get_error(np.array([1, 2]), np.array([0, 1]))
        self.assertEqual(err, 1)

    @parameterized.expand([["0", 0], ["1", 1]])
    def test_handle_missing(self, name, val):
        df = self.create_input(test_missing=True)
        model = self.c(data=df, handle_missing=val)

        model._fill_missing_data()
        assert df.notnull().values.any()

    @parameterized.expand([["ridge", "Ridge"], ["lasso", "Lasso"]])
    def test_fit(self, name, reg_type):
        df = self.create_input(use_Fred_data=True)
        start = pd.to_datetime("2010-03")
        end = pd.to_datetime("2011-03")
        nMonths = round((end - start) / np.timedelta64(1, "M"))
        lag = 4
        ncols = len(df.axes[1])
        model = self.c(
            dependent_variable_name="CPIAUCSL",
            data=df,
            start_date=start,
            end_date=end,
            window_size=100,
            regularisation_type=reg_type,
            model_lags=[lag] * nMonths,
        )
        model.fit()

        self.assertEqual(len(model.in_sample_error), nMonths)
        self.assertEqual(len(model.out_of_sample_error), nMonths)
        self.assertEqual(len(model.model_coef), nMonths)
        self.assertEqual(len(model.dates_tested), nMonths)
        self.assertEqual(len(model.model_coef[0]), ncols * lag)
        self.assertEqual(len(model.predicted), nMonths)
        self.assertEqual(len(model.targets), nMonths)


class test_Neural_Network(unittest.TestCase):
    def c(
        self,
        data=None,
        start_date=pd.to_datetime("2020-01"),
        end_date=pd.to_datetime("2020-05"),
        dependent_variable_name="please_define",
        hidden_layer_sizes="please_specify",
        activation="please_specify",
        max_iter="please_specify",
        window_size=2,
        handle_missing=0,
        model_lags=[1, 2],
    ):

        nnet = fred_regression.Neural_Network

        return nnet(
            data=data,
            start_date=start_date,
            end_date=end_date,
            dependent_variable_name=dependent_variable_name,
            window_size=window_size,
            activation=activation,
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            handle_missing=handle_missing,
            model_lags=model_lags,
        )

    def create_input(self, use_Fred_data=False, test_missing=False):

        if test_missing:
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

            df = pd.DataFrame({"Date": x, "column1": y, "column2": z})
            df.set_index("Date", inplace=True)

            return df

        elif use_Fred_data:
            data = get_fred.GetFred()
            df = data.get_fred_md()

        else:

            df = pd.DataFrame(
                columns=["Date", "col1", "col2", "result"],
                data=[
                    [pd.to_datetime("2010-01"), 1, 2, 4],
                    [pd.to_datetime("2010-02"), 2, 2, 6],
                    [pd.to_datetime("2010-03"), 3, 1, 7],
                    [pd.to_datetime("2010-04"), 1, 0, 2],
                    [pd.to_datetime("2010-05"), 5, 2, 10],
                    [pd.to_datetime("2010-06"), 4, 2, 10],
                ],
            )
        return df

    @parameterized.expand([["0", "relu", 100], ["1", "tanh", 10]])
    def test_class_initialisation(self, name, activation, max_iter):
        df = self.create_input()
        model = self.c(
            data=df,
            start_date=min(df.Date),
            end_date=max(df.Date),
            dependent_variable_name="result",
            activation=activation,
            hidden_layer_sizes=(1, 2),
            max_iter=max_iter,
        )

        self.assertEqual(model.start_date, pd.to_datetime("2010-01"))
        self.assertEqual(model.end_date, pd.to_datetime("2010-06"))
        self.assertEqual(model.dependent_variable_name, "result")
        self.assertEqual(model.window_size, 2)
        self.assertEqual(model.handle_missing, 0)
        self.assertEqual(model.activation, activation)
        self.assertEqual(model.max_iter, max_iter)
        self.assertEqual(len(model.hidden_layer_sizes), 2)

    def test_features_and_target(self):

        df = self.create_input()

        model = self.c(dependent_variable_name="result", data=df)

        model.features_and_target()

        self.assertEqual(len(model.target), 6)
        self.assertEqual(len(model.features.axes[1]), 3)

    def test_get_error(self):
        model = self.c()
        err = model.get_error(np.array([1, 2]), np.array([0, 1]))
        self.assertEqual(err, 1)

    @parameterized.expand([["0", 0], ["1", 1]])
    def test_handle_missing(self, name, val):
        df = self.create_input(test_missing=True)
        model = self.c(data=df, handle_missing=val)

        model._fill_missing_data()
        assert df.notnull().values.any()

    @parameterized.expand([["0", "relu", 100, (10, 4)], ["1", "logistic", 150, (4, 3)]])
    def test_fit(self, name, activation, max_iter, hidden):
        df = self.create_input(use_Fred_data=True)
        start = pd.to_datetime("2010-03")
        end = pd.to_datetime("2011-03")
        nMonths = round((end - start) / np.timedelta64(1, "M"))
        lag = 4
        ncols = len(df.axes[1])
        model = self.c(
            dependent_variable_name="CPIAUCSL",
            data=df,
            start_date=start,
            end_date=end,
            window_size=100,
            activation=activation,
            hidden_layer_sizes=hidden,
            max_iter=max_iter,
            model_lags=[lag] * nMonths,
        )
        model.fit()

        self.assertEqual(len(model.in_sample_error), nMonths)
        self.assertEqual(len(model.out_of_sample_error), nMonths)
        self.assertEqual(len(model.dates_tested), nMonths)
        self.assertEqual(len(model.predicted), nMonths)
        self.assertEqual(len(model.targets), nMonths)


if __name__ == "__main__":
    unittest.main()
