import abc
import dataclasses
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import statsmodels.api as sm
import pylab
import collections
from dateutil.relativedelta import relativedelta
import sklearn
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


@dataclasses.dataclass(frozen=True, eq=False)
class FredRegression(abc.ABC):
    """
    Base class for all regression methods.
    The user will call the method fit() which will fit the model
    and compute the test and train errors for the data.

    """

    def compute_bic(self):
        pass

    @abc.abstractmethod
    def get_error(self, y_pred, y_true):
        raise NotImplementedError

    @abc.abstractmethod
    def run_model(self):
        raise NotImplementedError

    @abc.abstractmethod
    def features_and_target(self):
        raise NotImplementedError

    @abc.abstractmethod
    def fit(self):
        raise NotImplementedError

    def _fill_missing_data(self):
        """
        NOTE : the operations are done IN-PLACE
        Fill missing values
        :param: handle_missing
        0: Forward Fill followed by Backward Fill missing values
        1: Fill missing values with mean of respective series
        default : 0

        """

        self.data.drop(index=self.data.index[[0, 1]], axis=0, inplace=True)
        self.original_input = self.data.copy()

        if self.handle_missing == 0:
            self.data = self.data.ffill().bfill()

        elif self.handle_missing == 1:
            self.data = self.data.fillna(self.data.mean())

    def plot_insample_and_outofsample_error(self):
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

        ax[0].plot(self.dates_tested, self.in_sample_error, "--o")
        ax[0].set_title("In sample error of {} Model".format(self.model_name))

        ax[1].plot(self.dates_tested, self.out_of_sample_error, "--o")
        ax[1].set_title("Out sample error of {} Model".format(self.model_name))

        plt.gcf().autofmt_xdate()


class AR_Model(FredRegression):
    def __init__(
        self,
        data,
        max_lag,
        start_date,
        end_date,
        dependent_variable_name,
        window_size,
        lag_patience=5,
        model_name="AR",
        handle_missing=0,
    ):
        """
        Fits the Auto-Regressive model on any time series data.

        :param data       : input time series dataframe, must contain a column with the user input value of
        dependent_variable_name. Data should have minimum [start_date - window_size to end_date] rows
        :param max_lag    : maximum number of lags for the AR model to be tested.
        :param start_date : first date model tests
        :param end_date   : last date model tests
        :param dependent_variable_name : variable to be predicted.
        :param window_size  : window size to use for the AR model
        :param lag_patience : Fitting AR models with higher orders take much longer and have
          convergence issues. We therefore use a heuristic to reduce the computatoin
          time. We keep monitoring the BIC value, and if the BIC value hasn't
          reduced for `lag_patience` number of orders, we stop the computation
          then and return the current lowest. We experimented with different
          `patience_thres` values and the results look qualitatively similar.
        :param model_name     : name of the model, used in plot_insample_and_outofsample_error()
        :param handle_missing : 0/1 - specifies how to handle missing data.

        """
        self.model_name = model_name

        self.data = data
        self.max_lag = max_lag
        self.start_date = start_date
        self.end_date = end_date
        self.window_size = window_size
        self.dependent_variable_name = dependent_variable_name
        self.lag_patience = lag_patience
        self.handle_missing = handle_missing

        self.out_of_sample_error = None
        self.in_sample_error = None
        self.dates_tested = None
        self.lag_from_ar_model = None

    def features_and_target(self):
        """
        splits the data into features and target based on
        the dependent_variable_name that user inputs
        """
        self.features = self.data.drop(self.dependent_variable_name, axis=1)
        self.target = self.data[self.dependent_variable_name]

    def create_lagged_data(self, data, max_lag, is_series=False):

        lagged = []
        for lag in range(1, max_lag + 1):
            lagged.append(data.shift(lag))
        if is_series:
            return np.stack(lagged).T
        else:
            return np.concatenate(lagged, axis=1)

    def get_error(self, y_pred, y_true):
        """returns the mean_squared_error"""
        return np.mean((y_pred - y_true) ** 2)

    def compute_bic(self, model):
        return model.bic

    def find_best_AR_model(self, series_data):
        """
        Find the best lags based on BIC computed.
        """

        best_bic = np.inf
        series_data = series_data.dropna()  # drop any na it has

        for lag in range(1, self.max_lag + 1):

            curr_features = self.create_lagged_data(
                series_data, max_lag=lag, is_series=True
            )[
                lag:,
            ]
            curr_target = series_data[lag:]

            # pick the last one as out-of-sample
            in_sample_y = curr_target[:-1].values.reshape(-1)
            in_sample_x = curr_features[:-1, :]
            out_sample_y = curr_target[-1]
            out_sample_x = curr_features[-1, :].reshape(1, -1)

            # add bias terms to features
            in_sample_x = sm.add_constant(in_sample_x)
            out_sample_x = sm.add_constant(out_sample_x, has_constant="add")

            # fit model
            ar_model = sm.OLS(in_sample_y, in_sample_x).fit()
            in_sample_y_pred = ar_model.predict(in_sample_x)
            out_sample_y_pred = ar_model.predict(out_sample_x)
            in_sample_error = self.get_error(in_sample_y_pred, in_sample_y)
            out_sample_error = self.get_error(out_sample_y_pred, out_sample_y)

            curr_bic = self.compute_bic(ar_model)

            if curr_bic < best_bic:
                best_bic = curr_bic
                best_lag = lag

            if lag - best_lag > self.lag_patience:
                return best_lag, in_sample_error, out_sample_error

        return best_lag, in_sample_error, out_sample_error

    def run_model(self):

        self.features_and_target()

        out_err_AR = []
        in_err_AR = []
        order_array_AR = []
        dates_tested = []

        date = self.start_date

        while date < self.end_date:
            dates_tested.append(date)
            curr_window_target = self.target[self.target.index <= date][
                -self.window_size :
            ]
            best_lag, in_sample_error, out_sample_error = self.find_best_AR_model(
                series_data=curr_window_target
            )

            out_err_AR.append(out_sample_error)
            in_err_AR.append(in_sample_error)
            order_array_AR.append(best_lag)

            date = date + relativedelta(months=1)

        return out_err_AR, in_err_AR, order_array_AR, dates_tested

    def fit(self):
        out_err, in_err, order_array, dates_tested = self.run_model()
        self.out_of_sample_error = out_err
        self.in_sample_error = in_err
        self.lag_from_ar_model = order_array
        self.dates_tested = dates_tested


class Regularised_Regression_Model(FredRegression):
    """
    Regularised Linear regression
    Please specify 'Ridge' or 'Lasso' as regularisation_type. Default is 'Ridge' (L2 regularisation)
    """

    def __init__(
        self,
        data,
        start_date,
        end_date,
        dependent_variable_name,
        model_lags,
        regularisation_type="Ridge",
        window_size=492,
        handle_missing=0,
        lambdas=np.logspace(-2, 1, 4),
    ):
        """
        :param model_lags : if not specified, we use lags from AR_model as the optimum lag.
        :param lambdas : lamba values to check for the model.
        we check 4 values as default : [ 0.01,  0.1 ,  1.  , 10.  ]
        """

        self.model_name = regularisation_type

        self.data = data
        self.start_date = start_date
        self.end_date = end_date
        self.window_size = window_size
        self.dependent_variable_name = dependent_variable_name
        self.model_lags = model_lags
        self.regularisation_type = regularisation_type
        self.handle_missing = handle_missing
        self.lambdas = lambdas

        self.out_of_sample_error = None
        self.in_sample_error = None
        self.dates_tested = None
        self.model_coef = None

    def compute_bic(self, num_samples, degrees, squared_errors):
        sigma_square = np.sum(squared_errors) / (num_samples - degrees)
        return np.log(sigma_square) + degrees * np.log(num_samples) / num_samples

    def compute_bic_ridge(self, training_data, training_errors, lam):

        # Get degrees of freedom using long formula

        res = np.matmul(training_data.T, training_data)
        res = res + np.identity(res.shape[0]) * lam
        res = np.linalg.inv(res)
        res = np.matmul(np.matmul(training_data, res), training_data.T)
        degrees = np.trace(res)

        return self.compute_bic(len(training_data), degrees, training_errors)

    def compute_bic_lasso(self, training_data, training_errors, coeffs):

        # get degrees of freedom as the number of non-zero coeffs
        degrees = np.sum(np.abs(coeffs) > 0)

        return self.compute_bic(len(training_data), degrees, training_errors)

    def _model_regularisation(self):
        return {
            "Ridge": sklearn.linear_model.Ridge,
            "Lasso": sklearn.linear_model.Lasso,
        }[self.regularisation_type]

    def _bic_regularisation(self):
        return {"Ridge": self.compute_bic_ridge, "Lasso": self.compute_bic_lasso}[
            self.regularisation_type
        ]

    def create_lagged_data(self, data, max_lag=4, is_series=False):

        lagged = []
        for lag in range(1, max_lag + 1):
            lagged.append(data.shift(lag))
        if is_series:
            return np.stack(lagged).T
        else:
            return np.concatenate(lagged, axis=1)

    def get_error(self, y_pred, y_true):
        """returns the mean_squared_error"""
        return np.mean((y_pred - y_true) ** 2)

    def find_best_model(self, series_data, features, lag_from_ar_model):

        best_bic = np.inf
        best_alpha = 0

        lag_features = self.create_lagged_data(
            series_data, max_lag=lag_from_ar_model, is_series=True
        )[
            lag_from_ar_model:,
        ]
        data_features = self.create_lagged_data(
            features, max_lag=lag_from_ar_model, is_series=False
        )[
            lag_from_ar_model:,
        ]

        curr_target = series_data[lag_from_ar_model:]

        curr_features = np.concatenate([lag_features, data_features], axis=1)
        curr_features = scale(curr_features)

        # pick the last one as out-of-sample
        in_sample_y = curr_target[:-1].values.reshape(-1)
        in_sample_x = curr_features[:-1, :]
        out_sample_y = curr_target[-1]
        out_sample_x = curr_features[-1, :].reshape(1, -1)

        for lam in self.lambdas:
            model = self._model_regularisation()(lam)

            model.fit(in_sample_x, in_sample_y)

            in_sample_y_pred = model.predict(in_sample_x)
            out_sample_y_pred = model.predict(out_sample_x)

            curr_bic = self._bic_regularisation()(
                in_sample_x, (in_sample_y_pred - in_sample_y) ** 2, lam
            )

            if curr_bic < best_bic:
                best_bic = curr_bic
                best_lam = lam
                best_model_coeffs = model.coef_
                in_sample_error = self.get_error(in_sample_y_pred, in_sample_y)
                out_sample_error = self.get_error(out_sample_y_pred, out_sample_y)

        return in_sample_error, out_sample_error, best_model_coeffs

    def features_and_target(self):
        """
        splits the data into features and target based on
        the dependent_variable_name that user inputs
        """
        self.features = self.data.drop(self.dependent_variable_name, axis=1)
        self.target = self.data[self.dependent_variable_name]

    def run_model(self):

        self._fill_missing_data()
        self.features_and_target()

        out_err = []
        in_err = []
        model_coef = []
        dates_tested = []

        date = self.start_date
        idx = 0
        while date < self.end_date:
            dates_tested.append(date)
            curr_window_target = self.target[self.target.index < date][
                -self.window_size :
            ]
            curr_features = self.features[self.features.index < date][
                -self.window_size :
            ]
            in_sample_error, out_sample_error, model_coeffs = self.find_best_model(
                series_data=curr_window_target,
                features=curr_features,
                lag_from_ar_model=self.model_lags[idx],
            )
            out_err.append(out_sample_error)
            in_err.append(in_sample_error)
            model_coef.append(model_coeffs)

            date = date + relativedelta(months=1)
            idx += 1

        return out_err, in_err, dates_tested, model_coef

    def fit(self):
        out_err, in_err, dates_tested, model_coef = self.run_model()
        self.out_of_sample_error = out_err
        self.in_sample_error = in_err
        self.dates_tested = dates_tested
        self.model_coef = model_coef


class Neural_Network(FredRegression):
    def __init__(
        self,
        data,
        start_date,
        end_date,
        dependent_variable_name,
        model_lags,
        hidden_layer_sizes,
        model_name="neural_network",
        window_size=492,
        max_iter=1000,
        activation="relu",
        handle_missing=0,
    ):

        self.model_name = model_name

        self.data = data
        self.start_date = start_date
        self.end_date = end_date
        self.window_size = window_size
        self.dependent_variable_name = dependent_variable_name
        self.model_lags = model_lags
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.activation = activation
        self.handle_missing = handle_missing

        self.out_of_sample_error = None
        self.in_sample_error = None
        self.dates_tested = None

    def features_and_target(self):
        """
        splits the data into features and target based on
        the dependent_variable_name that user inputs
        """
        self.features = self.data.drop(self.dependent_variable_name, axis=1)
        self.target = self.data[self.dependent_variable_name]

    def create_lagged_data(self, data, max_lag=4, is_series=False):
        lagged = []
        for lag in range(1, max_lag + 1):
            lagged.append(data.shift(lag))
        if is_series:
            return np.stack(lagged).T
        else:
            return np.concatenate(lagged, axis=1)

    def get_error(self, y_pred, y_true):
        """returns the mean_squared_error"""
        return np.mean((y_pred - y_true) ** 2)

    def neural_network_model(self, series_data, features, lag_from_ar_model):

        lag_features = self.create_lagged_data(
            series_data, max_lag=lag_from_ar_model, is_series=True
        )[
            lag_from_ar_model:,
        ]
        data_features = self.create_lagged_data(
            features, max_lag=lag_from_ar_model, is_series=False
        )[
            lag_from_ar_model:,
        ]

        curr_target = series_data[lag_from_ar_model:]

        curr_features = np.concatenate([lag_features, data_features], axis=1)
        curr_features = scale(curr_features)

        scaler_X = StandardScaler()
        X = scaler_X.fit_transform(curr_features)

        scaler_y = StandardScaler()
        y = scaler_y.fit_transform(curr_target.values.reshape(-1, 1))

        # pick the last one as out-of-sample
        in_sample_y = y[:-1]
        in_sample_x = X[:-1, :]
        out_sample_y = y[-1]
        out_sample_x = X[-1, :]

        model = MLPRegressor(
            random_state=1,
            hidden_layer_sizes=self.hidden_layer_sizes,
            max_iter=self.max_iter,
            activation=self.activation,
        )

        model = model.fit(in_sample_x, in_sample_y)

        test_pred_y = model.predict([out_sample_x])
        out_sample_y_pred = scaler_y.inverse_transform(test_pred_y.reshape(-1, 1))[0][0]

        train_pred_y = model.predict(in_sample_x)
        in_sample_y_pred = scaler_y.inverse_transform(train_pred_y.reshape(-1, 1))[0][0]

        in_sample_error = self.get_error(in_sample_y_pred, in_sample_y)
        out_sample_error = self.get_error(out_sample_y_pred, out_sample_y)

        return in_sample_error, out_sample_error

    def run_model(self):

        self._fill_missing_data()
        self.features_and_target()

        out_err = []
        in_err = []
        dates_tested = []

        date = self.start_date
        idx = 0
        while date < self.end_date:
            dates_tested.append(date)
            curr_window_target = self.target[self.target.index < date][
                -self.window_size :
            ]

            curr_features = self.features[self.features.index < date][
                -self.window_size :
            ]
            in_sample_error, out_sample_error = self.neural_network_model(
                series_data=curr_window_target,
                features=curr_features,
                lag_from_ar_model=self.model_lags[idx],
            )
            out_err.append(out_sample_error)
            in_err.append(in_sample_error)

            date = date + relativedelta(months=1)
            idx += 1

        return out_err, in_err, dates_tested

    def fit(self):
        out_err, in_err, dates_tested = self.run_model()
        self.out_of_sample_error = out_err
        self.in_sample_error = in_err
        self.dates_tested = dates_tested
