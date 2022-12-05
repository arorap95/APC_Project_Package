import dataclasses
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.preprocessing import scale
import statsmodels.api as sm
import pylab
import collections
from dateutil.relativedelta import relativedelta
import sklearn
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.tsatools import lagmat
from typing import Union, Tuple, List, Optional
import datetime
from golden_fred import fred_factors


class FredRegression:
    """
    Base class for all regression methods.
    The user will call the method fit()
    """

    def __init__(
        self,
        data: pd.DataFrame,
        start_date: Union[datetime.datetime, None],
        end_date: Union[datetime.datetime, None],
        dependent_variable_name: str,
        window_size: int,
        model_name: str,
        handle_missing: int = 0,
        frequency: str = "monthly",
        use_pca_features=False,
        fred_factors_kwargs: dict = None,
        model_lags: Optional[List[int]] = None,
    ):

        """
        :param data       : input time series dataframe, must contain a column with the user input value of
        dependent_variable_name. Data should have minimum [start_date - window_size to end_date] rows
        :param start_date : first date model tests
        :param end_date   : last date model tests
        :param dependent_variable_name : variable to be predicted.
        :param window_size  : window size to use for the model
        :param model_name : user input name of the model
        :param handle_missing : 0/1 - specifies how to handle missing data, default 0
        :param frequency : specifies if the data is monthly or qaurterly, default 'monthly'
        :param use_pca_features : default False, if True :  runs AR+PCR regression
        :param fred_factors_kwargs: required if use_pca_factors = True. Parameters required to instantiate class FredFactors
        :param model_lags : required if use_pca_factors = True. Lags to use for each window.
        """

        # Check that parameters are set correctly
        # -------------------------------------------------------------------------------------------------------
        assert start_date >= min(
            data.index.values
        ), "start_date provided not in range of data"
        assert end_date <= max(
            data.index.values
        ), "end_date provided not in range of data"
        assert dependent_variable_name in list(
            data
        ), "dependent_variable_name should be a column in input_data"
        assert handle_missing in [
            0,
            1,
        ], "Handle Missing parameter must be an integer in [0,1]"
        assert frequency in [
            "monthly",
            "quarterly",
        ], "frequency should be monthly or quarterly"
        if use_pca_features:
            assert (
                fred_factors_kwargs != None
            ), "Please specify arguments for fred_factors"
            assert (
                len(fred_factors_kwargs) == 4
            ), "Please specify all the 4 required arugments - standardization,maxfactors,factorselection,removeoutliers. Please refer to documentaion of class FredFactors in golden_fred/fred_factors for details."
            assert model_lags != None, "Please provide lags for each window"
        if model_lags:
            assert len(model_lags) == round(
                (end_date - start_date) / np.timedelta64(1, "M")
            ), "length of model_lags provided should match the number of windows"

        # -------------------------------------------------------------------------------------------------------

        self.model_name = model_name
        self.data = data
        self.start_date = start_date
        self.end_date = end_date
        self.window_size = window_size
        self.dependent_variable_name = dependent_variable_name
        self.handle_missing = handle_missing
        self.frequency = frequency
        self.use_pca_features = use_pca_features
        self.fred_factors_kwargs = fred_factors_kwargs
        self.model_lags = model_lags

        self.month_increment = {"monthly": 1, "quarterly": 3}[self.frequency]

        self.out_of_sample_error = []
        self.in_sample_error = []
        self.dates_tested = []
        self.true = []
        self.predicted = []

    def compute_bic(self):
        pass

    def _model(self):
        """
        Implementation specific to the model
        """
        raise NotImplementError

    def run_model(self):
        "calls _run_model() with required params, implemented in each sub class"
        raise NotImplementError

    def fit(self):
        """
        user calls the function fit() to run the model.
        """
        self.run_model()

    def _features_and_target(self):
        """
        splits the data into features and target based on
        the dependent_variable_name that user inputs
        """
        self.features = self.data.drop(self.dependent_variable_name, axis=1)
        self.target = self.data[self.dependent_variable_name]

        if self.use_pca_features:
            self.fred_factors_kwargs["inputdata"] = self.features
            self.fred_factors_kwargs["handle_missing"] = self.handle_missing
            factors = fred_factors.FredFactors(**self.fred_factors_kwargs)
            self.features = factors.get_fred_factors()
            self.target = self.target[2:]

    def _create_lagged_data(self, data, max_lag, is_series=False):
        """
        appends lags from 1 to max_lag(including) to the original data.
        """
        lagged = []
        for lag in range(1, max_lag + 1):
            lagged.append(data.shift(lag))
        if is_series:
            return np.stack(lagged).T
        else:
            return np.concatenate(lagged, axis=1)

    def _lagged_features_and_target(self, series_data, features, lag):
        """
        creates lag_features, data_features and concatenates both to create curr_features
        """

        self.lag_features = self._create_lagged_data(
            series_data, max_lag=lag, is_series=True
        )[
            lag:,
        ]
        self.data_features = self._create_lagged_data(
            features, max_lag=lag, is_series=False
        )[
            lag:,
        ]

        if self.use_pca_features:
            self.data_features = features[lag:]

        self.curr_target = series_data[lag:]
        curr_features = np.concatenate([self.lag_features, self.data_features], axis=1)
        self.curr_features = scale(curr_features)

    def get_error(self, y_pred, y_true):
        """returns the mean_squared_error"""
        return np.mean((y_pred - y_true) ** 2)

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

    def _run_model(self, use_lags=None):

        self._fill_missing_data()
        self._features_and_target()
        date = self.start_date
        idx = 0
        kwargs = {}
        while date < self.end_date:
            self.dates_tested.append(date)
            curr_window_target = self.target[self.target.index < date][
                -self.window_size :
            ]

            curr_features = self.features[self.features.index < date][
                -self.window_size :
            ]
            kwargs["series_data"] = curr_window_target
            kwargs["features"] = curr_features
            if use_lags:
                kwargs["lag"] = use_lags[idx]

            self._model(**kwargs)
            date = date + relativedelta(months=self.month_increment)
            idx += 1

    def plot_insample_and_outofsample_error(self):
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

        ax[0].plot(self.dates_tested, self.in_sample_error, "--o")
        ax[0].set_title("In sample error of {} model".format(self.model_name))

        ax[1].plot(self.dates_tested, self.out_of_sample_error, "--o")
        ax[1].set_title("Out sample error of {} model".format(self.model_name))

        ax[2].plot(self.dates_tested, self.true, "g.", label="True")
        ax[2].plot(self.dates_tested, self.predicted, "r.", label="Predicted")
        ax[2].set_title(
            "True and predicted values of of {} model".format(self.model_name)
        )
        ax[2].legend(loc="best")

        plt.gcf().autofmt_xdate()


class AR_Model(FredRegression):
    def __init__(
        self,
        data: pd.DataFrame,
        start_date: Union[datetime.datetime, None],
        end_date: Union[datetime.datetime, None],
        dependent_variable_name: str,
        window_size: int,
        max_lag: int,
        model_name: str = "AR",
        handle_missing: int = 0,
        find_ideal_lags: bool = True,
        frequency: str = "monthly",
        use_pca_features: bool = False,
        fred_factors_kwargs: dict = None,
        model_lags=None,
        lag_patience: int = 5,
    ):
        """
        Fits the Auto-Regressive model on FRED data. Also fits AR+PCR model is use_pca_features = True.
        Class specific parameters:

        :param lag_patience : Fitting AR models with higher orders take much longer and have
          convergence issues. We therefore use a heuristic to reduce the computatoin
          time. We keep monitoring the BIC value, and if the BIC value hasn't
          reduced for 'lag_patience' number of orders, we stop the computation
          then and return the current lowest. We experimented with different
          'patience_thres' values and the results look qualitatively similar.

        Please refer to base class documentation for details of other params.

        """
        # Check that parameters are set correctly
        # -------------------------------------------------------------------------------------------------------
        if find_ideal_lags == False:
            assert (model_lags) != None, "please provide the lags for each window"
        # -------------------------------------------------------------------------------------------------------

        super().__init__(
            model_name=model_name,
            data=data,
            start_date=start_date,
            end_date=end_date,
            window_size=window_size,
            dependent_variable_name=dependent_variable_name,
            handle_missing=handle_missing,
            frequency=frequency,
            use_pca_features=use_pca_features,
            fred_factors_kwargs=fred_factors_kwargs,
            model_lags=model_lags,
        )

        self.max_lag = max_lag
        self.lag_patience = lag_patience
        self.lag_from_ar_model = []
        self.find_ideal_lags = find_ideal_lags

    def compute_bic(self, model):
        return model.bic

    def _split_train_test_fit(self, series_data, features, lag):
        # pick the last one as out-of-sample

        self._lagged_features_and_target(series_data, features, lag)
        if self.use_pca_features:
            self.features_to_use = self.curr_features
        else:
            self.features_to_use = self.lag_features

        in_sample_y = self.curr_target[:-1].values.reshape(-1)
        in_sample_x = self.lag_features[:-1, :]
        out_sample_y = self.curr_target[-1]
        out_sample_x = self.lag_features[-1, :].reshape(1, -1)

        # add bias terms to features
        in_sample_x = sm.add_constant(in_sample_x)
        out_sample_x = sm.add_constant(out_sample_x, has_constant="add")

        # fit model
        ar_model = sm.OLS(in_sample_y, in_sample_x).fit()
        in_sample_y_pred = ar_model.predict(in_sample_x)
        out_sample_y_pred = ar_model.predict(out_sample_x)
        in_sample_error = self.get_error(in_sample_y_pred, in_sample_y)
        out_sample_error = self.get_error(out_sample_y_pred, out_sample_y)

        return (
            ar_model,
            out_sample_y_pred,
            in_sample_error,
            out_sample_error,
            out_sample_y,
        )

    def _model(self, series_data, features, lag=None):
        """
        Find the best lags based on BIC computed.
        """
        if self.find_ideal_lags:
            best_bic = np.inf
            for lag in range(1, self.max_lag + 1):
                (
                    ar_model,
                    out_sample_y_pred,
                    in_sample_error,
                    out_sample_error,
                    out_sample_y,
                ) = self._split_train_test_fit(series_data, features, lag)
                curr_bic = self.compute_bic(ar_model)

                if curr_bic < best_bic:
                    best_bic = curr_bic
                    best_lag = lag

                if lag - best_lag > self.lag_patience:
                    break
            self.lag_from_ar_model.append(best_lag)

        else:
            (
                ar_model,
                out_sample_y_pred,
                in_sample_error,
                out_sample_error,
                out_sample_y,
            ) = self._split_train_test_fit(series_data, features, lag)

        self.out_of_sample_error.append(out_sample_error)
        self.in_sample_error.append(in_sample_error)
        self.true.append(out_sample_y)
        self.predicted.append(out_sample_y_pred[0])

    def run_model(self):
        if self.find_ideal_lags:
            self._run_model()
        else:
            self._run_model(use_lags=self.model_lags)


class Regularised_Regression_Model(FredRegression):
    def __init__(
        self,
        data: pd.DataFrame,
        start_date: Union[datetime.datetime, None],
        end_date: Union[datetime.datetime, None],
        dependent_variable_name: str,
        window_size: int,
        model_lags: List[int],
        handle_missing: int = 0,
        frequency: str = "monthly",
        use_pca_features: bool = False,
        fred_factors_kwargs: dict = None,
        regularisation_type: str = "Ridge",
        lambdas: List[int] = np.logspace(-2, 1, 4),
    ):
        """
        Regularised Linear Regression model, currently fits Lasso and Ridge.

        Class specific parameters:

        :param regularisation_type : please specify Lasso or Ridge.
        :param lambdas : lamba values to check for the model.
        we check 4 values as default : [ 0.01,  0.1 ,  1.  , 10.  ]

        Please refer to base class documentation for details of other params.
        """

        # Check that parameters are set correctly
        # -------------------------------------------------------------------------------------------------------
        assert regularisation_type in [
            "Ridge",
            "Lasso",
        ], "Regularisation type should be Ridge or Lasso"
        assert model_lags != None, "please provide lags to use for each window"
        # -------------------------------------------------------------------------------------------------------

        super().__init__(
            model_name=regularisation_type,
            data=data,
            start_date=start_date,
            end_date=end_date,
            window_size=window_size,
            dependent_variable_name=dependent_variable_name,
            handle_missing=handle_missing,
            frequency=frequency,
            model_lags=model_lags,
            use_pca_features=use_pca_features,
            fred_factors_kwargs=fred_factors_kwargs,
        )

        self.regularisation_type = regularisation_type
        self.lambdas = lambdas
        self.model_coef = []

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

    def _model(self, series_data, features, lag):

        best_bic = np.inf
        best_alpha = 0

        self._lagged_features_and_target(series_data, features, lag)

        # pick the last one as out-of-sample
        in_sample_y = self.curr_target[:-1].values.reshape(-1)
        in_sample_x = self.curr_features[:-1, :]
        out_sample_y = self.curr_target[-1]
        out_sample_x = self.curr_features[-1, :].reshape(1, -1)

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

        self.out_of_sample_error.append(out_sample_error)
        self.in_sample_error.append(in_sample_error)
        self.true.append(out_sample_y)
        self.predicted.append(out_sample_y_pred)
        self.model_coef.append(best_model_coeffs)

    def run_model(self):
        self._run_model(use_lags=self.model_lags)


class Neural_Network(FredRegression):
    def __init__(
        self,
        data: pd.DataFrame,
        start_date: Union[datetime.datetime, None],
        end_date: Union[datetime.datetime, None],
        dependent_variable_name: str,
        window_size: int,
        model_lags: List[int],
        handle_missing: int = 0,
        frequency: str = "monthly",
        model_name: str = "neural_network",
        use_pca_features: bool = False,
        fred_factors_kwargs: dict = None,
        hidden_layer_sizes: Tuple[int] = (25, 20),
        max_iter: int = 1000,
        activation: str = "relu",
    ):
        """
        Fits neural network model:

        Class specific parameters:

        :param hidden_layer_sizes : tuple of hidden layer sizes. default (25,20) will be two hidden layers with 25 and 20 neurons respectively.
        :param max_iter : maximum number of iterations for the model, default 1000
        :param activation : activation to use for hidden layers, default relu, Supported activations are ['identity', 'logistic', 'relu', 'softmax', 'tanh']

        Please refer to base class documentation for details of other params.
        """
        super().__init__(
            model_name=model_name,
            data=data,
            start_date=start_date,
            end_date=end_date,
            window_size=window_size,
            dependent_variable_name=dependent_variable_name,
            handle_missing=handle_missing,
            model_lags=model_lags,
            frequency=frequency,
            use_pca_features=use_pca_features,
            fred_factors_kwargs=fred_factors_kwargs,
        )

        # Check that parameters are set correctly
        # -------------------------------------------------------------------------------------------------------
        assert activation in [
            "identity",
            "logistic",
            "relu",
            "softmax",
            "tanh",
        ], " Supported activations are ['identity', 'logistic', 'relu', 'softmax', 'tanh']"
        assert model_lags != None, "please provide lags to use for each window"
        # -------------------------------------------------------------------------------------------------------

        self.model_lags = model_lags
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.activation = activation

    def _model(self, series_data, features, lag):

        self._lagged_features_and_target(series_data, features, lag)

        scaler_X = StandardScaler()
        X = scaler_X.fit_transform(self.curr_features)

        scaler_y = StandardScaler()
        y = scaler_y.fit_transform(self.curr_target.values.reshape(-1, 1))

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

        model = model.fit(in_sample_x, in_sample_y.ravel())

        test_pred_y = model.predict([out_sample_x])
        out_sample_y_pred = scaler_y.inverse_transform(test_pred_y.reshape(-1, 1))[0][0]

        train_pred_y = model.predict(in_sample_x)
        in_sample_y_pred = scaler_y.inverse_transform(train_pred_y.reshape(-1, 1))[0][0]
        in_sample_error = self.get_error(in_sample_y_pred, self.curr_target[:-1])
        out_sample_error = self.get_error(out_sample_y_pred, self.curr_target[-1])

        self.out_of_sample_error.append(out_sample_error)
        self.in_sample_error.append(in_sample_error)
        self.true.append(out_sample_y)
        self.predicted.append(out_sample_y_pred)

    def run_model(self):
        self._run_model(use_lags=self.model_lags)
