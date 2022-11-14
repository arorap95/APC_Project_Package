import abc
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

@dataclasses.dataclass(frozen=True, eq=False)
class FredRegression(abc.ABC):        
     
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
        
    def find_feature_importance(self):
        pass
    
    def handle_missing(self):
        return 1
    
    def _fill_missing_data(self):
        '''
        NOTE : the operations are done IN-PLACE
        Fill missing values
        :param: handle_missing
        0: Forward Fill followed by Backward Fill missing values
        1: Fill missing values with mean of respective series
        
        '''
 
        self.data.drop(index=self.data.index[[0,1]], axis=0, inplace=True)
        self.original_input = self.data.copy()

        if self.handle_missing() == 0:
            self.data = self.data.ffill().bfill()
            
        elif self.handle_missing() == 1:
            self.data = self.data.fillna(self.data.mean())
            
    def plot_insample_and_outofsample_error(self):
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize = (10, 5))

        ax[0].plot(self.dates_tested, self.in_sample_error, '--o')
        ax[0].set_title('In sample error of {} Model'.format(self.model_name))

        ax[1].plot(self.dates_tested, self.out_of_sample_error, '--o')
        ax[1].set_title('Out sample error of {} Model'.format(self.model_name))

        plt.gcf().autofmt_xdate()
        
class AR_Model(FredRegression):
    
    def __init__(self, data, max_lag, start_date, end_date, dependent_variable_name, window_size, lag_patience = 5, model_name=None):
        '''
        Fits the Auto-Regressive model on any time series data.
        
        :param data: input time series data, must contain a column with the user input value of dependent_variable_name.
        data should have minimum [start_date - window_size to end_date] rows
        :param max_lag: maximum number of lags for the AR model to be tested. 
        :param start_date: first date model tests
        :param end_date : last date model tests
        
             
        '''
        self.model_name = model_name or 'AR'
        
        self.data                    = data
        self.max_lag                 = max_lag
        self.start_date              = start_date
        self.end_date                = end_date
        self.window_size             = window_size
        self.dependent_variable_name = dependent_variable_name
        self.lag_patience            = lag_patience
        
        self.out_of_sample_error     = None
        self.in_sample_error         = None
        self.dates_tested            = None  
        self.lag_from_ar_model       = None
        
  
    def features_and_target(self):
        '''
        splits the data into features and target based on 
        the dependent_variable_name that user inputs 
        '''      
        self.features = self.data.drop(self.dependent_variable_name, axis=1)
        self.target = self.data[self.dependent_variable_name]     
        
    def create_lagged_data(self, data, max_lag, is_series=False):

        lagged = []
        for lag in range(1, max_lag+1):
            lagged.append(data.shift(lag))
        if is_series:
            return np.stack(lagged).T
        else:
            return np.concatenate(lagged, axis=1)
        
    def get_error( self, y_pred, y_true):
        ''' returns the mean_squared_error'''
        return np.mean((y_pred-y_true)**2)
    
    def compute_bic( self, model):
        return model.bic
    
    
    def find_best_AR_model(self, series_data ):
        """
        lag_patience: Fitting AR models with higher orders take much longer and have
          convergence issues. We therefore use a heuristic to reduce the computatoin
          time. We keep monitoring the BIC value, and if the BIC value hasn't 
          reduced for `lag_patience` number of orders, we stop the computation 
          then and return the current lowest. We experimented with different 
          `patience_thres` values and the results look qualitatively similar. 
        """

        best_bic = np.inf
        series_data = series_data.dropna() # drop any na it has

        for lag in range(1, self.max_lag + 1):

            curr_features = self.create_lagged_data(series_data, max_lag = lag, is_series=True)[lag:, ]
            curr_target = series_data[lag:]      

            #pick the last one as out-of-sample
            in_sample_y = curr_target[:-1].values.reshape(-1)
            in_sample_x = curr_features[:-1, :]
            out_sample_y = curr_target[-1]
            out_sample_x = curr_features[-1, :].reshape(1, -1)

            # add bias terms to features
            in_sample_x = sm.add_constant(in_sample_x)        
            out_sample_x = sm.add_constant(out_sample_x, has_constant='add')


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
            curr_window_target = self.target[self.target.index <= date][-self.window_size:]
            best_lag, in_sample_error, out_sample_error = self.find_best_AR_model(series_data = curr_window_target) 
                                                                             
            out_err_AR.append(out_sample_error)
            in_err_AR.append(in_sample_error)
            order_array_AR.append(best_lag)

            date = date + relativedelta(months=1)
            
        return out_err_AR,in_err_AR,order_array_AR,dates_tested 
    
    def fit( self ):
        out_err, in_err, order_array,dates_tested = self.run_model()      
        self.out_of_sample_error = out_err
        self.in_sample_error     = in_err
        self.lag_from_ar_model   = order_array
        self.dates_tested        = dates_tested     
