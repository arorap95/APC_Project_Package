import pytest
import unittest
import numpy as np
import pandas as pd
from golden_fred import fred_regression
from golden_fred import get_fred
import datetime
from parameterized import parameterized

def create_input( use_Fred_data = False, test_missing = False ):

    if test_missing:
        x =[ pd.to_datetime('2010-01'),
             pd.to_datetime('2010-02'),
             pd.to_datetime('2010-03'),
             pd.to_datetime('2010-04'),
             pd.to_datetime('2010-05'),
             pd.to_datetime('2010-06')]

        y = [np.nan, np.nan, 12, 100, 8, 1]
        z = [np.nan, np.nan, 200, 2, 11, 2]

        df = pd.DataFrame({"Date": x, "col1": y, "result": z})
        df.set_index("Date", inplace=True)

    elif use_Fred_data:
        data = get_fred.GetFred()
        df   = data.get_fred_md()

    else:

        df = pd.DataFrame( columns = ['Date','col1', 'col2', 'result'],
                           data = [ [ pd.to_datetime('2010-01'), 1, 2, 4  ],
                                    [ pd.to_datetime('2010-02'), 2, 2, 6  ],
                                    [ pd.to_datetime('2010-03'), 3, 1, 7 ],
                                    [ pd.to_datetime('2010-04'), 1, 0, 2  ],
                                    [ pd.to_datetime('2010-05'), 5, 2, 10  ],
                                    [ pd.to_datetime('2010-06'), 4, 2, 10  ]
                                    ] )
        df.set_index("Date", inplace=True)
    return df


def get_params_fit( model_name):

    df = create_input(use_Fred_data=True)
    start = pd.to_datetime('2010-03')
    end   = pd.to_datetime('2011-03')
    nMonths = round((end-start)/np.timedelta64(1,'M'))
    lag = 4
    ncols = len(df.axes[1])
    if model_name == 'AR':
         param1 = {'dependent_variable_name' : 'CPIAUCSL', 
                    'data' : df,
                    'start_date' : start,
                    'end_date' : end,
                    'window_size' : 100,
                    'max_lag' : 5,
                   'model_lags':None,
                     }
    elif model_name == 'reg_regression' or model_name == 'nnet':
        
        param1 = {'dependent_variable_name' : 'CPIAUCSL', 
                        'data' : df,
                        'start_date' : start,
                        'end_date' : end,
                        'window_size' : 100,
                        'model_lags' : [lag]*nMonths }
        
    param2 = param1
    param2.update({'use_pca_features': True,
                   'fred_factors_kwargs' : { 'standardization':2, 
                                            'factorselection':{1:90}, 
                                            'removeoutliers':True, 
                                            'maxfactors':None},
                   'model_lags' : [lag]*nMonths 
                  })
    return param1, param2, ncols, lag, nMonths

class test_Fred_Regression(unittest.TestCase):      
    def c( self,
           data = create_input(), 
           start_date = pd.to_datetime('2010-03'),
           end_date = pd.to_datetime('2010-05'),
           dependent_variable_name = 'result',
           window_size = 2,
           handle_missing = 0,
           model_name = 'please_specify',
           frequency = 'monthly'):
        
        fred_reg = fred_regression.FredRegression
        return fred_reg(data = data,
                        start_date  = start_date,
                        end_date = end_date,
                        dependent_variable_name = dependent_variable_name,
                        window_size = window_size,
                        handle_missing = handle_missing, 
                        frequency = frequency, 
                       model_name = model_name)
    
    
    def test__features_and_target(self):
        model = self.c()
        model._features_and_target()

        self.assertEqual(len(model.target),6)
        self.assertEqual(len(model.features.axes[1]),2)
        
    
    def test_get_error(self):
        model = self.c()
        err = model.get_error(np.array([1,2]),np.array([0,1]))
        self.assertEqual(err,1) 
    
    @parameterized.expand([['0',True],['1',False]])    
    def test__create_lagged_data(self, name, is_series):
        model = self.c()
        lag = 4
        out = model._create_lagged_data(model.data, lag, is_series)
        if not is_series:
            self.assertEqual(out.shape, (len(model.data),len(model.data.columns)*lag))
        else:
            self.assertEqual(out.shape, (len(model.data.columns), len(model.data),lag) )          
        
          
    def test__lagged_features_and_target(self):
        model = self.c()
        model._features_and_target()
        lag = 4
        feature_cols = len(model.data.columns)-1
        rows = len(model.data[lag:])
        model._lagged_features_and_target(model.target,model.features,lag )
        self.assertEqual(model.lag_features.shape,(rows,lag))
        self.assertEqual(model.data_features.shape,(rows,lag*feature_cols))
        self.assertEqual(model.curr_features.shape,(rows,lag+lag*feature_cols))
                             
        
    @parameterized.expand([['0',0],['1',1]])
    def test_handle_missing(self,name,val):
        df = create_input(test_missing = True)
        model = self.c(data = df,
                        handle_missing=val, 
                        start_date = pd.to_datetime('2010-01'),
                       end_date = pd.to_datetime('2010-04'),)
                       
        model._fill_missing_data()
        assert df.notnull().values.any()

class test_AR_Model(unittest.TestCase):
    
    def c( self,
           data = pd.DataFrame(), 
           max_lag = 1,
           start_date = pd.to_datetime('2020-01'),
           end_date = pd.to_datetime('2020-05'),
           dependent_variable_name = 'please_define',
           window_size = 2,
           handle_missing = 0,
          model_lags = None,
           use_pca_features : bool = False,
          fred_factors_kwargs : dict = None,
          ):
        
        AR_model = fred_regression.AR_Model
    
        return AR_model(data = data,
                        max_lag = max_lag,
                        start_date  = start_date,
                        end_date = end_date,
                        dependent_variable_name = dependent_variable_name,
                        window_size = window_size,
                        handle_missing = handle_missing,
                        use_pca_features = use_pca_features,
                        fred_factors_kwargs = fred_factors_kwargs,
                        model_lags = model_lags)
            
   
    def test_class_initialisation(self):
        df = create_input()
        model = self.c(data = df, 
                       max_lag = 3,
                       start_date = pd.to_datetime('2010-01'),
                       end_date = pd.to_datetime('2010-06'),
                       dependent_variable_name = 'result')
                       
        self.assertEqual( model.start_date,pd.to_datetime('2010-01') )
        self.assertEqual( model.end_date,pd.to_datetime('2010-06') )
        self.assertEqual( model.dependent_variable_name,'result' )
        self.assertEqual( model.lag_patience, 5 )  
        self.assertEqual( model.window_size, 2 )
        self.assertEqual( model.handle_missing, 0 )
        self.assertEqual( model.max_lag, 3 )       
 
    param1, param2, _,_, _ = get_params_fit(model_name = 'AR')
    @parameterized.expand([['not_pca0', param1]])
    def test_fit(self, name, params ):
        _, _, ncols,lag, nMonths = get_params_fit(model_name = 'AR')
        model = self.c(**params)
        model.fit()
       
        self.assertEqual( len(model.in_sample_error), nMonths )
        self.assertEqual( len(model.out_of_sample_error), nMonths )
        self.assertEqual( len(model.lag_from_ar_model), nMonths )
        self.assertEqual( len(model.dates_tested), nMonths )
        self.assertEqual( len(model.predicted), nMonths )
        self.assertEqual( len(model.true), nMonths )

     
class test_Regularised_Regression_Model(unittest.TestCase): 
        
    def c( self,
           data = None, 
           regularisation_type = 'please_specify',
           start_date = pd.to_datetime('2020-01'),
           end_date = pd.to_datetime('2020-05'),
           dependent_variable_name = 'please_define',
           window_size = 2,
           handle_missing = 0,
           lambdas = [0.01],
           model_lags = [2]*4,
           use_pca_features : bool = False,
          fred_factors_kwargs : dict = None,):
        
        Regularised_Regression_Model = fred_regression.Regularised_Regression_Model
    
        return Regularised_Regression_Model(data = data,
                                            start_date  = start_date,
                                            end_date = end_date,
                                            dependent_variable_name = dependent_variable_name,
                                            window_size = window_size,
                                            handle_missing = handle_missing,
                                            regularisation_type = regularisation_type,
                                            model_lags = model_lags,
                                            lambdas = lambdas)
    
    @parameterized.expand([['ridge','Ridge'],['lasso','Lasso']])
    def test_class_initialisation(self, name,reg_type):
        df = create_input()
        model = self.c(data = df, 
                       start_date = pd.to_datetime('2010-01'),
                       end_date = pd.to_datetime('2010-06'),
                       dependent_variable_name = 'result',
                       model_lags = [2]*5,
                       regularisation_type = reg_type)
                       
        self.assertEqual( model.start_date,pd.to_datetime('2010-01') )
        self.assertEqual( model.end_date,pd.to_datetime('2010-06') )
        self.assertEqual( model.dependent_variable_name,'result' )
        self.assertEqual( model.window_size, 2 )
        self.assertEqual( model.handle_missing, 0 )
        self.assertEqual( model.regularisation_type, reg_type )
        self.assertEqual( len(model.lambdas), 1 )
    
    param1, param2, _,_, _ = get_params_fit(model_name = 'reg_regression')
    @parameterized.expand([['not_pca0','Ridge', param1],['not_pca1','Lasso',param1],
                          ['pca0','Ridge', param2],['pca1','Lasso',param2]])
    def test_fit(self, name, reg_type, params):
        _, _, ncols,lag, nMonths = get_params_fit(model_name = 'reg_regression')
        params.update({'regularisation_type':reg_type})
        model = self.c( **params)
        model.fit()
        
        self.assertEqual( len(model.in_sample_error), nMonths )
        self.assertEqual( len(model.out_of_sample_error), nMonths )
        self.assertEqual( len(model.model_coef), nMonths )
        self.assertEqual( len(model.dates_tested), nMonths )
        self.assertEqual( len(model.model_coef[0]), ncols*lag )
        self.assertEqual( len(model.predicted), nMonths )
        self.assertEqual( len(model.true), nMonths )

        
class test_Neural_Network(unittest.TestCase): 
    def c( self,
           data = None, 
           start_date = pd.to_datetime('2020-01'),
           end_date = pd.to_datetime('2020-05'),
           dependent_variable_name = 'please_define',
           hidden_layer_sizes = 'please_specify',
           activation = 'please_specify',
           max_iter = 'please_specify',
           window_size = 2,         
           handle_missing = 0,
           model_lags = [2]*4,
           use_pca_features : bool = False,
           fred_factors_kwargs : dict = None,):
        
        nnet = fred_regression.Neural_Network
    
        return nnet(data = data,
                    start_date  = start_date,
                    end_date = end_date,
                    dependent_variable_name = dependent_variable_name,
                    window_size = window_size,
                    activation = activation,
                    hidden_layer_sizes =hidden_layer_sizes,
                    max_iter = max_iter,
                    handle_missing = handle_missing,
                    model_lags = model_lags)

    
    @parameterized.expand([['0','relu',100],['1','tanh',10]])
    def test_class_initialisation(self, name, activation, max_iter):
        df = create_input()
        model = self.c(data = df, 
                       start_date = pd.to_datetime('2010-01'),
                       end_date = pd.to_datetime('2010-06'),
                       dependent_variable_name = 'result',
                       activation = activation,
                       hidden_layer_sizes = (1,2),
                       model_lags = [2]*5,
                       max_iter = max_iter )
                       
        self.assertEqual( model.start_date,pd.to_datetime('2010-01') )
        self.assertEqual( model.end_date,pd.to_datetime('2010-06') )
        self.assertEqual( model.dependent_variable_name,'result' )
        self.assertEqual( model.window_size, 2 )
        self.assertEqual( model.handle_missing, 0 )
        self.assertEqual( model.activation, activation )
        self.assertEqual( model.max_iter, max_iter )
        self.assertEqual( len(model.hidden_layer_sizes), 2 )
     
    param1, param2, _,_, _ = get_params_fit(model_name = 'nnet')
    @parameterized.expand([['not_pca0','relu',100, (10,4), param1],['not_pca1','logistic',150, (4,3), param1],
                          ['pca0','relu',100, (10,4), param2],['pca1','logistic',150, (4,3), param2]])
    def test_fit(self, name, activation, max_iter, hidden, params ):
        _, _, ncols,lag, nMonths = get_params_fit(model_name = 'nnet')
        params.update({'hidden_layer_sizes' :hidden,
                        'max_iter':max_iter,
                      'activation':activation})
        model = self.c( **params)
        model.fit()
        
        self.assertEqual( len(model.in_sample_error), nMonths )
        self.assertEqual( len(model.out_of_sample_error), nMonths )
        self.assertEqual( len(model.dates_tested), nMonths )
        self.assertEqual( len(model.predicted), nMonths )
        self.assertEqual( len(model.true), nMonths )
    
if __name__ == '__main__':
    unittest.main()
