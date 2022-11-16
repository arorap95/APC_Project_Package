import pytest
import unittest
import numpy as np
import pandas as pd
from golden_fred import fred_regression
from golden_fred import get_fred
import datetime
from parameterized import parameterized

class test_AR_Model(unittest.TestCase):
    
    def c( self,
           data = None, 
           max_lag = 1,
           start_date = pd.to_datetime('2020-01'),
           end_date = pd.to_datetime('2020-05'),
           dependent_variable_name = 'please_define',
           window_size = 2,
           handle_missing = 0):
        
        AR_model = fred_regression.AR_Model
    
        return AR_model(data = data,
                        max_lag = max_lag,
                        start_date  = start_date,
                        end_date = end_date,
                        dependent_variable_name = dependent_variable_name,
                        window_size = window_size,
                        handle_missing = handle_missing)
            
    
    def create_input(self, use_Fred_data = False, test_missing = False ):
        
        if test_missing:
            x = [
                datetime.datetime(2022, 11, 6),
                datetime.datetime(2022, 11, 7),
                datetime.datetime(2022, 11, 8),
                datetime.datetime(2022, 11, 9),
                datetime.datetime(2022, 11, 10),
                datetime.datetime(2022, 11, 11) ]

            y = [np.nan, np.nan, 12, 100, 8, 1]
            z = [np.nan, np.nan, 200, 2, 11, 2]

            df = pd.DataFrame({"Date": x, "column1": y, "column2": z})
            df.set_index("Date", inplace=True)

            return df
        
        elif use_Fred_data:
            data = get_fred.GetFred()
            df = data.get_fred_md()
            
        else:
        
            df = pd.DataFrame( columns = ['Date','col1', 'col2', 'result'],
                               data = [ [ pd.to_datetime('2010-01'), 1, 2, 4  ],
                                        [ pd.to_datetime('2010-02'), 2, 2, 6  ],
                                        [ pd.to_datetime('2010-03'), 3, 1, 7 ],
                                        [ pd.to_datetime('2010-04'), 1, 0, 2  ],
                                        [ pd.to_datetime('2010-05'), 5, 2, 10  ],
                                        [ pd.to_datetime('2010-06'), 4, 2, 10  ]
                                        ] )
        return df
    
   
    def test_class_initialisation(self):
        df = self.create_input()
        model = self.c(data = df, 
                       max_lag = 3,
                       start_date = min(df.Date),
                       end_date = max(df.Date),
                       dependent_variable_name = 'result')
                       
        self.assertEqual( model.start_date,pd.to_datetime('2010-01') )
        self.assertEqual( model.end_date,pd.to_datetime('2010-06') )
        self.assertEqual( model.dependent_variable_name,'result' )
        self.assertEqual( model.lag_patience, 5 )  
        self.assertEqual( model.window_size, 2 )
        self.assertEqual( model.handle_missing, 0 )
        self.assertEqual( model.max_lag, 3 )       
    
    def test_features_and_target(self):
        
        df = self.create_input()
        
        model = self.c( dependent_variable_name = 'result', data = df)
        
        model.features_and_target()
        
        self.assertEqual(len(model.target),6)
        self.assertEqual(len(model.features.axes[1]),3)
        
    
    def test_get_error(self):
        model = self.c()
        err = model.get_error(np.array([1,2]),np.array([0,1]))
        self.assertEqual(err,1) 
        
    @parameterized.expand([['0',0],['1',1]])
    def test_handle_missing(self,name,val):
        df = self.create_input(test_missing = True)
        model = self.c(data = df,handle_missing=val)
                       
        model._fill_missing_data()
        assert df.notnull().values.any()
                      
if __name__ == '__main__':
    unittest.main()
