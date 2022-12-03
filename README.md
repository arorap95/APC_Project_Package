# APC 524 Project: golden_fred
Members: Alex A., Poorva A., Sneha M.

## Project Description:

This package seeks to provide a golden copy of FRED-MD and FRED-QD data for use by economic and financial researchers, as well as provide a simple framework to perform common statistical tasks that researchers perform with this data, such as dimension reduction (by means of PCA), statistical modeling, and covariance matrix estimation.

In addition to providing a golden copy of the raw data, it also provides user-friendly methods of returning stationarized data, combinations of FRED-MD and FRED-QD data into a standard monthly panel using a fuzzy-match technique, and picking and choosing of which groups of variables to include from FRED-MD and FRED-QD. It thus seeks to prevent manual downloading of files from FRED-MD and FRED-QD, as well as the potential bugs that could arise from stationarizing the data, picking and choosing among variable groups, etc.

## How to Use the Package:
----------------------------------------------------------------------------------------------------------------------------------------------------------
### Import all modules

```python
from get_fred import *
from fred_factors import *
from fred_covariance import *
from fred_backtest import *
from fred_regression import *
```

----------------------------------------------------------------------------------------------------------------------------------------------------------
### Obtain Clean Fred Data

#### Get raw FRED data:
Get raw FRED-MD data
```python
y = GetFred()
data = y.get_fred_md()
data.tail()
```

Get raw FRED-QD data
```python
y = GetFred()
data = y.get_fred_qd()
data.tail()
```

#### Get stationarized FRED data
Get FRED data with FRED-recommended stationarity transformations
```python
y = GetFred(transform=True)
data_md = y.get_fred_md()
data_qd = y.get_fred_qd()
data_md.tail()
```

#### Filter for specific groups of variables / combine FRED-QD and FRED-MD variables into one panel
Filter for a specific group of variables (in this case, "Prices") in FRED-MD and replace variable names with descriptions.
```python
y = GetFred()
y.group_lookup['FRED-MD'] # to see which group number correponds to the price group
data_prices = y.get_fred_md(group_no=6,use_descriptions=True)
data_prices.tail()
```

Combine FRED-QD data with FRED-MD data using ``golden_fred`` fuzzy match methodology. Note that this will interpolate FRED-QD to monthly by default.

```python
y = GetFred()
data_combined = y.combine_fred()
data_combined.tail()
```

Combine a specific FRED-MD and FRED-QD collection of groups specified by the user.
```python
y = GetFred()
y.group_lookup['FRED-QD']
data_combined = y.combine_fred(fred_md_group = [6,3], fred_qd_group = [2])
data_combined.tail()
```

#### See appendix (e.g., background information provided by FRED-MD and FRED-QD authors)
```python
y = GetFred()
y.get_appendix(frequency='monthly')
```

----------------------------------------------------------------------------------------------------------------------------------------------------------
### Run Principal Component Analysis (PCA) to Obtain Factors

Step 1: Obtain Standardized Data
```python
y = GetFred(transform=True)
data = y.get_fred_md()
```

Step 2: Run Factor Analysis with custom inputs (or can use default settings)
```python
factors = FredFactors(data, standardization=2, factorselection={1:90}, removeoutliers=True, handle_missing=1)

#return PCA transformed data
factors.get_fred_factors()
```

Step 3: Extract other relevant outputs
```python
#PCA components
components = factors.components

#Original data with missing values replaced by factor projections
filleddata = factors.filleddata
```
----------------------------------------------------------------------------------------------------------------------------------------------------------
### Estimate Covariance Matrix
Step 1: Obtain Standardized Data
```python
y = GetFred(transform=True)
data = y.get_fred_md()
```

Step 2: Estimate Covariance Matrix with a range of available techniques
```python
covariance = CovarianceFred(data)

#Empirical Covariance
covariance.sample_covariance()

#Empirical Covariance with a minimum correlation threshold (Sparse technique)
covariance.threshold_covariance(correlationthreshold = 0.01)

#Covariance Matrix after transforming to semi-positive definite (Method 1)
covariance.positive_semidefinite_method1()

#Covariance Matrix after transforming to semi-positive definite (Method 2)
covariance.positive_semidefinite_method2()
```
----------------------------------------------------------------------------------------------------------------------------------------------------------
### Forecast Data Using Various Regression Techniques
We provide 4 prediction techniques to the user.
The input to all models is the data from the GetFred() module, along with other model specific parameters.
All models have the following methods:
model.dates_tested : dates for which we tested our regression model
model.in_sample_error : in sample or train error of the model
model.out_of_sample_error: out of sample or test error of the model
model.targets : true value of the dependent variable
model.predicted : predicted value by the model.
1. Auto Regressive Model
2. Regularised Regression Models:
a. Lasso
b. Ridge
3. Neural Network Model

Please refer to the docstrings for detailed explanation on each input parameter to the models.

Step 1: Obtain Standardized Data
```python

y = GetFred(transform=True)
data = y.get_fred_md()
```

Step 2: Instantiate an object of the required class :
AR model : class AR_Model
Lasso/Ridge : class Regularised_Regression_Model
Neural Network : class Neural_Network

Below I provide examples to instantiate each of these classes with some dummy parameter values
```python
model = AR_Model( data = data,
                  max_lag = 100,
                  start_date = pd.to_datetime('2010-01'),
                  end_date = pd.to_datetime('2020-01'),
                  dependent_variable_name = 'CPIAUCSL',
                  window_size = 100,
                  lag_patience=5,
                  model_name="AR",
                  handle_missing=0 )


 model = Regularised_Regression_Model( data = data,
                                      regularisation_type = 'Lasso',
                                      model_lags = [2]*120,
                                      start_date = pd.to_datetime('2010-01'),
                                      end_date = pd.to_datetime('2020-01'),
                                      dependent_variable_name = 'CPIAUCSL',
                                      window_size = 100,
                                      handle_missing=0 )

 model = Neural_Network( data = data,
                          max_iter = 100,
                          start_date = pd.to_datetime('2010-01'),
                          end_date = pd.to_datetime('2020-01'),
                          dependent_variable_name = 'CPIAUCSL',
                          hidden_layer_sizes = (50,20,30),
                          activation = "relu",
                          model_lags = [2]*120,
                          window_size = 100,
                          handle_missing=0 )

```

Step 3: Fit the model and obtain outputs
```python
model.fit()

model.in_sample_error #gives the in sample error
model.out_of_sample_error #gives the in out of sample error
model.true #gives the true values used in regression models
model.predicted #gives the predicted values
```

Step 4: Visualising results
We implement a function to visualise the in sample and out of sample error of the model
```python
model.plot_insample_and_outofsample_error()
```
Please refer to test_fit() in tests/test_fred_regression.py for working examples.

----------------------------------------------------------------------------------------------------------------------------------------------------------
### Run Backtest Functionality on Data or Portfolio of Data

#### Get Backtest Statistics on Custom Portfolio of Data
Step 1: Obtain Raw Data
```python
y = GetFred()
data = y.get_fred_md()
```

Step 2: Run Backtest with Custom Portfolio of Data Columns and Specified Turnover

```python
backtest = FredBacktest(data)
backtest.fred_compute_backtest(["RPI", "W875RX1", "IPDCONGD"], initialweights = [0.2, 0.5, 0.3], Tcosts = [0 ,0 ,0])
```

#### Run Regime Filtering on Historical Data to Identify Expansion, Transition and Contraction Regimes

Step 1: Obtain Raw Data
```python
y = GetFred()
data = y.get_fred_md()
```
Step 2: Run L1 Trend Filtering Algorithm with Specified Lambda Parameter

```python
backtest = FredBacktest(data)
backtest.regime_filtering(["RPI", "IPDCONGD"], lambda_param=[10000,10000])

```
----------------------------------------------------------------------------------------------------------------------------------------------------------
