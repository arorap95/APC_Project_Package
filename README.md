# APC 524 Project: golden_fred
Members: Alex, Poorva, Sneha

## Project Description:


## How to Use the Package:

----------------------------------------------------------------------------------------------------------------------------------------------------------
### Obtain Clean Fred Data

#### Get raw FRED data:
Get raw FRED-MD data
```python
y = GetFred()
data = y.get_fred_md()
data.head()
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
factors.getFredFactors()
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

----------------------------------------------------------------------------------------------------------------------------------------------------------
### Obtain Backtest Statistics on Data or Portfolio of Data
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
----------------------------------------------------------------------------------------------------------------------------------------------------------
