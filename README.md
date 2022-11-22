# APC 524 Project: golden_fred
Members: Alex, Poorva, Sneha

## Project Description:


## How to Use the Package:

----------------------------------------------------------------------------------------------------------------------------------------------------------
### Obtain Clean Fred Data


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
----------------------------------------------------------------------------------------------------------------------------------------------------------

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

----------------------------------------------------------------------------------------------------------------------------------------------------------
### Obtain Backtest Statistics on Data or Portfolio of Data
----------------------------------------------------------------------------------------------------------------------------------------------------------
Step 1: Obtain Raw Data
```python
y = GetFred()
data = y.get_fred_md()
```

Step 2: Run Backtest with Custom Portfolio of Data Columns and Specified Turnover


