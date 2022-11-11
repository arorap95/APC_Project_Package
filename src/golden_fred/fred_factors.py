import pandas as pd
import numpy as np
from urllib import request
import datetime
import warnings
from typing import Union, Tuple
import sklearn as sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class FredFactors:
    def __init__(
        self,
        inputdata: pd.DataFrame,
        standardization: int = 2,
        maxfactors: Union[int, None] = None,
        factorselection: Union[dict, None] = None,
        removeoutliers: bool = True,
        handle_missing: int = 1,
    ) -> None:

        """
        Run PCA on input data set and conduct factor analysis

        INPUTS:

        Standardization is an integer in [0,1,2] representing:
        0: No standardization
        1: Only demean but not scale
        2: Demean and scale

        Maximum factors is an integer representing the maximum factors

        Factor Selection, if not None, is a dictionary representing the method to select optimal number of factors
        With Keys in [0,1,2]:
        0: Stop at Kth PC such that K+1th PC does not add more than a specified value of to the already explained variance
        Where the Value is the specified additional variance explained as specified by the user
        1: Stop at Kth PC such that the total variance explained  by the components is a specified value
        Where the Value is the specified total variance explained as specified by the user
        2: Biggest Drop Method - identify Kth PC such that r := arg max lambda(j) / lambda(j+1)
        Where the value is the specified target rank of the ratio (0= arg max)

        Handle Missing is an integer in [0,1] representing:
        0: Forward Fill followed by Backward Fill missing values
        1: Fill missing values with mean of respective series
        """

        # Check that parameters are set correctly
        # -------------------------------------------------------------------------------------------------------
        assert standardization in [
            0,
            1,
            2,
        ], "Standardization parameter must be an integer in [0,1,2]"
        if maxfactors is None:
            maxfactors = len(inputdata.columns)
        assert maxfactors <= len(
            inputdata.columns
        ), "Maximum factors cannot exceed number of columns"
        assert handle_missing in [
            0,
            1,
        ], "Handle Missing parameter must be an integer in [0,1]"

        if factorselection is not None:
            assert (
                len(factorselection) == 1
            ), "Can only specify one factor selection method"
            assert list(factorselection.keys())[0] in [
                0,
                1,
                2,
            ], "Factor Selection Method must be in [0,1,2]"

            if list(factorselection.keys())[0] in [0, 1]:
                assert (
                    list(factorselection.values())[0] >= 0
                    and list(factorselection.values())[0] <= 100
                ), "Target variance must be between 0-100 percent"

            elif list(factorselection.keys()[0]) in [2]:
                assert (
                    list(factorselection.values())[0] < len(inputdata.columns) - 2
                ), "Maximum rank is number of columns - 2"
        # -------------------------------------------------------------------------------------------------------

        self.inputdata = inputdata
        self.standardization = standardization
        self.maxfactors = maxfactors
        self.factorselection = factorselection
        self.removeoutliers = removeoutliers
        self.outliers = None
        self.handle_missing = handle_missing
        self.currentdata = inputdata.copy()

    def getFredFactors(self):
        """
        Wrapper Function called by the user. Processes input data by removing outliers, fills missing values,
        runs PCA, and selects optimal number of factors.
        :return: self.factors
        """
        self._removeoutliers()
        self._fillmissing()
        self._runPCAalgorithm()
        self._selectoptimalfactors()

        return self.factors

    def _standardize(self):
        """
        Standardizes data per user specification.
        :param: standardization
        0: No standardization
        1: Only demean but not scale
        2: Demean and scale
        :return: Sklearn Scalar object
        """

        if self.standardization == 0:
            scaler = StandardScaler(with_mean=False, with_std=False)

        elif self.standardization == 1:
            scaler = StandardScaler(with_mean=True, with_std=False)

        elif self.standardization == 2:
            scaler = StandardScaler(with_mean=True, with_std=True)

        return scaler

    def _removeoutliers(self):
        """
        Remove outliers.
        :param: removeoutliers
        x is considered an outlier if: abs(x-median)>10*interquartile range
        :return: self.currentdata without outliers
        """
        if self.removeoutliers:
            median = self.currentdata.median(axis=0)
            iqr = self.currentdata.quantile(0.75, axis=0) - self.currentdata.quantile(
                0.25, axis=0
            )
            iqr_limit = iqr * 10
            outliers = abs(self.currentdata - median) > iqr_limit

            # replace outliers in data with NaNs
            self.currentdata = self.currentdata[~outliers]
            self.outliers = outliers.sum()

    def _selectoptimalfactors(self):
        """
        Select optimal number of factors after PCA
        :param: factorselection
        None: Default to max factors
        {0: x): Stop at Kth PC such that K+1th PC does not add more than x% to already explained variance
        {1:y}: Stop at Kth PC such that the total variance explained  by the components is y%
        {2: z}: Identify Kth PC such that r := arg max (z) lambda(j) / lambda(j+1) where z=0 corresponds to the maximum
        :return: self.optimalfactors: optimal number of factors,
                 self.factors: PCA factors with optimal number of factors
                 self.components: PCA components with optimal number of factors
                 self.eigenvalues = PCA eigenvalues with optimal number of factors
        """

        if self.factorselection is not None:
            selection = list(self.factorselection.keys())[0]
            target = self.factorselection[selection]

            if selection == 0:
                additional_explained_variance = (
                    np.round((self.explained_variance_ratio), decimals=4) * 100
                )
                optimalfactors = np.argmax(additional_explained_variance < target) + 1

            elif selection == 1:
                cumvar = np.cumsum(
                    np.round(self.explained_variance_ratio, decimals=4) * 100
                )
                optimalfactors = np.argmax(cumvar > target) + 1

            elif selection == 2:
                additional_explained_variance = (
                    np.round((self.explained_variance_ratio), decimals=4) * 100
                )
                drops = (
                    additional_explained_variance[:-1]
                    / additional_explained_variance[1:]
                )
                optimalfactors = (-drops).argsort()[target] + 1

            self.optimalfactors = optimalfactors

            # update PCA by selecting only the optimal number of factors
            self.factors = self.factors.iloc[:, :optimalfactors]
            self.components = self.components.iloc[:, :optimalfactors]
            self.eigenvalues = self.eigenvalues[:optimalfactors]
            self.explained_variance_ = self.explained_variance[:optimalfactors]
            self.explained_variance_ratio = self.explained_variance_ratio[
                :optimalfactors
            ]

    def _fillmissing(self):
        """
        Fill missing values
        :param: handle_missing
        0: Forward Fill followed by Backward Fill missing values
        1: Fill missing values with mean of respective series
        :return: self.currentdata without missing values
                 self.originaldata with missing values
        """

        self.currentdata.drop(
            index=self.currentdata.index[[0, 1]], axis=0, inplace=True
        )
        self.original_input = self.currentdata.copy()

        if self.handle_missing == 0:
            self.currentdata = self.currentdata.ffill().bfill()

        elif self.handle_missing == 1:
            self.currentdata = self.currentdata.fillna(self.currentdata.mean())

    def _runPCAalgorithm(self):
        """
        Steps to run PCA on self.currentdata:
        1) standardize data
        2) run PCA with maximum factors = maxfactors
        3) Unstandardize data
        :return: self.factors: PCA factors,
                 self.filleddata: self.originaldata with missing values replaced by factor projections
                 self.components: PCA components,
                 self.eigenvalues: PCA eigenvalues
                 self.explained_variance: explained variance of PCA eigenvalues
                 self.explained_variance_ratio: % explained variance of PCA eigenvalues

        """
        col_names = list(self.original_input)
        index = self.original_input.index

        # standardize
        scalar = self._standardize()
        scaleddata = scalar.fit_transform(self.currentdata)

        # PCA
        pca = PCA(n_components=self.maxfactors)
        transformeddata = pca.fit_transform(scaleddata)

        # de-standardize to get factor projections
        unscaleddata = scalar.inverse_transform(transformeddata)
        unscaleddata = pd.DataFrame(
            columns=[f"Factor {i}" for i in range(1, len(pca.components_) + 1)],
            data=unscaleddata,
        )
        unscaleddata.set_index(index, inplace=True, drop=True)
        self.factors = unscaleddata

        # replace missing values in original data with factors projections
        temp = self.factors.copy()
        temp.columns = col_names
        self.filleddata = self.original_input.fillna(temp)

        # save relevant PCA outputs
        self.components = pd.DataFrame(
            columns=[f"Components {i}" for i in range(1, len(pca.components_) + 1)],
            data=pca.components_,
        )
        self.eigenvalues = pca.singular_values_
        self.explained_variance = pca.explained_variance_
        self.explained_variance_ratio = pca.explained_variance_ratio_
