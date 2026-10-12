import pandas as pd
import numpy as np
import datetime
import warnings
import sys


class CovarianceFred:
    def __init__(
        self,
        data: pd.DataFrame,
    ):
        """Compute the covariance matrix of FRED data with options for various techniques available. Covariance matrices may be used as inputs for other data analysis.
        :param: FRED-MD data

        4 functions may be called:
        sample_covariance(): compute raw (sample) covariance matrix of data
        threshold_covariance(): compute sample covariance matrix of data with a minimum correlation theshold specified by user
        positive_semidefinite_method1(): converts the sample covariance matrix to a positive semidefinite matrix by setting all negative eigenvalues to 0
        positive_semidefinite_method2(): converts the sample covariance matrix to a positive semidefinite matrix by setting cov_matrix = (sample_cov + abs(lambda_min)*I) / 1+abs(lambda_min)
        where lambdas are the eigenvalues
        """

        self.originaldata = data

    def sample_covariance(self) -> pd.DataFrame:
        """Compute sample, raw covariance matrix
        return: Pandas DataFrame"""

        self.covariancematrix = self.originaldata.cov()

        return self.covariancematrix

    def threshold_covariance(self, correlationthreshold=0.01) -> pd.DataFrame:
        """Compute covariance matrix with thresholding of a minimum correlation between any 2 assets.
        Correlation of any assets below the minimum threshold is replaced by 0
        :param correlationthreshold: correlation threshold integer
        :return: Pandas DataFrame"""

        assert (
            abs(correlationthreshold) <= 1
        ), "Correlation threshold must range from -1 to 1"

        self.covariancematrix = self.originaldata.cov()[
            abs(self.originaldata.corr()) > correlationthreshold
        ]
        self.covariancematrix.fillna(0, inplace=True)
        return self.covariancematrix

    def positive_semidefinite_method1(self, correlationthreshold=0.0) -> pd.DataFrame:
        """Compute covariance matrix with thresholding and making sure that the matrix is positive semidefinite
        The method sets the eigenvalues of the matrix as max(0, eigenvalues)
        :param correlationthreshold: correlation threshold integer
        :return: Pandas DataFrame"""

        assert (
            abs(correlationthreshold) <= 1
        ), "Correlation threshold must range from -1 to 1"

        sample_covariance = self.originaldata.cov()[
            abs(self.originaldata.corr()) > correlationthreshold
        ]
        sample_covariance.fillna(0, inplace=True)

        # do eigendecomposition
        Lambda, U = np.linalg.eig(sample_covariance)

        # clean eigenvalues to max(eigenvalues,0)
        Lambda_pos = np.where(Lambda < 0, 0, Lambda)

        # reconstruct covariance matrix
        inv_U = np.linalg.inv(U)
        A = np.diag(Lambda_pos)
        self.covariancematrix = pd.DataFrame(np.dot(U, np.dot(A, inv_U)))

        return self.covariancematrix

    def positive_semidefinite_method2(self, correlationthreshold=0.0) -> pd.DataFrame:
        """Compute covariance matrix with thresholding and making sure that the matrix is positive semidefinite
        The method sets the covariance matrix = (sample_cov + abs(lambda_min)*I) / 1+abs(lambda_min)
        :param correlationthreshold: correlation threshold integer
        :return: Pandas DataFrame"""

        assert (
            abs(correlationthreshold) <= 1
        ), "Correlation threshold must range from -1 to 1"

        sample_covariance = self.originaldata.cov()[
            abs(self.originaldata.corr()) > correlationthreshold
        ]
        sample_covariance.fillna(0, inplace=True)

        # do eigendecomposition
        Lambda, U = np.linalg.eig(sample_covariance)

        # transform sample covariance matrix
        g = max(-Lambda.min(), 0)
        self.covariancematrix = (
            sample_covariance + (g * np.identity(len(sample_covariance)))
        ) / (1 + g)

        return self.covariancematrix
