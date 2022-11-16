import pytest
import unittest
import numpy as np
import pandas as pd
from golden_fred import fred_regression


class test_AR_Model(unittest.TestCase):
    def c(
        self,
        data=None,
        max_lag=492,
        start_date=pd.to_datetime("2010-01"),
        end_date=pd.to_datetime("2021-11"),
        dependent_variable_name="CPIAUCSL",
        window_size=492,
    ):

        AR_model = fred_regression.AR_Model

        return AR_model(
            data=data,
            max_lag=max_lag,
            start_date=start_date,
            end_date=end_date,
            dependent_variable_name=dependent_variable_name,
            window_size=window_size,
        )

    def create_input(self):
        df = pd.DataFrame(
            columns=["date", "col1", "col2", "result"],
            data=[
                [pd.to_datetime("2010-01"), 1, 2, 4],
                [pd.to_datetime("2010-02"), 2, 2, 6],
            ],
        )
        return df

    def test_class_initialisation(self):
        model = self.c()
        self.assertEqual(model.start_date, pd.to_datetime("2010-01"))
        self.assertEqual(model.end_date, pd.to_datetime("2021-11"))
        self.assertEqual(model.dependent_variable_name, "CPIAUCSL")
        self.assertEqual(model.lag_patience, 5)

    def test_features_and_target(self):

        df = self.create_input()

        model = self.c(dependent_variable_name="result", data=df)

        model.features_and_target()

        self.assertEqual(len(model.features), 2)
        self.assertEqual(len(model.features.axes[1]), 3)

    def test_get_error(self):
        model = self.c()
        err = model.get_error(np.array([1, 2]), np.array([0, 1]))
        self.assertEqual(err, 1)


if __name__ == "__main__":
    unittest.main()
