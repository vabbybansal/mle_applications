import unittest
from .. import data_manipulation
import pandas as pd

class TestDataManipulation(unittest.TestCase):

    # Run before and after class instantiation
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    # Run before and after every test
    def setUp(self):
        self.temp_series = pd.Series(['a','b','c'], name='alphabet')

    def tearDown(self):
        pass

    def test_convert_col_one_hot_series(self):

        returned_out = data_manipulation.convert_col_one_hot_series(self.temp_series)

        # 0th element of return type of a valid categorical series is a pandas data frame upon conversion
        self.assertEqual(isinstance(returned_out[0], pd.core.frame.DataFrame), True)

        # Check number of columns of the returned dataframe from the categorical series
        self.assertEqual(returned_out[0].shape[0], 3)

        # Check column name validity
        returned_columns = returned_out[0].columns
        self.assertEqual(('alphabet_a' in returned_columns and 'alphabet_b' in returned_columns and 'alphabet_c' in returned_columns), True)

        # Check return code from a successful run
        self.assertEqual(returned_out[1], 1)

        empty_series_out = data_manipulation.convert_col_one_hot_series(pd.Series())
        self.assertEqual(empty_series_out[0], None)
        self.assertEqual(empty_series_out[1], 0)

        # self.assertRaises(ValueError, funcName, arg1. arg2)
        # with self.assertRaises(ValueError):
            # funcName(arg1, arg2)