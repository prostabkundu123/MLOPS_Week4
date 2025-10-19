import pandas as pd
import pytest

def test_data_integrity():
    df = pd.read_csv("data/iris.csv")
    # 1. Check shape
    assert df.shape[1] == 5
    # 2. Check missing values
    assert df.isnull().sum().sum() == 0
    # 3. Check expected columns
    expected_cols = {"sepal_length", "sepal_width", "petal_length", "petal_width", "species"}
    assert expected_cols.issubset(df.columns)

def test_numeric_values_positive():
    df = pd.read_csv("data/iris.csv")
    numeric_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    # 4. Check all numeric values are positive
    for col in numeric_cols:
        assert (df[col] > 0).all(), f"Negative or zero values found in {col}"