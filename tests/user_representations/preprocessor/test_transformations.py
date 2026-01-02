import pytest
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from user_representations.preprocessor.transformations import (
    drop_na_and_duplicates,
    column_list_transform,
    shuffle_data,
    select_first_item_from_list_column,
    make_transformer,
    make_preprocessor,
)


def test_drop_na_and_duplicates_removes_nans_and_duplicates():
    df = pd.DataFrame(
        {
            "name": ["a", "a", "b", None],
            "year": [2020, 2020, 2021, 2022],
        }
    )

    cleaned = drop_na_and_duplicates(df, subset=["name"])

    # No NaNs in "name"
    assert cleaned["name"].isna().sum() == 0
    # Duplicated "a" row removed
    assert (cleaned["name"] == "a").sum() == 1
    # Original dataframe is not modified in-place
    assert df.shape != cleaned.shape


def test_column_list_transform_parses_stringified_lists():
    df = pd.DataFrame(
        {
            "artists": ["['A', 'B']", "['C']"],
        }
    )

    series = column_list_transform(df, "artists")

    assert isinstance(series.iloc[0], list)
    assert series.iloc[0] == ["A", "B"]
    assert series.iloc[1] == ["C"]


def test_shuffle_data_is_deterministic_with_random_state():
    df = pd.DataFrame({"x": list(range(10))})

    shuffled1 = shuffle_data(df, random_state=42)
    shuffled2 = shuffle_data(df, random_state=42)

    # Same ordering for same random_state
    pd.testing.assert_frame_equal(shuffled1, shuffled2)
    # Order is different from original
    assert not shuffled1["x"].equals(df["x"])


def test_select_first_item_from_list_column():
    series = pd.Series([["A", "B"], ["C"], ["D", "E", "F"]])

    first_items = select_first_item_from_list_column(series)

    assert list(first_items) == ["A", "C", "D"]


def test_make_transformer_with_standard_scaler():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})

    transformer = make_transformer([StandardScaler()])
    X = transformer.fit_transform(df)

    # Mean of transformed column should be ~0
    assert np.allclose(X.mean(), 0.0, atol=1e-7)


def test_make_preprocessor_runs_and_returns_2d_array():
    df = pd.DataFrame(
        {
            "num": [1.0, 2.0, 3.0, 4.0],
            "cat": ["a", "b", "a", "b"],
        }
    )

    num_transformer = make_transformer([StandardScaler()])
    cat_transformer = make_transformer([OneHotEncoder(drop="if_binary")])

    preprocessor = make_preprocessor(
        [
            (num_transformer, ["num"]),
            (cat_transformer, ["cat"]),
        ]
    )

    X = preprocessor.fit_transform(df)

    # Basic shape checks: correct number of samples and 2D output
    assert X.shape[0] == len(df)
    assert len(X.shape) == 2
