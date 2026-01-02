import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from user_representations.preprocessor.preprocessor import (
    PreprocessorConfig,
    TrackPreprocessor,
    export_to_csv,
)


def _make_dummy_pipeline():
    """Small helper to create a simple numeric pipeline."""
    return make_pipeline(StandardScaler())


def test_preprocessor_config_uses_export_to_csv_by_default(tmp_path: Path):
    pipeline = _make_dummy_pipeline()
    config = PreprocessorConfig(preprocessor=pipeline)

    # Sanity: default exporter is the module-level function
    assert config.exporter is export_to_csv

    # And it actually writes a CSV when used via TrackPreprocessor.export
    df = pd.DataFrame({"x": [1, 2, 3]})
    pre = TrackPreprocessor(config)
    out_path: Path = tmp_path / "out.csv"
    pre.export(df, out_path)

    assert out_path.exists()
    written = pd.read_csv(out_path)
    pd.testing.assert_frame_equal(written, df)


def test_clean_data_applies_all_cleaning_steps_in_order():
    calls = []

    def step_add_col_a(df: pd.DataFrame) -> pd.DataFrame:
        calls.append("a")
        df = df.copy()
        df["a"] = 1
        return df

    def step_add_col_b(df: pd.DataFrame) -> pd.DataFrame:
        calls.append("b")
        df = df.copy()
        df["b"] = 2
        return df

    df = pd.DataFrame({"x": [0, 1]})
    config = PreprocessorConfig(
        preprocessor=_make_dummy_pipeline(),
        cleaning_steps=[step_add_col_a, step_add_col_b],
    )
    pre = TrackPreprocessor(config)

    cleaned = pre.clean_data(df)

    # Steps applied in order
    assert calls == ["a", "b"]
    # Columns added by both steps
    assert list(cleaned.columns) == ["x", "a", "b"]
    assert (cleaned["a"] == 1).all()
    assert (cleaned["b"] == 2).all()


def test_fit_and_transform_return_dataframe_with_same_row_count():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    config = PreprocessorConfig(preprocessor=_make_dummy_pipeline())
    pre = TrackPreprocessor(config)

    pre_fitted = pre.fit(df)
    assert pre_fitted is pre

    X = pre.transform(df)

    assert isinstance(X, pd.DataFrame)
    assert X.shape[0] == len(df)
    # StandardScaler should roughly zero-center the data
    assert np.allclose(X.values.mean(), 0.0, atol=1e-7)


def test_export_uses_custom_exporter(tmp_path: Path):
    calls = []

    def custom_exporter(df: pd.DataFrame, path: Path):
        # Record call instead of touching disk
        calls.append((df.copy(), path))

    df = pd.DataFrame({"x": [10, 20]})
    dummy_pipeline = _make_dummy_pipeline()
    config = PreprocessorConfig(
        preprocessor=dummy_pipeline,
        exporter=custom_exporter,
    )
    pre = TrackPreprocessor(config)
    out_path: Path = tmp_path / "custom.csv"

    pre.export(df, out_path)

    assert len(calls) == 1
    recorded_df, recorded_path = calls[0]
    pd.testing.assert_frame_equal(recorded_df, df)
    assert recorded_path == out_path
