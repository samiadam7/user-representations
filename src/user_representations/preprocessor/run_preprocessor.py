from pathlib import Path
from functools import partial

import numpy as np
import pandas as pd
from sklearn.compose import make_column_selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import user_representations.preprocessor.transformations as transformations
from user_representations.preprocessor.preprocessor import TrackPreprocessor, PreprocessorConfig, export_to_csv
from user_representations.logging_config.logging_config import setup_logging

config_file = Path(__file__).resolve().parents[2] / "logging_config" / "config.json"
setup_logging(level="DEBUG", experiment_name="preprocessor", config_file=config_file)

META_COLS = ["name", "album", "main_artist", "track_number", "disc_number", "release_date", "key"]
FEATURE_COLS = ["explicit", "danceability", "energy", "loudness",
              "speechiness", "acousticness", "instrumentalness",
              "liveness", "valence", "tempo"]


def read_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def get_main_artist_column(df: pd.DataFrame, column: str) -> pd.Series:
    col_series = transformations.column_list_transform(df, column)
    col_series = transformations.select_first_item_from_list_column(col_series)
    return col_series
    

def main():
    # ----------------------------- read data -----------------------------
    data_file_path = Path("data/raw/tracks.csv")
    df = read_data(data_file_path)
    df["main_artist"] = get_main_artist_column(df, "artists")
    
    # ----------------------------- preprocess data -----------------------------
    num_transformer = transformations.make_transformer([StandardScaler()])
    cat_transformer = transformations.make_transformer([OneHotEncoder(drop="if_binary")])
    
    preprocessor = transformations.make_preprocessor([
        (num_transformer, make_column_selector(dtype_include=np.number)),
        (cat_transformer, make_column_selector(dtype_exclude=np.number)),
    ])
    cleaning_steps = [partial(transformations.drop_na_and_duplicates, subset=["name", "main_artist", "year"]), transformations.shuffle_data]
    
    config = PreprocessorConfig(
        cleaning_steps=cleaning_steps,
        preprocessor=preprocessor
    )
    
    track_preprocessor = TrackPreprocessor(config)
    cleaned_df = track_preprocessor.clean_data(df)
    
    meta_df = cleaned_df[META_COLS]
    export_to_csv(meta_df, Path("data/processed/meta.csv"))
    feature_df = cleaned_df[FEATURE_COLS]
    
    X = track_preprocessor.fit_transform(feature_df)
    track_preprocessor.export(X, Path("data/processed/tracks.csv"))


if __name__ == "__main__":
    main()


