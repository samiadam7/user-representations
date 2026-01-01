import pandas as pd
import ast
from typing import List, Any
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.base import TransformerMixin
from sklearn.decomposition import PCA


def drop_na_and_duplicates(df:pd.DataFrame, subset:List[str]) -> pd.DataFrame:
    return df.dropna().drop_duplicates(subset=subset).copy()
    
    
def column_list_transform(df:pd.DataFrame, columns:List[str]) -> pd.DataFrame:
    return df[columns].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)


def shuffle_data(df:pd.DataFrame, random_state:int | None = 42) -> pd.DataFrame:
    try:
        return df.sample(frac=1, random_state=random_state)
    except:
        return df.sample(frac=1)


def select_first_item_from_list_column(df:pd.DataFrame) -> pd.DataFrame:
    return df.apply(lambda x: x[0] if isinstance(x, list) else x)


def make_transformer(steps: List[TransformerMixin]) -> Pipeline:
    return make_pipeline(*steps)


def make_preprocessor(steps:[tuple[TransformerMixin, Any]], remainder:str = "drop") -> Pipeline:
    column_transfomer = make_column_transformer(*steps, remainder=remainder)
    full_pipeline = Pipeline([
        ("column_transformer", column_transfomer),
        ("PCA", PCA(whiten=True))
    ])
    return full_pipeline

