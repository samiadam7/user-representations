import ast
import logging
from typing import Any, List

import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.compose import make_column_transformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline, make_pipeline

logger = logging.getLogger(__name__)


def drop_na_and_duplicates(df:pd.DataFrame, subset:List[str]) -> pd.DataFrame:
    before = df.shape
    df_clean = df.dropna().drop_duplicates(subset=subset).copy()
    logger.debug(
        "drop_na_and_duplicates: subset=%s, before=%s, after=%s",
        subset,
        before,
        df_clean.shape,
    )
    return df_clean
    
    
def column_list_transform(df:pd.DataFrame, columns:List[str] | str) -> pd.DataFrame:
    logger.debug("column_list_transform: columns=%s", columns)
    return df[columns].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )


def shuffle_data(df:pd.DataFrame, random_state:int | None = 42) -> pd.DataFrame:
    try:
        shuffled = df.sample(frac=1, random_state=random_state)
        logger.debug(
            "shuffle_data: shuffled with random_state=%s, shape=%s",
            random_state,
            shuffled.shape,
        )
        return shuffled
    except:
        shuffled = df.sample(frac=1)
        logger.debug(
            "shuffle_data: shuffled without random_state, shape=%s",
            shuffled.shape,
        )
        return shuffled


def select_first_item_from_list_column(df:pd.DataFrame) -> pd.DataFrame:
    logger.debug("select_first_item_from_list_column: input shape=%s", df.shape)
    result = df.apply(lambda x: x[0] if isinstance(x, list) else x)
    logger.debug("select_first_item_from_list_column: output shape=%s", result.shape)
    return result


def make_transformer(steps: List[TransformerMixin]) -> Pipeline:
    logger.debug("make_transformer: building pipeline with %d steps", len(steps))
    return make_pipeline(*steps)


def make_preprocessor(steps:[tuple[TransformerMixin, Any]], remainder:str = "drop") -> Pipeline:
    logger.info(
        "make_preprocessor: building with %d steps, remainder=%s",
        len(steps),
        remainder,
    )
    column_transfomer = make_column_transformer(*steps, remainder=remainder)
    full_pipeline = Pipeline([
        ("column_transformer", column_transfomer),
        ("PCA", PCA(whiten=True))
    ])
    return full_pipeline

