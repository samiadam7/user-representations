import logging
from pathlib import Path
from typing import Callable, List, Self
from dataclasses import dataclass, field

import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

type CleaningStep = Callable[[pd.DataFrame], pd.DataFrame]
type Exporter = Callable[[pd.DataFrame, Path], None]


def _get_step_name(step: CleaningStep) -> str:
    """Return a readable name for a cleaning step (handles functions and partials)."""
    if hasattr(step, "__name__"):
        return step.__name__
    func = getattr(step, "func", None)  # functools.partial and similar
    if func is not None and hasattr(func, "__name__"):
        return func.__name__
    return repr(step)


def export_to_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Exporting to CSV located at {path}")
    df.to_csv(path, index=False)


@dataclass
class PreprocessorConfig:
    preprocessor: Pipeline
    cleaning_steps: List[CleaningStep] = field(default_factory=list)
    exporter: Exporter = field(default=export_to_csv)


class TrackPreprocessor(TransformerMixin):
    def __init__(self, config: PreprocessorConfig):
        self.preprocessor = config.preprocessor
        self.cleaning_steps = config.cleaning_steps
        self.exporter = config.exporter
        logger.debug(
            "Initialized TrackPreprocessor with %d cleaning steps",
            len(self.cleaning_steps),
        )

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        steps = len(self.cleaning_steps)
        for i, cleaning_step in enumerate(self.cleaning_steps, 1):
            logger.debug(
                "Cleaning step %d/%d: %s",
                i,
                steps,
                _get_step_name(cleaning_step),
            )
            df = cleaning_step(df)
        logger.info("Cleaning finished. DataFrame shape: %s", df.shape)
        return df

    def fit(self, df: pd.DataFrame) -> Self:
        logger.info("Fitting preprocessor on data with shape: %s", df.shape)
        self.preprocessor.fit(df)
        logger.info("Finished fitting preprocessor.")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.debug("Transforming data with shape: %s", df.shape)
        X = self.preprocessor.transform(df)

        try:
            column_names = self.preprocessor.get_feature_names_out()
            X = pd.DataFrame(X, columns=column_names)
        except ValueError:
            logger.warning(
                "Could not get feature names from preprocessor. Using default column names."
            )
            X = pd.DataFrame(X)

        logger.debug("Transformed data to shape: %s", X.shape)
        return X

    def export(self, X: pd.DataFrame, path: Path) -> None:
        logger.info("Exporting transformed data with shape %s to %s", X.shape, path)
        self.exporter(X, path)
