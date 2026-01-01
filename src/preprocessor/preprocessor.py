from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
import pandas as pd
from typing import Callable, List
from dataclasses import dataclass, field
from pathlib import Path


type CleaningStep = Callable[[pd.DataFrame], pd.DataFrame]
type Exporter = Callable[[pd.DataFrame, Path], None]


def export_to_csv(df:pd.DataFrame, path:Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


@dataclass
class PreprocessorConfig:
    preprocessor:Pipeline
    cleaning_steps:List[CleaningStep] = field(default_factory=list)
    exporter:Exporter = field(default_factory=export_to_csv)
    

class TrackPreprocessor(TransformerMixin):
    def __init__(self, config:PreprocessorConfig):
        self.preprocessor = config.preprocessor
        self.cleaning_steps = config.cleaning_steps
        self.exporter = config.exporter
        
        
    def clean_data(self, df:pd.DataFrame) -> pd.DataFrame:
        for cleaning_step in self.cleaning_steps:
            df = cleaning_step(df)
        return df
    
    
    def fit(self, df:pd.DataFrame):
        X_clean = self.clean_data(df)
        self.preprocessor.fit(X_clean)
    
        
    def transform(self, df:pd.DataFrame) -> pd.DataFrame:
        X = self.clean_data(df)
        X = self.preprocessor.transform(X)
    
        try:
            column_names = self.preprocessor.get_feature_names_out()
            X = pd.DataFrame(X, columns=column_names)
        except:
            X = pd.DataFrame(X)
        
        return X
    
    
    def export(self, X:pd.DataFrame, path:Path) -> None:
        self.exporter(X, path)
        
