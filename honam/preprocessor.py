from typing import Union, List, Tuple

import pandas as pd
import numpy as np

from sklearn.preprocessing import QuantileTransformer, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from honam._types import _TASK_INPUT

__all__ = ['Preprocessor']


class Preprocessor:

    """Preprocessor for input features and targets.

    Args:
        categorical_features: list of categorical features names.
        continuous_features: list of continuous feature names.
        task: target task name. one of 'reg', 'bincls', and 'multicls'.
    """

    def __init__(
        self,
        categorical_features: Union[None, List[str]],
        continuous_features: Union[None, List[str]],
        task: _TASK_INPUT
    ) -> None:
        
        # for categorical features
        self._ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
        self._categorical_imputer = SimpleImputer(strategy='most_frequent')

        # for continuous features
        self._continuous_imputer = SimpleImputer(strategy='mean')
        
        # for all features
        self._quantile_transformer = QuantileTransformer(output_distribution='uniform')

        # for targets
        if task == 'regression':
            self._target_transformer = StandardScaler()
        else:
            self._target_transformer = OrdinalEncoder()

        self._categorical_features = categorical_features
        self._continuous_features = continuous_features

    def fit(
        self,
        x: pd.DataFrame,
        y: np.ndarray
    ) -> Tuple[np.ndarray]:
        
        """Fit preprocessor and transform data.

        Args:
            x: input features.
            y: target values.

        Returns:
            transformed features and targets.
        """

        assert len(x.shape) == 2, 'x must be a 2D tensor.'
        assert len(y.shape) == 1 or len(y.shape) == 2, 'y must be a 1D or 2D tensor.'
        
        x = x.copy()
        y = y.copy()

        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        if self._categorical_features is not None:
            x[self._categorical_features] = self._ordinal_encoder.fit_transform(x[self._categorical_features])
            x[self._categorical_features] = self._categorical_imputer.fit_transform(x[self._categorical_features])

        if self._continuous_features is not None:
            x[self._continuous_features] = self._continuous_imputer.fit_transform(x[self._continuous_features])

        x = self._quantile_transformer.fit_transform(x)
        y = self._target_transformer.fit_transform(y)

        return x, y

    def transform(
        self,
        x: pd.DataFrame,
        y: np.ndarray
    ) -> Tuple[np.ndarray]:
        
        """Transform data.

        Returns:
            transformed features and targets.
        """
        
        assert len(x.shape) == 2, 'x must be a 2D tensor.'
        assert len(y.shape) == 1 or len(y.shape) == 2, 'y must be a 1D or 2D tensor.'
        
        x = x.copy()
        y = y.copy()

        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        if self._categorical_features is not None:
            x[self._categorical_features] = self._ordinal_encoder.transform(x[self._categorical_features])
            x[self._categorical_features] = self._categorical_imputer.transform(x[self._categorical_features])

        if self._continuous_features is not None:
            x[self._continuous_features] = self._continuous_imputer.transform(x[self._continuous_features])

        x = self._quantile_transformer.transform(x)
        y = self._target_transformer.transform(y)

        return x, y
