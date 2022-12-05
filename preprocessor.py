import numpy as np

from sklearn.preprocessing import OrdinalEncoder, QuantileTransformer, MinMaxScaler
from typing import Tuple, List

class Preprocessor:

    def __init__(self, task: str, n_quantiles: int=1000):

        """
        :param task: regression or binary_classification
        :param n_quantiles: the number of quantiles
        """

        self._task = task

        self._ordinary_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        self._quantile_transformer = QuantileTransformer(n_quantiles=n_quantiles)
        self._minmax_scaler = MinMaxScaler(feature_range=(-1, 1))

        if task == "regression":
            self._y_transformer = MinMaxScaler(feature_range=(-1, 1))
        elif task == "binary_classification":
            self._y_transformer = OrdinalEncoder()

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:

        """
        Fit preprocessor.
        :param x: feature
        :param y: target
        """

        self._transform_xy(x, y, fit=True)

    def transform(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        """
        Transform x and y.
        :param x: feature
        :param y: target
        :return: transformed x and y
        """

        x, y = self._transform_xy(x, y)

        return x, y

    def fit_transform(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        """
        Fit preprocessor then transform x and y.
        :param x: feature
        :param y: target
        :return: transformed x and y
        """

        x, y = self._transform_xy(x, y, fit=True)

        return x, y

    def _transform_xy(self, x: np.ndarray, y: np.ndarray, fit: bool=False) -> Tuple[np.ndarray, np.ndarray]:

        x, y = x.copy(), y.copy()
        cat_idx = self._get_cat_idx(x)

        if fit:
            if cat_idx:
                x[:, cat_idx] = self._ordinary_encoder.fit_transform(x[:, cat_idx])
            x = self._quantile_transformer.fit_transform(x)
            x = self._minmax_scaler.fit_transform(x)
            y = self._y_transformer.fit_transform(y)
        else:
            if cat_idx:
                x[:, cat_idx] = self._ordinary_encoder.transform(x[:, cat_idx])
            x = self._quantile_transformer.transform(x)
            x = self._minmax_scaler.transform(x)
            y = self._y_transformer.transform(y)

        return x, y

    @ staticmethod
    def _get_cat_idx(x: np.ndarray) -> List[int]:

        """
        Get categorical features' indices.
        :param x: feature
        :return: categorical features' indices
        """

        cat_idx = []
        num_features = x.shape[1]
        for i in range(num_features):
            try:
                x[:, i].astype(float)
            except ValueError:
                cat_idx.append(i)

        return cat_idx
