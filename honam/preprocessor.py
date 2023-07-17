import numpy as np

from sklearn.preprocessing import StandardScaler, QuantileTransformer, OrdinalEncoder
from sklearn.impute import SimpleImputer

__all__ = ['Preprocessor']


class FeatureTransformer:

    def __init__(self, categorical_features):

        self._categorical_features = categorical_features

        self._ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
        self._imputer = SimpleImputer(strategy='most_frequent', copy=False)
        
        self._quantile_transformer = QuantileTransformer(output_distribution='uniform', copy=False)

    def fit_transform(self, x):

        x = x.copy()

        if self._categorical_features is not None:
            x[self._categorical_features] = self._ordinal_encoder.fit_transform(x[self._categorical_features])
            x[self._categorical_features] = self._imputer.fit_transform(x[self._categorical_features])

        # quantile_noise = 1e-3
        # stds = np.std(x, axis=0, keepdims=True)
        # noise_std = quantile_noise / np.maximum(stds, quantile_noise)
        # quantile_x = x + noise_std * np.random.randn(*x.shape)
        # self._quantile_transformer.fit(quantile_x)
        # x = self._quantile_transformer.transform(x)

        x = self._quantile_transformer.fit_transform(x)

        return x
    
    def transform(self, x):

        x = x.copy()

        if self._categorical_features is not None:
            x[self._categorical_features] = self._ordinal_encoder.transform(x[self._categorical_features])
            x[self._categorical_features] = self._imputer.transform(x[self._categorical_features])
            
        x = self._quantile_transformer.transform(x)

        return x


class TargetTransformer:

    def __init__(self, task):
        if task == 'regression':
            self._transformer = StandardScaler()
        elif task == 'classification':
            self._transformer = OrdinalEncoder()
        else:
            raise ValueError('task must be regression or classification')

    def fit_transform(self, y):
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        y = self._transformer.fit_transform(y)
        return y

    def transform(self, y):
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        y = self._transformer.transform(y)
        return y


class Preprocessor:

    def __init__(self, categorical_features, task):
        self._x_transformer = FeatureTransformer(categorical_features)
        self._y_transformer = TargetTransformer(task)

    def fit_transform(self, x, y):
        x = self._x_transformer.fit_transform(x)
        y = self._y_transformer.fit_transform(y)
        return x, y

    def transform(self, x, y):
        x = self._x_transformer.transform(x)
        y = self._y_transformer.transform(y)
        return x, y
