import pandas as pd

from sklearn.model_selection import train_test_split

__all__ = ['Dataset', 'CaliforniaHousing', 'GraduateAdmission', 'Credit']


class Dataset:

    def __init__(self, path, test_rate, target_column):

        self._dataframe = pd.read_csv(path)
        self._dataframe = self._dataframe.dropna(axis=1)

        self._test_rate = test_rate
        self._target_column = target_column

    def load(self):

        num_data = len(self._dataframe)
        test_size = self._test_rate * num_data
        test_size = int(test_size)

        x_train, x_test = train_test_split(self._dataframe, test_size=test_size)
        x_train, x_val = train_test_split(x_train, test_size=test_size)

        y_train = x_train[[self._target_column]]
        y_val = x_val[[self._target_column]]
        y_test = x_test[[self._target_column]]

        x_train.drop(self._target_column, axis=1)
        x_val.drop(self._target_column, axis=1)
        x_test.drop(self._target_column, axis=1)

        return (x_train, y_train), (x_val, y_val), (x_test, y_test)

    @property
    def task(self):
        raise NotImplementedError

    @property
    def num_features(self):
        raise NotImplementedError

    @property
    def out_size(self):
        raise NotImplementedError


class CaliforniaHousing(Dataset):

    """
    This dataset can be downloaded from
        https://www.kaggle.com/datasets/camnugent/california-housing-prices.
    """

    def __init__(self, path, test_rate=0.2, target_column='median_house_value'):
        super().__init__(path, test_rate, target_column)

    @property
    def task(self):
        return 'reg'

    @property
    def num_features(self):
        return 8

    @property
    def out_size(self):
        return 1


class GraduateAdmission(Dataset):

    """
    This dataset can be downloaded from
        https://www.kaggle.com/datasets/mohansacharya/graduate-admissions
    """

    def __init__(self, path, test_rate=0.2, target_column='Chance of Admit '):
        super().__init__(path, test_rate, target_column)
        self._dataframe = self._dataframe.drop(['Serial No.'], axis=1)

    @property
    def task(self):
        return 'reg'


class Credit(Dataset):

    """
    (Dal Pozzolo, A. 2015. Adaptive Machine Learning for Credit Card Fraud Detection.) released this dataset.
    The dataset can be downloaded from
        https://www.ulb.ac.be/di/map/adalpozz/data/creditcard.Rdata
    """

    def __init__(self, path, test_rate=0.2, target_column='Class'):
        super().__init__(path, test_rate, target_column)
        self._dataframe = self._dataframe.drop(['Unnamed: 0', 'Time'], axis=1)

    @property
    def task(self):
        return 'bin_cls'
