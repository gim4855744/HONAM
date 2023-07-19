import warnings

import pytorch_lightning as pl
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from honam.models import HONAM
from honam.data import get_loader
from honam.fetchdata import fetch_BIKESHARE

warnings.filterwarnings('ignore', '.*does not have many workers.*')


def main():

    data = fetch_BIKESHARE()
    x_train, y_train = data['X_train'], data['y_train']
    x_test, y_test = data['X_test'], data['y_test']
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)
    
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    batch_size = 32
    train_loader = get_loader(x_train, y_train, batch_size, shuffle=True)
    val_loader = get_loader(x_val, y_val, batch_size)
    test_loader = get_loader(x_test, y_test, batch_size)
    
    n_features = x_train.shape[-1]
    n_outputs = 1
    hidden_dims = [32, 64]

    model = HONAM(n_features, n_outputs, hidden_dims)
    trainer = pl.Trainer(max_epochs=10, precision='32-true', logger=False, enable_checkpointing=False, enable_model_summary=True)
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    main()
