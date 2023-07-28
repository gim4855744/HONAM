import warnings
import logging
import os
import argparse

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split

from honam.fetchdata import DATASET_MAP
from honam.preprocessor import Preprocessor
from honam.utils import get_loader
from honam.models import HONAM
from honam.utils import evaluate, save_pickle, load_pickle

warnings.filterwarnings('ignore')
logging.disable(logging.WARNING)

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, choices=['train', 'test'], required=True)
parser.add_argument('--dataset', type=str, choices=DATASET_MAP.keys(), required=True)
parser.add_argument('--seed', type=int, choices=[0, 1, 2, 3, 4], required=True)
args = parser.parse_args()


def main():

    pl.seed_everything(args.seed)

    data = DATASET_MAP[args.dataset](fold=args.seed)

    categorical_features = data['categorical_features']
    continuous_features = data['continuous_features']
    n_features = data['n_features']
    n_outputs = data['n_outputs']
    
    task = data['task']
    ckpt_dir = f'./checkpoints/{args.dataset}'
    ckpt_filename = 'honam'
    ckpt_path = os.path.join(ckpt_dir, f'{ckpt_filename}.ckpt')
    
    if args.mode == 'train':

        x_train, y_train = data['x_train'], data['y_train']
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

        preprocessor = Preprocessor(categorical_features, continuous_features, task)
        x_train, y_train = preprocessor.fit(x_train, y_train)
        x_val, y_val = preprocessor.transform(x_val, y_val)

        os.makedirs(ckpt_dir, mode=0o755, exist_ok=True)
        save_pickle(preprocessor, os.path.join(ckpt_dir, 'preprocessor.pkl'))
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)

        checkpoint_callback = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename=ckpt_filename,
            monitor='val_loss'
        )
        earlystop_callback = EarlyStopping(
            monitor='val_loss',
            patience=20,
        )
        callbacks = [
            checkpoint_callback,
            earlystop_callback
        ]

        batch_size = len(x_train) // 100
        train_loader = get_loader(x_train, y_train, batch_size, shuffle=True)
        val_loader = get_loader(x_val, y_val, batch_size)

        hidden_dims = [32, 64, 32]
        model = HONAM(n_features, n_outputs, hidden_dims)
        trainer = pl.Trainer(max_epochs=1000, precision='32-true', logger=False, callbacks=callbacks)
        trainer.fit(model, train_loader, val_loader)

    elif args.mode == 'test':

        preprocessor = load_pickle(os.path.join(ckpt_dir, 'preprocessor.pkl'))

        x_test, y_test = data['x_test'], data['y_test']
        x_test, y_test = preprocessor.transform(x_test, y_test)
        
        x_test = torch.tensor(x_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)

        model = HONAM.load_from_checkpoint(ckpt_path).to('cpu')
        y_hat = model(x_test).detach()

        evaluate(y_test, y_hat, task)


if __name__ == '__main__':
    main()
