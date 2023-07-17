import argparse
import os

import torch
from sklearn.model_selection import train_test_split

from honam.fetchdata import DATASETS
from honam.preprocessor import Preprocessor
from honam.model import *
from honam.utils import set_global_seed, evaluate

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, choices=DATASETS.keys(), required=True, help='dataset name')
parser.add_argument('--seed', type=int, required=True, help='random seed, 0<=seed<=4')
parser.add_argument('--mode', type=str, required=True, choices=['train', 'test'], help='train or test')
parser.add_argument('--order', default=2, type=int, help='interaction order, only used for HONAM, HONAMCrossNet, and CrossNet')
parser.add_argument('--num_workers', default=0, type=int, help='number of workers of dataloader')
parser.add_argument('--device', default='cpu', type=str, choices=['cpu', 'cuda', 'mps'], help='device name')
args = parser.parse_args()


def main():

    assert 0 <= args.seed <= 4, f'the random seed is only allowed between 0 and 4, but get {args.seed}'

    set_global_seed(args.seed)
    
    data = DATASETS[args.dataset](fold=args.seed)
    x_train, y_train = data['X_train'], data['y_train']
    x_test, y_test = data['X_test'], data['y_test']

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

    categorical_features = data['cat_features']
    task = data['problem']
    n_features = x_train.shape[1]

    os.makedirs('./checkpoints/', mode=0o755, exist_ok=True)
    ckpt_path = f'./checkpoints/{args.dataset}_{args.seed}.pt'

    preprocessor = Preprocessor(categorical_features, task)
    x_train, y_train = preprocessor.fit_transform(x_train, y_train)
    x_val, y_val = preprocessor.transform(x_val, y_val)
    x_test, y_test = preprocessor.transform(x_test, y_test)
    
    if args.mode == 'train':
        hparam = {'n_features': n_features, 'order': args.order, 'task': task, 'ckpt_path': ckpt_path}
        model = HONAM(**hparam).to(args.device)
        model.fit(x_train, y_train, x_val, y_val, num_workers=args.num_workers)
    else:
        checkpoint= torch.load(ckpt_path)
        model = checkpoint['model'].to(args.device)
        model.load_state_dict(checkpoint['state_dict'])
        test_y_hat = model.predict(x_test)
        evaluate(y_test, test_y_hat, task)


if __name__ == '__main__':
    main()
