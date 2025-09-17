import warnings
import logging
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from matplotlib.colors import CenteredNorm

from honam.fetchdata import DATASETS
from honam.preprocessor import Preprocessor
from honam.model import HONAM
from honam.utils import set_global_seed, evaluate

warnings.filterwarnings('ignore')
logging.disable(logging.WARNING)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, choices=DATASETS.keys(), required=True, help='dataset name')
parser.add_argument('--mode', type=str, required=True, choices=['train', 'test', 'interpret'], help='one of train, test or interpret')
parser.add_argument('--emb_size', type=int, default=32, help='size of feature representation')
parser.add_argument('--order', default=2, type=int, help='interaction order')
parser.add_argument('--seed', type=int, required=True, help='random seed, 0 <= seed <= 4')
parser.add_argument('--num_workers', default=0, type=int, help='number of workers of dataloader')
parser.add_argument('--device', default='cpu', type=str, help='device name')
args = parser.parse_args()


def main():

    assert 0 <= args.seed <= 4, f'the random seed is only allowed between 0 and 4, but get {args.seed}'

    set_global_seed(args.seed)
    
    data = DATASETS[args.dataset](fold=args.seed)
    x_train, y_train = data['X_train'], data['y_train']
    x_test, y_test = data['X_test'], data['y_test']
    feature_names = x_test.columns

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

    categorical_features = data['cat_features']
    task = data['problem']
    n_features = x_train.shape[1]
    ckpt_path = f'./checkpoints/{args.dataset}_honam{args.order}_emb{args.emb_size}_{args.seed}.pt'

    preprocessor = Preprocessor(categorical_features, task)
    x_train, y_train = preprocessor.fit_transform(x_train, y_train)
    x_val, y_val = preprocessor.transform(x_val, y_val)
    x_test, y_test = preprocessor.transform(x_test, y_test)
    
    if args.mode == 'train':
        hparam = {'n_features': n_features, 'emb_size': args.emb_size, 'order': args.order, 'task': task, 'ckpt_path': ckpt_path}
        model = HONAM(**hparam).to(args.device)
        model.fit(x_train, y_train, x_val, y_val, num_workers=args.num_workers)

    elif args.mode == 'test':
        checkpoint= torch.load(ckpt_path, map_location=args.device)
        model = checkpoint['model']
        model.load_state_dict(checkpoint['state_dict'])
        test_y_hat = model.predict(x_test)
        evaluate(y_test, test_y_hat, task)

    elif args.mode == 'interpret':

        os.makedirs('./interpretations/', exist_ok=True)

        checkpoint= torch.load(ckpt_path, map_location=args.device)
        model = checkpoint['model']
        model.load_state_dict(checkpoint['state_dict'])

        # Visualize the local first-order feature interactions for the first sample.
        first_order_interactions = []
        for i in range(n_features):
            contributions = model.interpret(x_train, i)
            first_order_interactions.append(contributions)
        first_order_interactions = np.concatenate(first_order_interactions, axis=1)

        plt.bar(feature_names, first_order_interactions[0], color='tab:blue')  # select the first sample.
        plt.xticks(rotation=90)
        plt.ylabel('Feature Contribution')
        plt.title('Local Feature Contributions (First-order)')
        plt.tight_layout()
        plt.savefig(f'./interpretations/{args.dataset}_honam{args.order}_emb{args.emb_size}_{args.seed}_sample0_local_first_order.pdf')
        plt.clf()
        ##########

        # Visualize the global first-order feature interactions.
        x, first_order_interactions = [], []
        for i in range(n_features):
            x.append(np.arange(x_train[:, i].min(), x_train[:, i].max(), 0.01))
        x = np.array(x).T
        
        for i in range(n_features):
            contributions = model.interpret(x, i).reshape(-1)
            density = np.histogram(x_train[:, i], bins=25, density=True)[0]
            density = density / density.max()
            plt.plot(x[:, i], contributions, color='tab:blue')
            plt.title(feature_names[i])
            for j, v in enumerate(np.arange(x_train[:, i].min(), x_train[:, i].max(), 0.04)):
                plt.axvspan(v, v + 0.04, color='lightcoral', alpha=density[j])
            plt.tight_layout()
            plt.savefig(f'./interpretations/{args.dataset}_honam{args.order}_emb{args.emb_size}_{args.seed}_global_first_order_{feature_names[i]}.pdf')
            plt.clf()
        ##########

        # Visualize the local second-order feature interactions for the first sample.
        second_order_interactions = []
        for i in range(n_features):
            second_order_interactions_i = []
            for j in range(n_features):
                if i == j:
                    second_order_interactions_i.append(np.zeros(shape=(len(x_train), 1)))
                else:
                    contributions = model.interpret(x_train, i, j)
                    second_order_interactions_i.append(contributions)
            second_order_interactions_i = np.concatenate(second_order_interactions_i, axis=1)
            second_order_interactions.append(second_order_interactions_i)
        second_order_interactions = np.stack(second_order_interactions, axis=1)

        plt.imshow(second_order_interactions[0], norm=CenteredNorm(), cmap='bwr')
        plt.xticks(np.arange(n_features), feature_names, rotation=90)
        plt.yticks(np.arange(n_features), feature_names)
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(f'./interpretations/{args.dataset}_honam{args.order}_emb{args.emb_size}_{args.seed}_sample0_local_second_order.pdf')
        plt.clf()
        ##########

        # Visualize the global second-order feature interactions.
        x, second_order_interactions = [], []
        for i in range(n_features):
            x.append(np.arange(x_train[:, i].min(), x_train[:, i].max(), 0.01))
        x = np.array(x).T
        
        for i in range(n_features):

            for j in range(i + 1, n_features):

                new_input = np.zeros(shape=(10000, n_features))
                a, b = np.meshgrid(x[:, i], x[:, j])
                new_input[:, i] = a.reshape(-1)
                new_input[:, j] = b.reshape(-1)

                contributions = model.interpret(new_input, i, j).reshape(-1)
                
                ax = plt.axes(projection='3d')
                ax.plot_trisurf(new_input[:, i], new_input[:, j], contributions, linewidth=0.2, cmap='bwr', norm=CenteredNorm(), antialiased=True)
                ax.set_xlabel(feature_names[i])
                ax.set_ylabel(feature_names[j])
                ax.set_zlabel('Feature Contribution')
                ax.set_title(f'{feature_names[i]} X {feature_names[j]}')
                plt.tight_layout()
                plt.savefig(f'./interpretations/{args.dataset}_honam{args.order}_emb{args.emb_size}_{args.seed}_global_second_order_{feature_names[i]}_{feature_names[j]}.pdf')
                plt.clf()
        ##########


if __name__ == '__main__':
    main()
