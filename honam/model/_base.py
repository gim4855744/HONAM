import torch

from tqdm import tqdm
from torch.nn import Module
from torch.utils.data import TensorDataset, DataLoader

__all__ = ['PyTorchModel']


class PyTorchModel(Module):

    def __init__(self, task, ckpt_path):

        super().__init__()

        self._task = task
        self._ckpt_path = ckpt_path

        self._criterion_map = {
            'regression': torch.nn.MSELoss,
            'classification': torch.nn.BCEWithLogitsLoss,
        }

    def fit(self, x_train, y_train, x_val, y_val, num_workers=0):
        
        device = next(self.parameters()).device
        patience = 100
        earlystop_count = 0

        x_train = torch.tensor(x_train, dtype=torch.float32)
        x_val = torch.tensor(x_val, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32)

        batch_size = x_train.size(dim=0) // 100
        train_dataset = TensorDataset(x_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        criterion = self._criterion_map[self._task]()

        tqdm_range = tqdm(range(1000))
        min_val_loss = 999.

        for _ in tqdm_range:

            total_train_loss, train_steps = 0, 0

            self.train()
            
            for batch_x, batch_y in train_loader:

                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                batch_y_hat = self(batch_x)
                loss = criterion(batch_y_hat, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()
                train_steps += 1

            self.eval()
            with torch.no_grad():
                x_val, y_val = x_val.to(device), y_val.to(device)
                y_hat = self(x_val)
                val_loss = criterion(y_hat, y_val)

            tqdm_range.set_postfix_str(f'Train Loss: {total_train_loss / train_steps}, Val Loss: {val_loss}, Min Val Loss: {min_val_loss}')

            if min_val_loss > val_loss:
                min_val_loss = val_loss
                checkpoint = {
                    'model': self,
                    'state_dict': self.state_dict(),
                }
                torch.save(checkpoint, self._ckpt_path)
                earlystop_count = 0
            else:
                earlystop_count += 1
                if earlystop_count >= patience:
                    return

    def predict(self, x):

        device = next(self.parameters()).device
        
        x = torch.tensor(x, dtype=torch.float32, device=device)

        self.eval()
        with torch.no_grad():
            y_hat = self(x)
            if self._task == 'classification':
                y_hat = torch.sigmoid(y_hat)

        return y_hat.detach().cpu().numpy()
