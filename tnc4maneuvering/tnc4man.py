"""
    Temporal Neighborhood Coding for Maneuvering (TNC4Maneuvering) an unsupervised learning representation exctreaction
    method from non-stationary accelerations time series
"""

import os
import sys
import math
import pickle
import random
import argparse
import numpy as np
import arch.unitroot as ur
from datetime import datetime
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

import torch
from torch.utils import data

from models import WFEncoder
from tnc4m_utils import plot_distribution, track_encoding
from tnc4m_evals import WFClassificationExperiment

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
class Discriminator(torch.nn.Module):
    def __init__(self, input_size, device):
        super(Discriminator, self).__init__()
        self.device = device
        self.input_size = input_size
        self.model = torch.nn.Sequential(torch.nn.Linear(2*self.input_size, 4*self.input_size),
                                         torch.nn.ReLU(inplace=True),
                                         torch.nn.Dropout(0.5),
                                         torch.nn.Linear(4*self.input_size, 1))
        torch.nn.init.xavier_uniform_(self.model[0].weight)
        torch.nn.init.xavier_uniform_(self.model[3].weight)

    def forward(self, x, x_tild):
        """
            Gives the probability of two inputs belonging in the same neighbourhood or not.
        """
        x_all = torch.cat([x, x_tild], -1)
        p = self.model(x_all)
        return p.view((-1,))


start_time = datetime.now()
class TNC4ManDataset(data.Dataset):
    def __init__(self, x, mc_sample_size, window_size, augmentation, epsilon=3, state=None, adf=False):
        super(TNC4ManDataset, self).__init__()
        self.time_series = x
        self.T = x.shape[-1]
        self.window_size = window_size
        self.sliding_gap = int(window_size*25.2)
        self.window_per_sample = (self.T-2*self.window_size)//self.sliding_gap
        self.mc_sample_size = mc_sample_size
        self.state = state
        self.augmentation = augmentation
        self.adf = adf
        if not self.adf:
            self.epsilon = epsilon #changing epsilon steps
            self.delta = 5*window_size*epsilon

    def __len__(self):
        return len(self.time_series)*self.augmentation

    def __getitem__(self, ind):
        ind = ind%len(self.time_series)
        t = np.random.randint(2*self.window_size, self.T-2*self.window_size)
        x_t = self.time_series[ind][:,t-self.window_size//2:t+self.window_size//2]
        X_close = self._find_neighours(self.time_series[ind], t)
        X_distant = self._find_non_neighours(self.time_series[ind], t)

        if self.state is None:
            y_t = -1
        else:
            y_t = torch.round(torch.mean(self.state[ind][t-self.window_size//2:t+self.window_size//2]))
        return x_t, X_close, X_distant, y_t

    def _find_neighours(self, x, t):
        T = self.time_series.shape[-1]
        if self.adf:
            gap = self.window_size
            corr = []
            for w_t in range(self.window_size, 4*self.window_size, gap): # change step size from 1 to 10
                try:
                    p_val = 0
                    for f in range(x.shape[-2]):
                        # This is the ADF testing used
                        p = ur.ADF(np.array(x[f, max(0,t - w_t):min(x.shape[-1], t + w_t)].reshape(-1, ))).pvalue
                        p_val += 0.01 if math.isnan(p) else p #0.01 can be a bit slower and can make the code more slower (adjust accordingly)
                    corr.append(p_val/x.shape[-2])
                except:
                    corr.append(0.6)
            self.epsilon = len(corr) if len(np.where(np.array(corr) >= 0.01)[0])==0 else (np.where(np.array(corr) >= 0.01)[0][0] + 1) #+1
            self.delta = 5*self.epsilon*self.window_size

        ## Drawn using randn() from a Gaussian
        t_p = [int(t+np.random.randn()*self.epsilon*self.window_size) for _ in range(self.mc_sample_size)]
        t_p = [max(self.window_size//2+1,min(t_pp,T-self.window_size//2)) for t_pp in t_p]
        x_p = torch.stack([x[:, t_ind-self.window_size//2:t_ind+self.window_size//2] for t_ind in t_p])
        return x_p

    def _find_non_neighours(self, x, t):
        T = self.time_series.shape[-1]
        if t>T/2:
            t_n = np.random.randint(self.window_size//2, max((t - self.delta + 1), self.window_size//2+1), self.mc_sample_size)
        else:
            t_n = np.random.randint(min((t + self.delta), (T - self.window_size-1)), (T - self.window_size//2), self.mc_sample_size)
        x_n = torch.stack([x[:, t_ind-self.window_size//2:t_ind+self.window_size//2] for t_ind in t_n])

        if len(x_n)==0:
            rand_t = np.random.randint(0,self.window_size//5)
            if t > T / 2:
                x_n = x[:,rand_t:rand_t+self.window_size].unsqueeze(0)
            else:
                x_n = x[:, T - rand_t - self.window_size:T - rand_t].unsqueeze(0)
        return x_n


def epoch_run(loader, disc_model, encoder, device, w=0, optimizer=None, train=True):
    if train:
        encoder.train()
        disc_model.train()
    else:
        encoder.eval()
        disc_model.eval()
    loss_fn = torch.nn.BCEWithLogitsLoss()
    encoder.to(device)
    disc_model.to(device)
    epoch_loss = 0
    epoch_acc = 0
    batch_count = 0
    for x_t, x_p, x_n, _ in loader:
        mc_sample = x_p.shape[1]
        batch_size, f_size, len_size = x_t.shape
        x_p = x_p.reshape((-1, f_size, len_size))
        x_n = x_n.reshape((-1, f_size, len_size))
        x_t = np.repeat(x_t, mc_sample, axis=0)
        neighbors = torch.ones((len(x_p))).to(device)
        non_neighbors = torch.zeros((len(x_n))).to(device)
        x_t, x_p, x_n = x_t.to(device), x_p.to(device), x_n.to(device)
        z_t = encoder(x_t)
        z_p = encoder(x_p)
        z_n = encoder(x_n)

        d_p = disc_model(z_t, z_p)
        d_n = disc_model(z_t, z_n)

        p_loss = loss_fn(d_p, neighbors)
        n_loss = loss_fn(d_n, non_neighbors)
        n_loss_u = loss_fn(d_n, neighbors)
        loss = (p_loss + w*n_loss_u + (1-w)*n_loss)/2

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        p_acc = torch.sum(torch.nn.Sigmoid()(d_p) > 0.5).item() / len(z_p)
        n_acc = torch.sum(torch.nn.Sigmoid()(d_n) < 0.5).item() / len(z_n)
        epoch_acc = epoch_acc + (p_acc+n_acc)/2
        epoch_loss += loss.item()
        batch_count += 1
    return epoch_loss/batch_count, epoch_acc/batch_count



def learn_encoder(x, encoder, window_size, w, lr=0.001, decay=0.005, mc_sample_size=20,
                  n_epochs=100, path='one_ds', device='cpu', augmentation=1, n_cross_val=1, cont=False):
    accuracies, losses = [], []
    for cv in range(n_cross_val):
        if 'one_ds' in path:
            encoder = WFEncoder(encoding_size=64).to(device)
            batch_size = 5

        elif 'one_dl' in path:
            encoder = WFEncoder(encoding_size=64).to(device)
            batch_size = 5

        elif 'eight_d' in path:
            encoder = WFEncoder(encoding_size=64).to(device)
            batch_size = 5

        if not os.path.exists('./ckpt/%s'%path):
            os.mkdir('./ckpt/%s'%path)
        if cont:
            checkpoint = torch.load('./ckpt/%s/checkpoint_%d.pth.tar'%(path, cv))
            encoder.load_state_dict(checkpoint['encoder_state_dict'])

            disc_model = Discriminator(encoder.encoding_size, device)
            params = list(disc_model.parameters()) + list(encoder.parameters())
            optimizer = torch.optim.Adam(params, lr=lr, weight_decay=decay)
            inds = list(range(len(x)))
            random.shuffle(inds)
            x = x[inds]
            n_train = int(0.8 * len(x))
            performance = []
            best_acc = 0
            best_loss = np.inf

            for epoch in range(n_epochs + 1):
                trainset = TNC4ManDataset(x=torch.Tensor(x[:n_train]), mc_sample_size=mc_sample_size,
                                          window_size=window_size, augmentation=augmentation, adf=True)
                train_loader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4,
                                               pin_memory=False)
                validset = TNC4ManDataset(x=torch.Tensor(x[n_train:]), mc_sample_size=mc_sample_size,
                                          window_size=window_size, augmentation=augmentation, adf=True)
                valid_loader = data.DataLoader(validset, batch_size=batch_size, shuffle=True)

                epoch_loss, epoch_acc = epoch_run(train_loader, disc_model, encoder, optimizer=optimizer,
                                                  w=w, train=True, device=device)
                test_loss, test_acc = epoch_run(valid_loader, disc_model, encoder, train=False, w=w, device=device)
                performance.append((epoch_loss, test_loss, epoch_acc, test_acc))
                if epoch % 10 == 0:
                    print(
                        '(cv:%s)Epoch %d Loss =====> Training Loss: %.5f \t Training Accuracy: %.5f \t Test Loss: %.5f \t Test Accuracy: %.5f'
                        % (cv, epoch, epoch_loss, epoch_acc, test_loss, test_acc))
                if best_loss > test_loss or path == 'one_ds':
                    best_acc = test_acc
                    best_loss = test_loss
                    state = {
                        'epoch': epoch,
                        'encoder_state_dict': encoder.state_dict(),
                        'discriminator_state_dict': disc_model.state_dict(),
                        'best_accuracy': test_acc
                    }
                    torch.save(state, './ckpt/%s/checkpoint_%d.pth.tar'%(path,cv))
            accuracies.append(best_acc)
            losses.append(best_loss)

            # Save performance plots
            if not os.path.exists('./plots/%s'%path):
                os.mkdir('./plots/%s'%path)
            train_loss = [t[0] for t in performance]
            test_loss = [t[1] for t in performance]
            train_acc = [t[2] for t in performance]
            test_acc = [t[3] for t in performance]

            plt.figure()
            plt.plot(np.arange(n_epochs+1), train_loss, label="Train")
            plt.plot(np.arange(n_epochs+1), test_loss, label="Test")
            plt.title("Loss")
            plt.legend()
            plt.savefig(os.path.join("./plots/%s"%path, "loss_%d.png"%cv), dpi=100)
            plt.figure()
            plt.plot(np.arange(n_epochs+1), train_acc, label="Train")
            plt.plot(np.arange(n_epochs+1), test_acc, label="Test")
            plt.title("Accuracy")
            plt.legend()
            plt.savefig(os.path.join("./plots/%s"%path, "accuracy_%d.png"%cv), dpi=100)

        print('=======> Performance Summary:')
        print('Accuracy: %.2f +- %.2f'%(100*np.mean(accuracies), 100*np.std(accuracies)))
        print('Loss: %.4f +- %.4f'%(np.mean(losses), np.std(losses)))
        return encoder




def main(is_train, data_type, cv, w, cont):
    if not os.path.exists("./plots"):
        os.mkdir("./plots")
    if not os.path.exists("./ckpt/"):
        os.mkdir("./ckpt/")
    
    if data_type in ['one_ds', 'one_dl', 'eight_d']: 
        window_size = 250
        encoder = WFEncoder(encoding_size=64).to(device)
        
        path = './tnc4maneuvering/dataset/one_ds'
        if is_train:
            with open(os.path.join(path, 'x_train.pkl'), 'rb') as f:
                x = pickle.load(f)
            T = x.shape[-1]

            x_window = np.concatenate(np.split(x[:, :, :T // 5 * 5], 5, -1), 0)
            learn_encoder(torch.Tensor(x_window), encoder, w=w, lr=1e-3, decay=1e-4, n_epochs=1, window_size=window_size, path='one_ds', mc_sample_size=12, device=device, augmentation=7, n_cross_val=cv, cont=cont)
        else:
            with open(os.path.join(path, 'x_test.pkl'), 'rb') as f:
                x_test = pickle.load(f)
            with open(os.path.join(path, 'state_test.pkl'), 'rb') as f:
                y_test = pickle.load(f)
            checkpoint = torch.load('./ckpt/%s/checkpoint_0.pth.tar' % (data_type))
            
            encoder.load_state_dict(checkpoint['encoder_state_dict'])
            encoder = encoder.to(device)
            if data_type == 'one_ds':
                track_encoding(x_test[0, :, :], y_test[0, :], encoder, window_size, 'one_ds', int((1957 - 250) / (1957 / 250 - 1)))
            elif data_type == 'one_dl':
                track_encoding(x_test[0, :, :], y_test[0, :], encoder, window_size, 'one_dl', int((16568 - 250) / (16568 / 250 - 1)))
            elif data_type == 'eight_d':
                track_encoding(x_test[0, :, :], y_test[0, :], encoder, window_size, 'eight_d', int((19273 - 250) / (19273 / 250 - 1)))
            else:
                print("Incorrect data_type specified.")

            for cv_ind in range(cv):
                plot_distribution(x_test, y_test, encoder, window_size=window_size, path='one_ds',
                                device=device, augment=1, cv=cv_ind, title='TNC4maneuvering')
            exp = WFClassificationExperiment(window_size=window_size, cv=cv_ind)
            exp.run(data='one_ds', n_epochs=3, lr_e2e=0.0001, lr_cls=0.01)  

if __name__ == '__main__':
    random.seed(1234)
    parser = argparse.ArgumentParser(description='Run TNC')
    parser.add_argument('--data', type=str, choices=['one_ds', 'one_dl', 'eight_d'], help='Specify the dataset to use')
    parser.add_argument('--cv', type=int, default=1)
    parser.add_argument('--w', type=float, default=0.05)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--cont', action='store_true')
    args = parser.parse_args()
    print('TNC4man model with w=%f' % args.w)
    main(args.train, args.data, args.cv, args.w, args.cont)  
    

end_time = datetime.now()
time_taken = end_time - start_time
print("Overall time taken: ", time_taken)
