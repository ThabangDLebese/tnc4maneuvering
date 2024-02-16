"""
    This part of the code is where almost all the models are called from, including downstreams task functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as weight_norm
from sklearn.decomposition import PCA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# A CNN encoder with causal dilated convolutions: Main encoder 
class WFEncoder(nn.Module):
    def __init__(self, encoding_size=64, classify=False, n_classes=None):
        super(WFEncoder, self).__init__()
        self.conv1 = nn.Conv1d(2, 32, kernel_size=3, padding=2, dilation=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=4, dilation=2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=8, dilation=4)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.encoding_size = encoding_size
        self.linear = nn.Linear(128, encoding_size).to(device)
        self.classify = classify
        self.n_classes = n_classes
        self.classifier = None
        if self.classify:
            if self.n_classes is None:
                raise ValueError('Specify number of output classes for the encoder')
            else:
                self.classifier = nn.Sequential(nn.Dropout(0.5),
                    nn.Linear(self.encoding_size, self.n_classes))
                nn.init.xavier_uniform_(self.classifier[1].weight)

    def forward(self, x):
        device = next(self.parameters()).device
        x = x.to(device)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x).view(x.size(0), -1)
        x = F.normalize(x)
        encoding = self.linear(x)
        if self.classify:
            c = self.classifier(encoding)
            return c
        else:
            return encoding



# Pruning with PCA: uses the the main encoder WFEncoder with extensions of dimensionality reduction with Principal Component Analysis (PCA)
class WFPcaEncoder(WFEncoder):
    def __init__(self, encoding_size=64, classify=False, n_classes=None, num_pca_components=3):
        super(WFPcaEncoder, self).__init__(encoding_size=encoding_size, classify=classify, n_classes=n_classes)
        self.num_pca_components = num_pca_components
        self.pca = PCA(n_components=num_pca_components)

    def forward(self, x):
        device = next(self.parameters()).device
        x = x.to(device)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x).view(x.size(0), -1)
        x = F.normalize(x)
        encoding = self.linear(x)

        # Perform PCA on the encoding to reduce dimensionality
        encoded_data = encoding.cpu().detach().numpy()
        pca_result = self.pca.fit_transform(encoded_data)
        pca_result_tensor = torch.tensor(pca_result).to(device)

        if self.classify:
            c = self.classifier(pca_result_tensor)
            return c
        else:
            return pca_result_tensor




# Pruning with PCC:  uses the the main encoder WFEncoder with extensions of dimensionality reduction using Pearson Correlation Coeffitient (PCC)
class WFPCCEncoder(WFEncoder):
    def __init__(self, encoding_size=64, classify=False, n_classes=None):
        super(WFPCCEncoder, self).__init__(encoding_size, classify, n_classes)

    def forward(self, x):
        device = next(self.parameters()).device
        x = x.to(device)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x).view(x.size(0), -1)
        x = F.normalize(x)
        # Perform PCC based feature selection
        x = self.pcc_feature_selection(x)
        encoding = self.linear(x)
        if self.classify:
            c = self.classifier(encoding)
            return c
        else:
            return encoding

    def pcc_feature_selection(self, x):
        x_t = x.t()
        correlation_matrix = torch.mm(x_t, x) / x.size(0)
        pv_limit= 0.7 # (You can change this threshold)
        correlated_indices = torch.any(torch.abs(correlation_matrix) > pv_limit, dim=1)
        selected_features = x[:, ~correlated_indices]
        return selected_features




# Classifier: For the linear classifcation task
class StateClassifier(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(StateClassifier, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.normalize = torch.nn.BatchNorm1d(self.input_size)
        self.nn = torch.nn.Linear(self.input_size, self.output_size)
        torch.nn.init.xavier_uniform_(self.nn.weight)

    def forward(self, x):
        x = self.normalize(x)
        logits = self.nn(x)
        return logits



class E2EStateClassifier(torch.nn.Module):
    def __init__(self, hidden_size, in_channel, encoding_size, output_size, cell_type='GRU', num_layers=5, dropout=0,
                 bidirectional=True, device='cpu'):
        super(E2EStateClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.in_channel = in_channel
        self.num_layers = num_layers
        self.cell_type = cell_type
        self.encoding_size = encoding_size
        self.bidirectional = bidirectional
        self.output_size = output_size
        self.device = device

        self.fc = torch.nn.Sequential(torch.nn.Linear(self.hidden_size*(int(self.bidirectional) + 1), self.encoding_size)).to(self.device)
        self.nn = torch.nn.Sequential(torch.nn.Linear(self.encoding_size, self.output_size)).to(self.device)
        if cell_type=='GRU':
            self.rnn = torch.nn.GRU(input_size=self.in_channel, hidden_size=self.hidden_size, num_layers=num_layers,
                                    batch_first=False, dropout=dropout, bidirectional=bidirectional).to(self.device)
        elif cell_type=='LSTM':
            self.rnn = torch.nn.LSTM(input_size=self.in_channel, hidden_size=self.hidden_size, num_layers=num_layers,
                                    batch_first=False, dropout=dropout, bidirectional=bidirectional).to(self.device)
        else:
            raise ValueError('Cell type not defined, must be one of the following {GRU, LSTM, RNN}')

    def forward(self, x):
        x = x.permute(2,0,1)
        if self.cell_type=='GRU':
            past = torch.zeros(self.num_layers * (int(self.bidirectional) + 1), x.shape[1], self.hidden_size).to(self.device)
        elif self.cell_type=='LSTM':
            h_0 = torch.zeros(self.num_layers * (int(self.bidirectional) + 1), (x.shape[1]), self.hidden_size).to(self.device)
            c_0 = torch.zeros(self.num_layers * (int(self.bidirectional) + 1), (x.shape[1]), self.hidden_size).to(self.device)
            past = (h_0, c_0)
        out, _ = self.rnn(x, past)  # out shape = [seq_len, batch_size, num_directions*hidden_size]
        encodings = self.fc(out[-1].squeeze(0))
        return self.nn(encodings)




class MimicEncoder(torch.nn.Module):
    def __init__(self, input_size, in_channel, encoding_size):
        super(MimicEncoder, self).__init__()
        self.input_size = input_size
        self.in_channel = in_channel
        self.encoding_size = encoding_size

        self.nn = torch.nn.Sequential(torch.nn.Linear(input_size, 64),
                                      torch.nn.Dropout(),
                                      torch.nn.ReLU(),
                                      torch.nn.Linear(64, encoding_size))

    def forward(self, x):
        x = torch.mean(x, dim=1)
        encodings = self.nn(x)
        return encodings
    


class WFClassifier(torch.nn.Module):
    def __init__(self, encoding_size, output_size):
        super(WFClassifier, self).__init__()
        self.encoding_size = encoding_size
        self.output_size = output_size
        self.classifier = nn.Linear(self.encoding_size, output_size).to(device)
        torch.nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, x):
        c = self.classifier(x.to(device))
        return c