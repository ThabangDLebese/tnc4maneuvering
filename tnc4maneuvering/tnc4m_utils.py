import os
import pickle
import numpy as np
import pandas as pd
import pyarrow as pa

import seaborn as sns
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from matplotlib import gridspec

import torch
from torch.utils import data
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def track_encoding(sample, label, encoder, window_size, path, sliding_gap=1):
    T = sample.shape[-1]
    windows_label = []
    encodings = []
    device = 'cuda'
    encoder.to(device)
    encoder.eval()
    for t in range(window_size//2,T-window_size//2,sliding_gap):
        windows = sample[:, t-(window_size//2):t+(window_size//2)]
        windows_label.append((np.bincount(label[t-(window_size//2):t+(window_size//2)].astype(int)).argmax()))
        encodings.append(encoder(torch.Tensor(windows).unsqueeze(0).to(device)).view(-1,))
    for t in range(window_size//(2*sliding_gap)):
        encodings.append(encodings[-1])
        encodings.insert(0, encodings[0])
    encodings = torch.stack(encodings, 0)
    # encodings_df = pd.DataFrame(encodings)

    if path in ['one_ds', 'one_dl', 'eight_d']:
        f, axs = plt.subplots(3)
        f.set_figheight(12)
        f.set_figwidth(27)
        gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 7])
        axs[0] = plt.subplot(gs[0])
        axs[1] = plt.subplot(gs[1])
        axs[2] = plt.subplot(gs[2])
        sns.lineplot(np.arange(0, sample.shape[1] / 250, 1. / 250), sample[0], ax=axs[0], color='red', label ='$a_{lat}$')
        sns.lineplot(np.arange(0, sample.shape[1] / 250, 1. / 250), sample[1], ax=axs[1], color='blue', label = '$a_{lon}$')
        axs[1].margins(x=0)
        axs[1].grid(False)
        axs[1].xaxis.set_tick_params(labelsize=22)
        axs[1].yaxis.set_tick_params(labelsize=22)
    else:
        f, axs = plt.subplots(2)
        f.set_figheight(10)
        f.set_figwidth(27)
        for feat in range(min(sample.shape[0], 5)):
            sns.lineplot(np.arange(sample.shape[1]), sample[feat], ax=axs[0], )
    
    axs[0].set_title('Acceleration Signals', fontsize=30, fontweight='bold')
    axs[0].xaxis.set_tick_params(labelsize=22)
    axs[0].yaxis.set_tick_params(labelsize=22)
    axs[-1].xaxis.set_tick_params(labelsize=22)
    axs[-1].yaxis.set_tick_params(labelsize=22)
    axs[-1].set_ylabel('Encoding dimensions', fontsize=28)
    axs[0].margins(x=0)
    axs[0].grid(False)
    t_0 = 0
    color10 = ["red","green","blue","yellow", "orange","purple","cyan","magenta","black", 'm']

    if path not in ['one_ds', 'one_dl', 'eight_d']:
        for t in range(1, label.shape[-1]):
            if label[t] == label[t-1]:
                continue
            else:
                axs[0].axvspan(t_0, min(t+1, label.shape[-1]-1), facecolor=color10[int(label[t_0])], alpha=0.5)
                t_0 = t
        axs[0].axvspan(t_0, label.shape[-1]-1, facecolor=color10[int(label[t_0])], alpha=0.5)
    axs[-1].set_title('Representations', fontsize=30, fontweight='bold')
    sns.heatmap(encodings.detach().cpu().numpy().T, cbar=True, linewidth=0.5, ax=axs[-1], linewidths=0.05, xticklabels=False, cbar_kws={'orientation': 'horizontal'})
    f.tight_layout()
    plt.savefig(os.path.join("./plots/%s" % path, "reps_overtime.png"), dpi=100)
    plt.savefig(os.path.join("./plots/%s" % path, "reps_overtime.eps"), format='eps', dpi=100)

    reps_overtime = pd.DataFrame(encodings.detach().cpu().numpy())
    # Convert column names to strings
    reps_overtime.columns = reps_overtime.columns.astype(str)  
    print(' --- Saving representations overtime --- ')
    print('Reps_overtime: ', reps_overtime.shape)
    # Saving Dataset as a compressed parquet file
    reps_overtime.to_parquet(os.path.join("./plots/%s"%path, "reps_overtime.parquet.gz"))

    pca = PCA(n_components=2)
    embedding = pca.fit_transform(encodings.detach().cpu().numpy())
    d = {'f1':embedding[:,0], 'f2':embedding[:,1], 'time':np.arange(len(embedding))}
    df = pd.DataFrame(data=d)
    fig, ax = plt.subplots()
    ax.set_title("Encoding")
    sns.scatterplot(x="f1", y="f2", data=df, hue="time")
    plt.savefig(os.path.join("./plots/%s" % path, "embedding_trajectory.png"),dpi=100)
    plt.savefig(os.path.join("./plots/%s" % path, "embedding_trajectory.eps"), format='eps', dpi=100)



# augment=1,5 
def plot_distribution(x_test, y_test, encoder, window_size, path, device, title="", augment=0.009079, cv=0):
    checkpoint = torch.load('./ckpt/%s/checkpoint_%d.pth.tar'%(path, cv))
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    encoder = encoder.to(device)
    n_test = len(x_test)
    inds = np.random.randint(0, x_test.shape[-1] - window_size, n_test * augment)
    windows = np.array([x_test[int(i % n_test), :, ind:ind + window_size] for i, ind in enumerate(inds)])
    windows_state = [np.round(np.mean(y_test[i % n_test, ind:ind + window_size], axis=-1)) for i, ind in
                     enumerate(inds)]
    encodings = encoder(torch.Tensor(windows).to(device))
    print('Encodings-shape:',encodings.shape)

    tsne = TSNE(n_components=2)
    embedding = tsne.fit_transform(encodings.detach().cpu().numpy())
    original_embedding = TSNE(n_components=2).fit_transform(windows.reshape((len(windows), -1)))
    df_original = pd.DataFrame({"f1": original_embedding[:, 0], "f2": original_embedding[:, 1], "state": windows_state})
    print('df_original:', df_original.shape)
    tsne_encoded = pd.DataFrame({"f1": embedding[:, 0], "f2": embedding[:, 1], "state": windows_state})
    
    print(' --- Saving t-SNE components --- ')
    print('tsne_encoded:', tsne_encoded.shape)
    # Save the DataFrame as a compressed Parquet file
    tsne_encoded.to_parquet(os.path.join("./plots/%s"%path, "encoding_%d.gz"%cv))
    plt.savefig(os.path.join("./plots/%s"%path, "tsne_encoded_%d.png"%cv), dpi=100)
    plt.savefig(os.path.join("./plots/%s"%path, "tsne_encoded_%d.eps"%cv), format='eps', dpi=100)
    
    # Save plots
    if not os.path.exists(os.path.join("./plots/%s"%path)):
        os.mkdir(os.path.join("./plots/%s"%path))
    fig, ax = plt.subplots()
    ax.set_title("Origianl signals TSNE", fontweight="bold")
    sns.scatterplot(x="f1", y="f2", data=df_original, hue="state")
    plt.savefig(os.path.join("./plots/%s"%path, "tsne_signals.png"), dpi=100)
    plt.savefig(os.path.join("./plots/%s"%path, "tsne_signals.eps"), format='eps', dpi=100)
    fig, ax = plt.subplots()
    ax.set_title("%s"%title, fontweight="bold", fontsize=18)

    if 'one_ds' in path: #, 'one_dl', 'eight_d'] 
        sns.scatterplot(x="f1", y="f2", data=df_original, hue = "state")
    else:
        sns.scatterplot(x="f1", y="f2", data=tsne_encoded, hue="state")
    plt.savefig(os.path.join("./plots/%s"%path, "tsne_encoded_%d.png"%cv), dpi=100)
    plt.savefig(os.path.join("./plots/%s"%path, "tsne_encoded_%d.eps"%cv), format='eps', dpi=100)