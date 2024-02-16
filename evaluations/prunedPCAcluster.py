"""
    This downstream task measures the clusterability performances of TNC4Maneuvering baselines for different datasets
"""

import os
import torch
import pickle
import argparse
import numpy as np
import seaborn as sns
from datetime import datetime
from tnc4maneuvering.models import WFPcaEncoder
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Other clustering methods that we tried
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main(args):

    start_time = datetime.now()
    
    data_path = './tnc4maneuvering/dataset/'
    if args.dataset == 'one_ds':
        data_path = os.path.join(data_path, 'one_ds/')
    elif args.dataset == 'one_dl':
        data_path = os.path.join(data_path, 'one_dl/')
    elif args.dataset == 'eight_d':
        data_path = os.path.join(data_path, 'eight_d/')
    else:
        print("Incorrect data type specified.")
        exit()


    encoder = WFPcaEncoder(encoding_size=64) # since it references the original inputs
    window_size = args.window_size 
    path = data_path

    # Load test data
    with open(os.path.join(data_path, 'x_test.pkl'), 'rb') as f:
        x_test = pickle.load(f)
    with open(os.path.join(data_path, 'state_test.pkl'), 'rb') as f:
        y_test = pickle.load(f)

    T = x_test.shape[-1]
    x_chopped_test = np.split(x_test[:, :, :window_size * (T // window_size)], (T // window_size), -1)
    y_chopped_test = np.concatenate(np.split(y_test[:, :window_size * (T // window_size)], (T // window_size), -1),
                                    0).astype(int)
    x_chopped_test = torch.Tensor(np.concatenate(x_chopped_test, 0))
    y_chopped_test = torch.Tensor(np.array([np.bincount(yy).argmax() for yy in y_chopped_test]))
    testset = torch.utils.data.TensorDataset(x_chopped_test, y_chopped_test)
    loader = torch.utils.data.DataLoader(testset, batch_size=100)

    n_test = len(x_test)
    inds = np.random.randint(0, x_test.shape[-1] - window_size, n_test * 200)
    windows = np.array([x_test[int(i % n_test), :, ind:ind + window_size] for i, ind in enumerate(inds)])
    windows = torch.Tensor(windows).to(device)  # Assuming 'device' is defined somewhere
    y_window = np.array([y_test[i % n_test, ind:ind + window_size] for i, ind in enumerate(inds)]).astype(int)
    windows_state = np.array([np.bincount(yy).argmax() for yy in y_window])


    print(f'DATASET: {args.dataset}')
    for i, path in enumerate(['one_ds']):
        s_score = []
        db_score = []
        intra_cluster_distances = [] 
        inter_cluster_distances = [] 
        for cv in range(1):
            checkpoint = torch.load('./ckpt/%s/checkpoint_%d.pth.tar'%(path, cv))
            encoder.load_state_dict(checkpoint['encoder_state_dict'])
            encoder = encoder.to(device)
            encoder.eval()
            encodings = []
            for windows, _ in loader:
                windows = windows.to(device)
                encoding = encoder(windows).detach().cpu().numpy()
                encodings.append(encoding)
            encodings = np.concatenate(encodings, 0)
            kmeans = KMeans(n_clusters=4, random_state=1).fit(encodings)
            cluster_labels = kmeans.labels_
            centroids = kmeans.cluster_centers_
            s_score.append(silhouette_score(encodings, cluster_labels))
            db_score.append(davies_bouldin_score(encodings, cluster_labels))

            # Total number of points in each cluster
            unique, counts = np.unique(cluster_labels, return_counts=True)
            total_points = dict(zip(unique, counts))

            # Intra-Cluster Distance, Inter-Cluster Distance, and Calinski-Harabasz Index
            calinski_harabasz_index = calinski_harabasz_score(encodings, cluster_labels)
            # Intra-Cluster Distance
            for i, centroid in enumerate(centroids):
                indices = np.where(cluster_labels == i)[0]
                cluster_points = encodings[indices]
                distances = np.linalg.norm(cluster_points - centroids[i], axis=1)
                intra_cluster_distances.append(np.mean(distances))
            # Inter-Cluster Distance
            for i in range(3):
                for j in range(i):
                    dist = np.linalg.norm(centroids[i] - centroids[j])
                    inter_cluster_distances.append(dist)

            dist_mat = cdist(centroids, centroids, metric='euclidean')
            plt.figure(figsize=(12, 8))
            sns.heatmap(dist_mat, annot=True, fmt=".2f", cmap="YlGnBu")
            plt.title("Distance Matrix among Centroids")
            plt.xlabel("Centroids")
            plt.ylabel("Centroids")
            plt.tight_layout()
            plt.savefig(os.path.join("./plots/%s" %path, "PCA-prunedKMeans_cent_dist.png"), dpi=100)
            plt.close()

            cluster_labels_ = []
            cluster_sizes = []
            sse = []
            for k in range(1, 3):
                kmeans = KMeans(n_clusters=k, random_state=1).fit(encodings)
                cluster_labels_.append(kmeans.labels_)
                cluster_sizes.append([len(kmeans.labels_[kmeans.labels_ == i]) for i in range(k)])
                sse.append(kmeans.inertia_)

            # Plot SSE vs K (other useful metrics)
            plt.figure(figsize=(12, 8))
            plt.plot(range(1, 3), sse, 'o:')
            plt.title('SSE vs K')
            plt.xlabel('Number of clusters (K)')
            plt.ylabel('Sum of Squared Errors (SSE)')
            plt.grid(which='both', axis='x')
            plt.tight_layout()
            plt.savefig(os.path.join("./plots/%s" %path, "PCA-prunedsse_KMeansClusters.png"), dpi=100)
            plt.close()

            min_idx = sse.index(min(sse))
            for i in range(len(cluster_sizes)):
                print(f'Cluster sizes for K={i+1}: {cluster_sizes[i]}, SSE={sse[i]}')
            print(f'Best SSE={sse[min_idx]} found for K={min_idx+1}')

            # Total number of points in each cluster
            cluster_sizes = [0] * 10
            for label in kmeans.labels_:
                cluster_sizes[label] += 1
            plt.bar(range(10), cluster_sizes)
            plt.title('Cluster Sizes')
            plt.xlabel('Cluster Number')
            plt.ylabel('Number of Points')
            plt.tight_layout()
            plt.savefig(os.path.join("./plots/%s" %path, "PCA-prunedCluster_points.png"), dpi=100)
            del encodings

        print('PCA-pruned Silhouette score: ', np.mean(s_score),'+-', np.std(s_score))
        print('PCA-pruned Davies Bouldin score: ', np.mean(db_score),'+-', np.std(db_score))
        print("PCA-pruned Intra-Cluster Distance:", np.mean(intra_cluster_distances),'+-', np.std(intra_cluster_distances))
        print("PCA-pruned Inter-Cluster Distance:", np.mean(inter_cluster_distances),'+-', np.std(inter_cluster_distances))
        print("PCA-pruned Calinski-Harabasz Index:", calinski_harabasz_index)


    end_time = datetime.now()
    time_taken = end_time - start_time
    print("Overall time taken: ", time_taken)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read CSV data from specified dataset.')
    parser.add_argument('--dataset', type=str, choices=['one_ds', 'one_dl', 'eight_d'], help='Specify the dataset to use')
    parser.add_argument('--window_size', type=int, default=250)
    args = parser.parse_args()

    main(args)