import os
import pickle
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller



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

    # Make sure there's one CSV file each specified directory
    csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
    if len(csv_files) != 1:
        print("Error: multiple or no CSV files in this specified directory.")
        exit()

    # Read the first CSV file found in the directory into a pandas DataFrame
    file_path = os.path.join(data_path, csv_files[0])
    df_original = pd.read_csv(file_path)

    df_original.reset_index(drop=True, inplace=True)
    df_temp = df_original[['longacceleration', 'latacceleration']].copy()
    feat1 = df_original[['longacceleration', 'latacceleration']].copy().T

    # Dataset splitting
    split_index = int(len(df_temp) * 0.8) # 80% train, 20% test
    train_data = df_temp[:split_index]
    test_data = df_temp[split_index:]

    x = df_temp['longacceleration'].values.astype(np.float32)
    y = df_temp['latacceleration'].values.astype(np.float32)

    x_windows = [x[i:i+args.window_size] for i in range(0, len(x), args.window_size)]
    y_windows = [y[i:i+args.window_size] for i in range(0, len(y), args.window_size)]

    window_labels = []
    window_categories = []
    # Perform the ADF test for each window and determine the category
    for i in range(len(x_windows)):
        adf_x = adfuller(x_windows[i])
        adf_y = adfuller(y_windows[i])
        
        if adf_x[1] > args.p_value:
            category_x = 3 #'Non-stationary data for x'
        else:
            category_x = 1 #'x likely to be stationary'
            
        if adf_y[1] > args.p_value:
            category_y = 4 #'Non-stationary data for y'
        else:
            category_y = 2 #'y likely to be stationary'
        
        # Determine the category based on ADF p-values for x and y
        if category_x == 3 and category_y == 4:
            category = 3 #'Non-stationary data for both x and y'
        elif category_x == 3:
            category = 1 #'x likely to be stationary, y non-stationary'
        elif category_y == 4:
            category = 2 #'y likely to be stationary, x non-stationary'
        else:
            category = 0 #'Similar stationarity'
        
        window_labels.extend([i] * len(x_windows[i]))
        window_categories.extend([category] * len(x_windows[i]))
    data = {'Windowlabel': window_labels, 'Stationaritycat': window_categories}
    df = pd.DataFrame(data)
    df_temp['Windowlabel'] = np.repeat(window_labels, args.window_size)[:len(df_temp)]


    # This prepares the dataset for ingestion into the TNC4maneuvering model:
    df2 = feat1.copy()
    all_signals = []
    all_states = []
    for _ in range(feat1.shape[1]):
        all_signals.append(feat1)
        all_states.append(df['Stationaritycat'])
        
    dataset = np.array(all_signals)
    states = np.array(all_states)
    n_train = int(len(dataset) * 0.8)
    train_data = dataset[:n_train]
    test_data = dataset[n_train:]

    def normalize(train_data, test_data, config='zero_to_one'):
        """ 
            This fucntion is to be called inorder to normalize the datasetsif not normalized
            It usses the mean and std of each feature from the training set
        """
        feature_size = feat1.shape[0]
        sig_len = feat1.shape[1] 
        d = [x.T for x in train_data]
        d = np.stack(d, axis=1)    
        if config == 'mean_normalized':
            feature_means = np.mean(train_data, axis=(1))
            feature_std = np.std(train_data, axis=(1))
            np.seterr(divide='ignore', invalid='ignore')
            train_data_n = train_data - feature_means[np.newaxis,:,np.newaxis]/np.where(feature_std == 0, 1, feature_std)[np.newaxis,:,np.newaxis]
            test_data_n = test_data - feature_means[np.newaxis,:,np.newaxis]/np.where(feature_std == 0, 1, feature_std)[np.newaxis, :, np.newaxis]
        elif config == 'zero_to_one':
            feature_max = np.tile(np.max(d.reshape(-1, feature_size), axis=0), (sig_len, 1)).T
            feature_min = np.tile(np.min(d.reshape(-1, feature_size), axis=0), (sig_len, 1)).T
            train_data_n = np.array([(x) / (feature_max) for x in train_data])
            test_data_n = np.array([(x) / (feature_max) for x in test_data])
        return train_data_n, test_data_n
            
    # Now normalizing the data
    train_data_n, test_data_n = normalize(train_data, test_data) #, config=None)
    train_state = states[:n_train]
    test_state = states[n_train:]
    print("Datase Shapes || \tTrainset: ", train_data_n.shape,' || ', "\tTestset: ", test_data_n.shape)
    print("States Shapes || \tTrainset: ", train_state.shape,' || ', "\tTestset: ", test_state.shape)

    # For visualization of the datasets
    f, axes = plt.subplots(2,1)
    f.set_figheight(5)
    f.set_figwidth(15)
    labels = ['$a_{lat}$', '$a_{lon}$' ]
    colors = ['r', 'b']
    for i, ax in enumerate(axes):
        ax.plot(train_data[0, i, :], c=colors[i], linestyle='-', label=labels[i])
        ax.axhline(y=0, color = 'grey', linestyle = ':')
        ax.set_ylabel(df_original.columns[i] + '\n - Normalized - ', fontsize=10) 
        ax.set_ylabel(labels[i]+ '\n - Normalized - ', fontsize=15) 
        plt.xlabel(' - Time (index) - ') 
        # for t in range(train_data[0,i,:].shape[-1]):
        #     ax.axvspan(t, min(t+1, train_state.shape[-1]-1), facecolor=colors[train_state[0,t]])
        plt.suptitle(' - Vehicle Accelerations - ' ,fontsize=15) # w/ 4 states from created labels  
    f.tight_layout()
    plt.savefig(data_path + f"/{args.dataset}.eps", format='eps', dpi=100)
    plt.savefig(data_path + f"/{args.dataset}.png", dpi=100)

    # Save signals to file
    if not os.path.exists(data_path + 'one_ds/'):
        os.mkdir(data_path)
    with open(data_path+'/x_train.pkl', 'wb') as f:
        pickle.dump(train_data_n, f)
    with open(data_path+'/x_test.pkl', 'wb') as f:
        pickle.dump(test_data_n, f)
    with open(data_path+'/state_train.pkl', 'wb') as f:
        pickle.dump(train_state, f)
    with open(data_path+'/state_test.pkl', 'wb') as f:
        pickle.dump(test_state, f)

    end_time = datetime.now()
    time_taken = end_time - start_time
    print("Overall time taken: ", time_taken)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read CSV data from specified dataset.')
    parser.add_argument('--dataset', type=str, choices=['one_ds', 'one_dl', 'eight_d'], help='Specify the dataset to use')
    parser.add_argument('--window_size', type=int, default=50)
    parser.add_argument('--p_value', type=float, default=0.01)
    args = parser.parse_args()

    if not args.dataset:
        print("Please specify the dataset to use using the --dataset argument.")
        exit()

    main(args)