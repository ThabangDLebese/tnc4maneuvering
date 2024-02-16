import os
import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import pyarrow.parquet as pq
from datetime import datetime
import matplotlib.pyplot as plt

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"


def main(args):


    start_time = datetime.now()
    # Check if data argument is provided
    if not args.data:
        print("Please specify the dataset to use using the --data argument.")
        exit()

    data_path = './tnc4maneuvering/dataset/'
    # Choose the dataset based on the argument
    if args.data == 'one_ds':
        data_path = os.path.join(data_path, 'one_ds/')
    elif args.data == 'one_dl':
        data_path = os.path.join(data_path, 'one_dl/')
    elif args.data == 'eight_d':
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

    # Extraction of manual reps 
    def extract_extreme_diffs_squared_sum(time_series, window_size):
        num_windows = len(time_series) // window_size
        remainder = len(time_series) % window_size
        if remainder > 0:
            padded_length = num_windows * window_size + window_size
            pad_width = padded_length - len(time_series)
            time_series = np.pad(time_series, (0, pad_width), 'constant')  # Pad with zeros since window is not divisor of data-length
        
        num_windows = len(time_series) // window_size 
        extreme_diffs_squared_sum = np.zeros(num_windows)
        
        for i in range(num_windows):
            window_start = i * window_size
            window_end = window_start + window_size
            window_values = time_series[window_start:window_end]
            extreme_values = np.array([np.min(window_values), np.max(window_values)])
            extreme_diffs = np.diff(extreme_values)
            extreme_diffs_squared_sum[i] = np.sum(extreme_diffs ** 2)
        
        return extreme_diffs_squared_sum
    optimal_window_size = args.window_size

    A_tot = 0
    for column in ['latacceleration', 'longacceleration']:
        time_series = df_temp[column]
        window_size = optimal_window_size
        extreme_diffs_squared_sum = extract_extreme_diffs_squared_sum(time_series, window_size)
        A = extreme_diffs_squared_sum
        A_tot += A ** 2
    A_tot = np.sqrt(A_tot)


    # print(' ------ Reading pre-prunnining results from a saved encodings file ------ ')
    # one_ds
    file_path = f"./plots/{args.data}/reps_overtime.parquet.gz"
    # one_dl
    # file_path = f"./plots/{args.data}/reps_overtime.parquet.gz"
    # eight_d
    # file_path = f"./plots/{args.data}/reps_overtime.parquet.gz"

    # Read the zipped Parquet file into a PyArrow table
    table = pq.read_table(file_path)
    reps_overtime = table.to_pandas()


    def multivariate_linear_regression(x_train, y_train, x_test, y_test, lr=0.01, n_epochs=200000, patience=2000):
        x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
        
        num_features = x_train.shape[0]
        num_outputs = y_train.shape[1]
        
        model = nn.Linear(num_features, num_outputs)
        nn.init.xavier_normal_(model.weight)
        loss_fn = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=lr)
        
        # Train the model
        train_losses = []
        test_losses = []
        best_test_loss = float('inf')
        patience_counter = 0
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            y_train_pred = model(x_train_tensor.transpose(0, 1))
            y_test_pred = model(x_test_tensor.transpose(0, 1))
            train_loss = loss_fn(y_train_pred, y_train_tensor)
            test_loss = loss_fn(y_test_pred, y_test_tensor)
            
            train_loss.backward()
            optimizer.step()
            train_losses.append(train_loss.item())
            test_losses.append(test_loss.item())
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Train Loss = {train_loss.item():.4f}, Val Loss = {test_loss.item():.4f}")
            
            # Early stopping
            if test_loss.item() < best_test_loss:
                best_test_loss = test_loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Stopping early after {epoch} epochs")
                    break
        
        # Plot the loss function
        fig, ax = plt.subplots()
        ax.plot(train_losses, label="Train Loss")
        ax.plot(test_losses, label="Test Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        plt.savefig(data_path + "/reg_loss.png", dpi=100)
        
        # Get the final model parameters
        weights = model.weight.detach().numpy()
        biases = model.bias.detach().numpy()
        
        # Compute the predicted values
        y_train_pred = model(x_train_tensor.transpose(0, 1)).detach().numpy()
        y_test_pred = model(x_test_tensor.transpose(0, 1)).detach().numpy()
        
        # Plot the predicted vs true values for train and validation sets
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        axs[0].scatter(y_train, y_train_pred)
        axs[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r-.', lw=2)
        axs[0].set_xlabel('True Values')
        axs[0].set_ylabel('Predictions')
        axs[0].set_title('Training Set')

        axs[1].scatter(y_test, y_test_pred)
        axs[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r-.', lw=2)
        axs[1].set_xlabel('True Values')
        axs[1].set_ylabel('Predictions')
        axs[1].set_title('Testing Set')
        plt.savefig(data_path + "/pred_true.png", dpi=100)

        # Get the final model parameters
        weights = model.weight.detach().numpy()
        biases = model.bias.detach().numpy()

        # Compute the R-squared
        y_train_mean = torch.mean(y_train_tensor)
        ss_tot = torch.sum((y_train_tensor - y_train_mean) ** 2)
        ss_res = torch.sum((torch.tensor(y_train_pred).clone() - y_train_tensor) ** 2)
        train_r2 = 1 - ss_res / ss_tot

        y_test_mean = torch.mean(y_test_tensor)
        ss_tot = torch.sum((torch.tensor(y_test_tensor).detach().clone() - y_test_mean) ** 2)
        ss_res = torch.sum((torch.tensor(y_test_pred).detach().clone() - y_test_tensor) ** 2)
        val_r2 = 1 - ss_res / ss_tot

        return weights, biases, train_r2.item(), val_r2.item(), train_loss, test_loss

    x = reps_overtime.T.to_numpy() # encodings
    y = A_tot.reshape(len(A_tot), 1) # Mecanically made data similaring fatigue
    x_train = x[:, :int(len(x[1])*0.7)]
    y_train = y[:int(len(x[1])*0.7)]
    x_test = x[:, int(len(x[1])*0.7):]
    y_test = y[int(len(x[1])*0.7):]

    weights, biases, r2_train, r2_test, train_loss, test_loss = multivariate_linear_regression(x_train, y_train, x_test, y_test, lr=0.005, n_epochs=args.n_epochs)
    print("Train R-squared: ", round(r2_train,3), '||', "Test R-squared: ", round(r2_test,3))

    end_time = datetime.now()
    time_taken = end_time - start_time
    print("Overall time taken: ", time_taken)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read CSV data from specified dataset.')
    parser.add_argument('--dataset', type=str, choices=['one_ds', 'one_dl', 'eight_d'], help='Specify the dataset to use')
    parser.add_argument('--window_size', type=int, default=50)
    parser.add_argument('--n_epochs', type=int, default=10)
    args = parser.parse_args()

    main(args)