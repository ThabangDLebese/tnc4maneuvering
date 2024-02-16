import os
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller


def main(args):

    start_time = datetime.now()
    
    data_path = './tnc4maneuvering/dataset/'
    # Choose the dataset based on the argument
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
    dataset = pd.read_csv(file_path)

    x = dataset['longacceleration'].values.astype(np.float32)
    y = dataset['latacceleration'].values.astype(np.float32)


    def evaluate_window(window_size, num_evaluations):
        results = []
        for _ in range(num_evaluations):
            dx = np.gradient(x)
            dy = np.gradient(y)
            
            num_samples = len(dx) // window_size
            sample_indices = np.random.choice(range(num_samples), num_samples, replace=False)
            window_starts = [i * window_size for i in sample_indices]
            window_ends = [(i + 1) * window_size for i in sample_indices]
            
            dx_windows = np.array([np.mean(dx[i*window_size:(i+1)*window_size]) for i in sample_indices])
            dy_windows = np.array([np.mean(dy[i*window_size:(i+1)*window_size]) for i in sample_indices])
            
            p_values = []  # List for storing p-values
            for i in sample_indices:
                dx_sample = x[i*window_size:(i+1)*window_size]
                dy_sample = y[i*window_size:(i+1)*window_size]
                adfx_pvalue = adfuller(dx_sample)[1]
                adfy_pvalue = adfuller(dy_sample)[1]
                p_values.append((adfx_pvalue, adfy_pvalue)) 
                
            # Check if the gradient absolute threshold is satisfied for any of the samples
            is_satisfied = any(
                np.all(dx_sample >= positive_threshold) and np.all(dy_sample >= positive_threshold)
                and np.all(dx_sample <= negative_threshold) and np.all(dy_sample <= negative_threshold)
                for dx_sample, dy_sample in zip(dx_windows, dy_windows))
            
            results.append((dx_windows, dy_windows, p_values, is_satisfied, window_starts, window_ends))
        return results

    # Set the gradient absolute threshold
    positive_threshold = 0.0002
    negative_threshold = -0.0002

    # Set initial window size
    window_size =  args.window_size0

    while window_size <=  args.window_sizeT:
        trials_count = 0
        
        print('-'*30)
        print("Evaluating window size:", window_size)
        results = evaluate_window(window_size, args.n_evals)
        
        for i, (dx_windows, dy_windows, p_values, is_satisfied, window_starts, window_ends) in enumerate(results):
            trials_count += 1
            print("Trial {}: Threshold Satisfied: {}".format(i+1, is_satisfied))
            
            # Create a pandas table for each trial
            trial_table = pd.DataFrame({
                'dx_grads': dx_windows,
                'dy_grads': dy_windows,
                'adfx_pvals': [p[0] for p in p_values],
                'adfy_pvals': [p[1] for p in p_values]
            })
            
            # You can uncomment this section if yuo want to visualise changes in each window
            plt.figure(figsize=(12, 6))
            plt.subplot2grid((3, 2), (1, 1), colspan=1)
            ax2 = sns.boxplot(data=trial_table[['adfx_pvals', 'adfy_pvals']], orient='h')
            plt.title('P-values, Trial: {}, window_size: {}'.format(trials_count, window_size))
            plt.xlabel('P-Values of (dx, dy) per window')
            ax2.axvline(x=0.01, color='red', linestyle='--', label='p-value = 0.01')
            plt.legend()
            
            plt.subplot2grid((3, 2), (1, 0), colspan=1)
            ax1 = sns.boxplot(data=trial_table[['dx_grads', 'dy_grads']], orient='h')
            plt.xlabel('(dx, dy) per window')
            plt.title('Gradients, Trial: {}, window_size: {}'.format(trials_count, window_size))
            plt.axvline(x=positive_threshold, color='r', linestyle='--', label='pos-grad')
            plt.axvline(x=-positive_threshold, color='g', linestyle='--', label='neg-grad')
            plt.legend()
            
            # Fourth plot: Scatter plot of dy_grads vs. adfy_pvals
            plt.subplot2grid((3, 2), (2, 0), colspan=2)
            plt.plot(trial_table['dy_grads'], 'o:', label='dy_grads')
            plt.axhline(y=positive_threshold, c='c', ls='--')
            plt.axhline(y=negative_threshold, c='c', ls='--')

            # plt.fill_between(positive_threshold, negative_threshold, where=(np.abs(dx_windows) > positive_threshold),facecolor='gray', alpha=0.5)
            # Plot only -pvalues that are >= 0.01
            for i, p in enumerate(p_values):
                if p[1] >= 0.01:  # Check the -p-value for adfy_pvals
                    plt.annotate('*', xy=(i, trial_table['dy_grads'].iloc[i]), color='r', fontsize=12)
            plt.xlabel('No. Grads')
            plt.title('Y-grads with Ypvals>=0.01')
            plt.legend()

            plt.subplot2grid((3, 2), (0, 0), colspan=2)
            plt.plot(trial_table['dx_grads'], 'o:', label = 'dx_grads') 
            plt.axhline(y=positive_threshold, c='c', ls='--')
            plt.axhline(y=negative_threshold, c='c', ls='--')
            # Plot only -pvalues that are >= 0.01
            for i, p in enumerate(p_values):
                if p[1] >= 0.01:  # Check the -p-value for adfy_pvals
                    plt.annotate('*', xy=(i, trial_table['dx_grads'].iloc[i]), color='r', fontsize=12)
            plt.xlabel('No. Grads')
            plt.title('X-grads with Xpvals>0.01')
            plt.legend()

            plt.tight_layout()
            plt.savefig(data_path +f"grads_adf_window_{window_size}_{args.dataset}.png", dpi=100)
            
            # Calculate the number of p-values >= 0.01
            pv_threshold = 0.01 # Similar to that of TNC
            num_pvalues_adfx_geq_threshold = sum(p[0] >= pv_threshold for p in p_values)
            num_pvalues_adfy_geq_threshold = sum(p[1] >= pv_threshold for p in p_values)
            
            # Calculate the fraction of gradients with p-values >= 0.01
            num_gradients = len(trial_table)
            fraction_adfx_geq_threshold = num_pvalues_adfx_geq_threshold / num_gradients
            fraction_adfy_geq_threshold = num_pvalues_adfy_geq_threshold / num_gradients
            print("Frac dx w/ p-values >= {}: {:.4f}".format(pv_threshold, fraction_adfx_geq_threshold))
            print("Frac dy w/ p-values >= {}: {:.4f}".format(pv_threshold, fraction_adfy_geq_threshold))
            print("Total-Ratio :", (fraction_adfx_geq_threshold + fraction_adfy_geq_threshold).round(4)/2)

        print("Number of trials for window size {}: {}".format(window_size, trials_count))
        if any(is_satisfied for _, _, _, is_satisfied, _, _ in results):
            break
        # Increase the window size
        window_size += args.steps
    print("Final window size:", window_size)
    print( )

    end_time = datetime.now()
    time_taken = end_time - start_time
    print("Overal time taken: ", time_taken) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read CSV data from specified dataset.')
    parser.add_argument('--dataset', type=str, choices=['one_ds', 'one_dl', 'eight_d'], help='Specify the dataset to use')
    parser.add_argument('--window_size0', type=int, default=50)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--window_sizeT', type=int, default=300)
    parser.add_argument('--n_evals', type=int, default=1)
    args = parser.parse_args()

    # Check if data argument is provided
    if not args.dataset:
        print("Please specify the dataset to use using the --data argument.")
        exit()
        
    main(args)