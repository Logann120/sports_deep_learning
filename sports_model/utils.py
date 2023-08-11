import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

def load_data():
    data = np.load('data_final.npy')
    num_data = data.shape[0]
    X, y = data[:, :-1], data[:, -1]
    X = X.reshape((num_data, X.shape[1], 1))
    y = y.reshape((num_data, 1))
    idxs = np.random.permutation(num_data)
    X, y = X[idxs], y[idxs]
    X_test, y_test = X[:int(0.1*num_data)], y[:int(0.1*num_data)]
    X_train, y_train = X[int(0.1*num_data):], y[int(0.1*num_data):]

    return X_train, y_train, X_test, y_test



# Data augmentation functions
def apply_random_noise(data, scale=0.3):
    noise = np.random.normal(size=data.shape, loc=0.0, scale=scale)
    augmented_data = data + noise
    augmented_data = np.clip(augmented_data, -1.0, 1.0)
    return augmented_data

# Define a function for feature swapping
def apply_feature_swapping(data, labels, swap_prob=0.8):
    swapped_data = np.copy(data)
    
    for i in range(len(data)):
        sample_idx = i
        swap_idx = np.random.choice(data.shape[0])

        if labels[i] == labels[swap_idx]:
            for feature_idx in range(len(data[sample_idx])):
                if np.random.random() < swap_prob:
                    temp = swapped_data[sample_idx, feature_idx]
                    swapped_data[sample_idx, feature_idx] = swapped_data[swap_idx, feature_idx]
                    swapped_data[swap_idx, feature_idx] = temp
               
    return swapped_data

def apply_feature_permutation(data, permute_prob=0.1):
    permutd_data = np.copy(data)
    for i in range(len(data)):
        for feature_idx in np.random.choice(range(len(data[i])), 25, replace = False):
            if np.random.random() < permute_prob:
                random_feature = np.random.randint(len(data[i]))
                temp = permutd_data[i, feature_idx]
                permutd_data[i, feature_idx] = permutd_data[i, random_feature]
                permutd_data[i, random_feature] = temp
    return permutd_data

def make_plots():
    df = pd.read_csv('final_data.csv', index_col='Unnamed: 0')
    cor_mat = df.corr()
    sns.heatmap(cor_mat, linewidths=1, square=True)
    plt.savefig('correlation_matrix.png')

    sns.scatterplot(x=df.Home_win_stat, y=df.Yards_gained, hue=df.Home_win_stat)
    plt.savefig('home_win_yards.png')
    
    sns.kdeplot(x=df.Recent_wins, y=df.Recent_wins_away, hue=df.Home_win, fill=True, alpha=0.65,
    gridsize=750, hue_order=[1,0], palette=[(4/255,2/255,225/255), (255/255,125/255,5/255)], thresh=0.3)
    plt.savefig('recent_wins.png')

    
