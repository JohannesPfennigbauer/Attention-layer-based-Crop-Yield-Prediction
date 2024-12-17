import numpy as np

def Preprocess(time_steps, start_year, save=False):
    """
    Main function to execute the preprocessing. More details in the functions preprocess_data and format_data.
    """
    BigX = np.load('../data/soybean_data_compressed.npz') ## order: locID, year, yield, W(52*6), S(6*11), P(14)
    X=BigX['data']
    del BigX
    
    X, M, S = preprocess_data(X, time_steps)
    X_train_in, y_train, X_val_in, y_val, X_test_in, y_test = split_data(X, start_year)
    
    if save:
        np.savez_compressed('data/soybean_data_preprocessed.npz', 
                            X_train=X_train_in, y_train=y_train, 
                            X_val=X_val_in, y_val=y_val, 
                            X_test=X_test_in, y_test=y_test, 
                            m=m, s=s)

        print("Preprocessed data saved.")
    
    return X_train_in, y_train, X_val_in, y_val, X_test_in, y_test, M, S
    
    
def preprocess_data(X, time_steps):
    """
    Preprocessing steps:
    1. Remove low yield observations
    2. Calculate average yield of each year and standardize it
    3. Standardize the data on the training data only
    4. Add time steps
    """
    
    print("--- Preprocessing ---")
    # 1. remove low yield observations
    X = np.nan_to_num(X)
    index_low_yield = X[:,2] < 5
    print("Remove low yield observations: ", np.sum(index_low_yield))
    print("of years: ", X[index_low_yield][:, 1])
    X = X[np.logical_not(index_low_yield)]
    
    # 2. calculate average yield of each year and standardize it
    years = np.arange(1980, 2017)  # Exclude the last two years (2017 and 2018) for standardization
    _avg = {str(year): np.mean(X[X[:, 1] == year][:, 2]) for year in years}
    avg_m = np.mean(list(_avg.values()))
    avg_s = np.std(list(_avg.values()))
    
    years = np.arange(1980, 2019)
    avg = {str(year): np.mean(X[X[:, 1] == year][:, 2]) for year in years}
    avg = {str(year): (value - avg_m) / avg_s for year, value in avg.items()}
    
    # 3. standardize the data on the training data only
    X_train = X[X[:,1] <= 2016][:, 2:]
    print("Full train data available: ", X_train.shape)

    M=np.mean(X_train, axis=0, keepdims=True)
    S=np.std(X_train, axis=0, keepdims=True)
    epsilon = 1e-8
    
    X[:,2:] = (X[:,2:] - M) / (S + epsilon)
    
    # 4. add time steps  
    for i in range(time_steps):
        avg_prev = np.array([avg[str(int(year - i))] if (year - i) > 1979 else 0 for year in X[:, 1] ])
        X = np.concatenate((X, avg_prev.reshape(-1, 1)), axis=1)
        M = np.concatenate((M, avg_m.reshape(-1, 1)), axis=1)
        S = np.concatenate((S, avg_s.reshape(-1, 1)), axis=1)
    
    # Assertions to verify the preprocessing steps
    assert X.shape[1] == 395 + time_steps, "The number of features after adding time steps is incorrect."
    assert not np.any(np.isnan(X)), "There are NaN values in the preprocessed data."
    
    return X, M, (S + epsilon)



def split_data(X, start_year):
    """
    split data into train, validation and test sets
    """
    
    X_train = X[(X[:, 1] <= 2016) & (X[:, 1] >= start_year)]
    y_train = X_train[:, 2].reshape(-1, 1, 1)
    X_train = X_train[:, 3:]          # without loc_id, year, yield // shape (*, 392 + time_steps)
    print(f"Train data used: {X_train.shape}, starting from year {start_year}.")

    X_val = X[X[:, 1] == 2017][:, 3:]
    y_val = X[X[:, 1] == 2017][:, 2].reshape(-1, 1, 1)

    X_test = X[X[:, 1] == 2018][:, 3:]  
    y_test = X[X[:, 1] == 2018][:, 2].reshape(-1, 1, 1)

    print("- Preprocessed data -")
    print("Train data", X_train.shape)
    print("Validation data", X_val.shape)
    print("Test data", X_test.shape)
    print(f"Test data has mean {round(np.mean(y_test),2)} and std {round(np.std(y_test),2)}.\n")

    return X_train, y_train, X_val, y_val, X_test, y_test


def get_sample(X, batch_size):
    """
    Get a random sample of the data. For testing purposes only, not used in final model.
    """
    
    sample = np.zeros(shape = [batch_size, X.shape[1]])

    for i in range(batch_size):
        r = np.random.randint(X.shape[0])       # random index
        obs = X[r]
        sample[i] = obs

    return sample.reshape(-1, X.shape[1])      # shape (batch_size, 395 + time_steps)