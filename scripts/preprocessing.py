import numpy as np

def Preprocess(time_steps, start_year, save=False):
    BigX = np.load('../data/soybean_data_compressed.npz') ## order: locID, year, yield, W(52*6), S(6*11), P(14)
    X=BigX['data']
    time_steps = 5      # number of time steps to include last years average yield for LSTM
    start_year = 2000   # 1980 for full data
    del BigX
    
    X, m, s = preprocess_data(X, time_steps)
    X_train_in, y_train, X_val_in, y_val, X_test_in, y_test = format_data(X, time_steps, start_year)
    
    if save:
        np.savez_compressed('data/soybean_data_preprocessed.npz', 
                            X_train=X_train_in, y_train=y_train, 
                            X_val=X_val_in, y_val=y_val, 
                            X_test=X_test_in, y_test=y_test, 
                            m=m, s=s)

        print("Preprocessed data saved.")
    
    return X_train_in, y_train, X_val_in, y_val, X_test_in, y_test, m, s
    
    
def preprocess_data(X, time_steps):
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
    
    return X, M[0, 0], (S[0, 0] + epsilon)



def format_data(X, time_steps, start_year):
    
    # training data
    X_train = X[(X[:, 1] <= 2016) & (X[:, 1] >= start_year)]
    # X_train = get_sample(X_train, batch_size) if batch_size > 0 else X_train
    y_train = X_train[:, 2].reshape(-1, 1, 1)
    X_train = X_train[:, 3:]          # without loc_id, year, yield // shape (*, 392 + time_steps)
    print(f"Train data used: {X_train.shape}, starting from {start_year}")

    X_train = np.expand_dims(X_train, axis=-1)    
    X_train_in = {f'w{i}': X_train[:, 52*i:52*(i+1), :] for i in range(6)}
    X_train_in.update({f's{i}': X_train[:, 312+6*i:312+6*(i+1), :] for i in range(11)})
    X_train_in['p'] = X_train[:, 378:392, :]
    X_train_in['avg_yield'] = X_train[:, -time_steps:, :]

    # validation data
    X_val = X[X[:, 1] == 2017][:, 3:]
    y_val = X[X[:, 1] == 2017][:, 2].reshape(-1, 1, 1)

    X_val = np.expand_dims(X_val, axis=-1)
    X_val_in = {f'w{i}': X_val[:, 52*i:52*(i+1), :] for i in range(6)}
    X_val_in.update({f's{i}': X_val[:, 312+6*i:312+6*(i+1), :] for i in range(11)})
    X_val_in['p'] = X_val[:, 378:392, :]
    X_val_in['avg_yield'] = X_val[:, -time_steps:, :]

    # testing data
    X_test = X[X[:, 1] == 2018][:, 3:]  
    y_test = X[X[:, 1] == 2018][:, 2].reshape(-1, 1, 1)

    X_test = np.expand_dims(X_test, axis=-1) 
    X_test_in = {f'w{i}': X_test[:, 52*i:52*(i+1), :] for i in range(6)}
    X_test_in.update({f's{i}': X_test[:, 312+6*i:312+6*(i+1), :] for i in range(11)})
    X_test_in['p'] = X_test[:, 378:392, :]
    X_test_in['avg_yield'] = X_test[:, -time_steps:, :]

    print("- Preprocessed data -")
    print("Train data", X_train.shape)
    print("Validation data", X_val.shape)
    print("Test data", X_test.shape)
    print(f"Test data has mean {round(np.mean(y_test),2)} and std {round(np.std(y_test),2)}.\n")

    return X_train_in, y_train, X_val_in, y_val, X_test_in, y_test


def get_sample(X, batch_size):
    sample = np.zeros(shape = [batch_size, X.shape[1]])

    for i in range(batch_size):
        r = np.random.randint(X.shape[0])       # random index
        obs = X[r]
        sample[i] = obs

    return sample.reshape(-1, X.shape[1])      # shape (batch_size, 395 + time_steps)