import numpy as np
import matplotlib.pyplot as plt
import time
import tensorflow as tf
from tensorflow.keras.layers import Input, Flatten, Dense, Conv1D, AveragePooling1D, Concatenate, Reshape, LSTM
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2



def conv_W(input_shape):
    inputs = Input(shape=input_shape)

    # Layer 1: Extract initial features
    X = Conv1D(filters=8, kernel_size=5, strides=1, padding='same',
               kernel_initializer=GlorotUniform(), activation='relu')(inputs)
    X = AveragePooling1D(pool_size=2, strides=2, padding='same')(X)
    
    # Layer 2: Further feature extraction
    X = Conv1D(filters=12, kernel_size=3, strides=1, padding='same',
               kernel_initializer=GlorotUniform(), activation='relu')(X)
    X = AveragePooling1D(pool_size=2, strides=2, padding='same')(X)
    
    # Layer 3: Reduce feature map size
    X = Conv1D(filters=16, kernel_size=3, strides=2, padding='same',
               kernel_initializer=GlorotUniform(), activation='relu')(X)

    X = Flatten()(X)
    X = Dense(16, activation='relu', kernel_initializer=GlorotUniform(), kernel_regularizer=l2(0.001))(X)
    X = Dense(11, activation='linear', kernel_initializer=GlorotUniform())(X)
    
    model = Model(inputs=inputs, outputs=X)
    return model



def conv_S(input_shape):
    inputs = Input(shape=input_shape)
    # Layer 1: Reduce temporal dimension with convolution
    X = Conv1D(filters=4, kernel_size=2, strides=1, padding='same',
               kernel_initializer=GlorotUniform(), activation='relu',
               data_format='channels_last')(inputs)
    X = AveragePooling1D(pool_size=2, strides=2, padding='same', data_format='channels_last')(X)

    # Layer 2: Additional convolution for feature extraction
    X = Conv1D(filters=8, kernel_size=2, strides=1, padding='same',
               kernel_initializer=GlorotUniform(), activation='relu',
               data_format='channels_last')(X)

    X = Flatten()(X)
    X = Dense(4, activation='linear', kernel_initializer=GlorotUniform(), kernel_regularizer=l2(0.001))(X)
    
    model = Model(inputs=inputs, outputs=X)
    return model



def full_model(time_steps, num_units, num_layers, dropout):
    
    w_inputs = {f'w{i}': Input(shape=(52, 1), dtype=tf.float32, name=f'w{i}') for i in range(6)}
    w_out = [conv_W((52,1))(w) for w in w_inputs.values()]
    w_out = Concatenate()(w_out)
    print("\n--- Model Architecture ---")
    print(" - CNN for Weather data - ")
    print("Output W-CNN:", w_out.shape)
    
    print(" - CNN for Soil data - ")
    s_inputs = {f's{i}': Input(shape=(6, 1), dtype=tf.float32, name=f's{i}') for i in range(11)}
    s_out = [conv_S((6,1))(s) for s in s_inputs.values()]
    s_out = Concatenate()(s_out)
    print("Output S-CNN:", s_out.shape)
    
    print(" - Concatenate weather, soil and management data - ")
    m_input = Input(shape=(14, 1), dtype=tf.float32, name='p')
    static_features = Concatenate()([w_out, s_out, Reshape((14,))(m_input)])
    print("W+S+M concatenated:", static_features.shape)
    static_features = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(static_features)
    print("W+S+M after Dense:", static_features.shape)
    
    print(" - LSTM for yield data - ")
    avg_yield_input = Input(shape=(time_steps, 1), dtype=tf.float32, name='avg_yield')
    x = avg_yield_input
    for _ in range(num_layers):
        x = LSTM(num_units, return_sequences=True, dropout=dropout)(x)
    x = LSTM(num_units, return_sequences=False, dropout=dropout)(x)
    print("LSTM output:", x.shape)
    
    print(" - Combine static and dynamic features - ")
    combined = Concatenate()([x, static_features])
    print("Combined:", combined.shape)
    combined = Dense(16, activation='relu')(combined)
    print("Combined Dense:", combined.shape)
    output = Dense(1, activation=None, name='yield')(combined)
    print("Output:", output.shape)

    inputs = {**w_inputs, **s_inputs, 'p': m_input, 'avg_yield': avg_yield_input}
    model = Model(inputs=inputs, outputs=output)
    return model


def cost_function(alpha, beta):
    def loss(y_true, y_pred):       
        huber = Huber(delta=5.0)
        weights = tf.concat([beta * tf.ones_like(y_true[:, :-1]), alpha * tf.ones_like(y_true[:, -1:])], axis=-1)
        per_timestep_loss = huber(y_true, y_pred)
        weighted_loss = tf.reduce_mean(weights * per_timestep_loss)
        return weighted_loss    
    return loss



def get_sample(X, n_batches, time_steps):
    sample = np.zeros(shape = [n_batches, time_steps, X.shape[1] + time_steps])

    for i in range(n_batches):
        r1 = np.random.randint(1979 + time_steps, 2018)            # random start year for each batch
        years = np.array([(r1 + i - time_steps + 1) for i in range(time_steps)])
        
        for j, y in enumerate(years):
            r2 = np.random.randint(0, X[X[:, 1] == y].shape[0])    # random observation for the specific year         
            obs = X[X[:, 1] == y][r2, :]
            
            avg_yield_values = []
            for k in range(time_steps):
                prev_y = y - k - 1
                prev_data = X[X[:, 1] == prev_y]
                
                if prev_y < 1980:
                    avg_yield_values.append(0)
                else:
                    avg_yield_values.append(prev_data[0, -1])
            
            avg_yield_values = np.array(avg_yield_values[::-1])
            obs_with_avg_yield = np.concatenate((obs, avg_yield_values))
            sample[i, j, :] = obs_with_avg_yield

    return sample.reshape(-1, X.shape[1] + time_steps)       # shape (n_batches*time_steps, 396 + time_steps)


def get_sample_test(X, time_steps):
    sample = []
    X_test = X[X[:, 1] == 2018]

    for obs in X_test:
        avg_yield_values = []
        
        for k in range(time_steps):
            prev_y = 2018 - k - 1
            prev_data = X[X[:, 1] == prev_y]
            avg_yield_values.append(prev_data[0, -1])
        
        avg_yield_values = np.array(avg_yield_values[::-1])
        obs_with_avg_yield = np.concatenate((obs, avg_yield_values))
        sample.append(obs_with_avg_yield)

    return np.array(sample).reshape(-1, X.shape[1] + time_steps)    # shape (n_batches*time_steps, 396 + time_steps)


def preprocess_data(X):
    print("--- Preprocessing ---")
    # 1. remove low yield observations
    X = np.nan_to_num(X)
    index_low_yield = X[:,2] < 5
    print("Remove low yield observations: ", np.sum(index_low_yield))
    print("of years: ", X[index_low_yield][:, 1])
    X = X[np.logical_not(index_low_yield)]
    
    # 2. calculate and append average yield of each year
    years = np.arange(1980, 2019)
    avg = {str(year): np.mean(X[X[:, 1] == year][:, 2]) for year in years}
    avg['2018'] = avg['2017']
    X = np.concatenate((X, np.array([avg[str(int(year))] for year in X[:, 1]]).reshape(-1, 1)), axis=1)
    
    # 3. standardize the data on the training data only
    X_train = X[X[:,1] <= 2017][:, 2:]
    print("Full train data available: ", X_train.shape)

    M=np.mean(X_train, axis=0, keepdims=True)
    S=np.std(X_train, axis=0, keepdims=True)
    epsilon = 1e-8
    
    X[:,2:] = (X[:,2:] - M) / (S + epsilon)
    
    return X, M[0, 0], S[0, 0] 

#
##
### Main program 
##
#


def main_program(X, Max_it, learning_rate, val_fraction, n_batches, time_steps, alpha, beta, num_units, num_layers, dropout):
    
    model = full_model(time_steps, num_units, num_layers, dropout)
    model.compile(optimizer = Adam(learning_rate = learning_rate), 
                  loss = cost_function(alpha, beta))
       
    total_parameters = 0
    for variable in model.trainable_weights:
        shape = variable.shape
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim
        total_parameters += variable_parameters
    print("Total parameters of the model: ",total_parameters, "\n")
    

    X, m, s = preprocess_data(X)

    X_test = get_sample_test(X, time_steps)
    X_test = X_test[:, 3:]                        # without loc_id, year, yield // shape (*, 393 + time_steps)
    X_test = np.expand_dims(X_test, axis=-1)    # for input format
    y_test = X[X[:, 1] == 2018][:, 2].reshape(-1, 1, 1)


    full_batch = get_sample(X, n_batches, time_steps) if n_batches > 0 else X
    train_size = int(n_batches * (1 - val_fraction)) * time_steps
    assert train_size % time_steps == 0
    
    X_train = full_batch[:train_size]
    y_train = X_train[:, 2].reshape(-1, 1, 1)
    X_train = X_train[:, 3:]           # without loc_id, year, yield // shape (*, 393 + time_steps)
    X_train = np.expand_dims(X_train, axis=-1)   # for input format
    
    X_val = full_batch[train_size:]
    y_val = X_val[:, 2].reshape(-1, 1, 1)
    X_val = X_val[:, 3:]               # without loc_id, year, yield // shape (*, 393 + time_steps)
    X_val = np.expand_dims(X_val, axis=-1)     # for input format

    print("- Preprocessed data -")
    print("Train data", X_train.shape)
    print("Validation data", X_val.shape)
    print("Test data", X_test.shape)
    print(f"Test data has mean {round(np.mean(y_test),2)} and std {round(np.std(y_test),2)}.\n")

 
    X_train_in = {
        f'w{i}': X_train[:, 52*i:52*(i+1), :] for i in range(6)
    }
    X_train_in.update({
        f's{i}': X_train[:, 312+6*i:312+6*(i+1), :] for i in range(11)
    })
    X_train_in['p'] = X_train[:, 378:392, :]
    X_train_in['avg_yield'] = X_train[:, -time_steps:, :]
    
    
    X_val_in = {
        f'w{i}': X_val[:, 52*i:52*(i+1), :] for i in range(6)
    }
    X_val_in.update({
        f's{i}': X_val[:, 312+6*i:312+6*(i+1), :] for i in range(11)
    })
    X_val_in['p'] = X_val[:, 378:392, :]
    X_val_in['avg_yield'] = X_val[:, -time_steps:, :]


    for key, x in X_train_in.items():
        assert x.shape[0] == X_train.shape[0]

    def scheduler(epoch, lr):
        if epoch == 60:
            return lr / 2
        elif epoch == 120:
            return lr / 2
        elif epoch == 180:
            return lr / 2
        else:
            return lr
    lr_scheduler = LearningRateScheduler(scheduler)
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    print("--- Input Shapes ---")
    for key, value in X_train_in.items():
        print(f"{key}: {value.shape}")
    print("Y_train shape:", y_train.shape, "\n")
    

    t1=time.time()
    history = model.fit(X_train_in, y_train,
                        epochs=Max_it,
                        validation_data=(X_val_in, y_val),
                        callbacks=[lr_scheduler, early_stopping])
    t2=time.time()  

    # Training data
    y_train_p = model.predict(X_train_in) 
    y_train_p = y_train_p.reshape(-1, 1) * s + m
    y_train = y_train.reshape(-1, 1) * s + m
    rmse_tr = np.sqrt(np.mean((y_train - y_train_p) ** 2))

    # Validation data
    y_val_p = model.predict(X_val_in)
    y_val_p = y_val_p.reshape(-1, 1) * s + m
    y_val = y_val.reshape(-1, 1) * s + m
    rmse_val = np.sqrt(np.mean((y_val - y_val_p) ** 2))
    
    # Testing
    X_test_in = {
        f'w{i}': X_test[:, 52*i:52*(i+1), :] for i in range(6)
    }
    X_test_in.update({
        f's{i}': X_test[:, 312+6*i:312+6*(i+1), :] for i in range(11)
    })
    X_test_in['p'] = X_test[:, 378:392, :]
    X_test_in['avg_yield'] = X_test[:, -time_steps:, :]

    for key, x in X_test_in.items():
        assert x.shape[0] == X_test.shape[0]

    y_test_p = model.predict(X_test_in)
    y_test_p = y_test_p.reshape(-1, 1) * s + m
    y_test = y_test.reshape(-1, 1) * s + m
    
    rmse_te = np.sqrt(np.mean((y_test - y_test_p) ** 2))
    
    print(f"Training time was {round(t2-t1, 2)} seconds.")
    print(f"RMSE of training data: {rmse_tr}")
    print(f"RMSE of validation data: {rmse_val}")
    print(f"RMSE of test data: {rmse_te}")
    
    model.save('./model_soybean.keras')  # Saving the model
    
    # Plot the loss curves for training and validation
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./assets/loss_curve.png')
    
    # Visualize Actual vs Predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_train, y_train_p, label='Training Data', alpha=0.6, color='orange')
    plt.scatter(y_val, y_val_p, label='Validation Data', alpha=0.6, color='blue')
    plt.scatter(y_test, y_test_p, label='Test Data', alpha=0.6, color='green')
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], color='red', linestyle='--')
    plt.title('Actual vs Predicted')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.legend()
    plt.savefig('./assets/val_actual_vs_predicted.png')
    
    return  rmse_tr, rmse_val, rmse_te


#################################################################################################################################################################
#################################################################################################################################################################

# Next Steps:
#     4. Use 2017 as validation set
#     6. Add attention mechanism (optional)
#     7. Add LRP
#     8. Hyperparameter tuning

#################################################################################################################################################################
#################################################################################################################################################################

BigX = np.load('data/soybean_data_compressed.npz') ## order: locID, year, yield, W(52*6), S(6*11), P(14)
X=BigX['data'] 

del BigX

# Parameters
Max_it = 1000                 #150000 could also be used with early stopping
learning_rate = 0.0003        # Learning rate

val_fraction = 0.1            # Fraction of data used for validation
n_batches = 4000               # traning batch size
time_steps = 5                # Number of time steps for the RNN

alpha = 1                     # Weight of loss for final time step
beta = 0.1                      # Weight of loss for years before final time steps

num_units = 64                # Number of hidden units for LSTM cells
num_layers = 2                # Number of layers of LSTM cell
dropout = 0.3                 # Dropout rate

rmse_tr, train_loss, rmse_te = main_program(X, Max_it, learning_rate,
                                            val_fraction, n_batches, time_steps, 
                                            alpha, beta, 
                                            num_units, num_layers, dropout)