import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.layers import Input, Flatten, Dense, Conv1D, AveragePooling1D, Concatenate, Reshape, LSTM
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model



def conv_W(input_shape):
    inputs = Input(shape=input_shape)

    # Layer 1
    X = Conv1D(filters=8, kernel_size=9, strides=1, padding='valid',
               kernel_initializer=GlorotUniform(), activation='relu',
               data_format='channels_last')(inputs)
    X = AveragePooling1D(pool_size=2, strides=2, data_format='channels_last')(X)
    
    # Layer 2
    X = Conv1D(filters=12, kernel_size=3, strides=1, padding='valid',
               kernel_initializer=GlorotUniform(), activation='relu',
               data_format='channels_last')(X)
    X = AveragePooling1D(pool_size=2, strides=2, data_format='channels_last')(X)
    
    # Layer 3
    X = Conv1D(filters=16, kernel_size=3, strides=1, padding='valid',
               kernel_initializer=GlorotUniform(), activation='relu',
               data_format='channels_last')(X)
    X = AveragePooling1D(pool_size=2, strides=2, data_format='channels_last')(X)
    
    # Layer 4
    X = Conv1D(filters=20, kernel_size=3, strides=1, padding='valid',
               kernel_initializer=GlorotUniform(), activation='relu',
               data_format='channels_last')(X)
    X = AveragePooling1D(pool_size=2, strides=2, data_format='channels_last')(X)
    X = Flatten()(X)
    model = Model(inputs=inputs, outputs=X)
    return model



def conv_S(input_shape):
    inputs = Input(shape=input_shape)
    # Layer 1
    X = Conv1D(filters=4, kernel_size=2, strides=1, padding='same',
               kernel_initializer=GlorotUniform(), activation='relu',
               data_format='channels_last')(inputs)
    X = AveragePooling1D(pool_size=2, strides=2, padding='same', data_format='channels_last')(X)

    # Layer 2
    X = Conv1D(filters=8, kernel_size=2, strides=1, padding='same',
               kernel_initializer=GlorotUniform(), activation='relu',
               data_format='channels_last')(X)

    # Layer 3
    X = Conv1D(filters=10, kernel_size=2, strides=1, padding='valid',
               kernel_initializer=GlorotUniform(), activation='relu',
               data_format='channels_last')(X)

    # Layer 4 (to reduce length to 1)
    X = Conv1D(filters=10, kernel_size=2, strides=1, padding='valid',
               kernel_initializer=GlorotUniform(), activation='relu',
               data_format='channels_last')(X)
    X = Flatten()(X)
    model = Model(inputs=inputs, outputs=X)
    return model



def full_model(time_steps, num_units, num_layers, dropout):
    w_inputs = {f'w{i}': Input(shape=(52, 1), dtype=tf.float32, name=f'w{i}') for i in range(6)}
    w_out = [conv_W((52,1))(w) for w in w_inputs.values()]
    
    s_inputs = {f's{i}': Input(shape=(6, 1), dtype=tf.float32, name=f's{i}') for i in range(11)}
    s_out = [conv_S((6,1))(s) for s in s_inputs.values()]

    m_input = Input(shape=(14, 1), dtype=tf.float32, name='p')
    
    static_features = Concatenate()(w_out + s_out + [Flatten()(m_input)])
    static_features = Dense(128, activation='relu')(static_features)
    
    # RNN part
    avg_yield_input = Input(shape=(time_steps, 1), dtype=tf.float32, name='avg_yield')
    
    x = avg_yield_input
    for _ in range(num_layers):
        x = LSTM(num_units, return_sequences=True, dropout=dropout)(x)
    x = LSTM(num_units, return_sequences=False, dropout=dropout)(x)
    
    # Combine all together
    combined = Concatenate()([x, static_features])
    output = Dense(1, activation=None, name='yield')(combined)

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



def get_sample(X, batch_size, time_steps):
    sample = np.zeros(shape = [batch_size, time_steps, X.shape[1] + time_steps])

    for i in range(batch_size):
        r1 = np.random.randint(1979 + time_steps, 2019)            # random start year for each batch
        years = np.array([(r1 + i - time_steps + 1) for i in range(time_steps)])
        
        for j, y in enumerate(years):
            print(X[X[:, 1] == y].shape)
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

    return sample.reshape(-1, X.shape[1] + time_steps)       # shape (batch_size*time_steps, 396 + time_steps)


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

    return np.array(sample).reshape(-1, X.shape[1] + time_steps)    # shape (batch_size*time_steps, 396 + time_steps)


def preprocess_data(X):
    
    # 1. remove low yield observations
    X = np.nan_to_num(X)
    index_low_yield = X[:,2] < 5
    print('low yield observations', np.sum(index_low_yield))
    print(X[index_low_yield][:, 1])
    X = X[np.logical_not(index_low_yield)]
    
    # 2. calculate and append average yield of each year
    years = np.arange(1980, 2019)
    avg = {str(year): np.mean(X[X[:, 1] == year][:, 2]) for year in years}
    avg['2018'] = avg['2017']
    X = np.concatenate((X, np.array([avg[str(int(year))] for year in X[:, 1]]).reshape(-1, 1)), axis=1)
    
    # 3. standardize the data on the training data only
    X_train = X[X[:,1] <= 2017][:, 3:]

    M=np.mean(X_train, axis=0, keepdims=True)
    S=np.std(X_train, axis=0, keepdims=True)
    epsilon = 1e-8
    
    X[:,3:] = (X[:,3:] - M) / (S + epsilon)
    return X

#
##
### Main program 
##
#


def main_program(X, Max_it, learning_rate, batch_size, time_steps, alpha, beta, num_units, num_layers, dropout):
    
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
    print("total_parameters",total_parameters)
    

    X = preprocess_data(X)
    
    X_test = X[X[:, 1] == 2018][:, 3:]           # without loc_id, year, yield // shape (*, 393 + time_steps)
    X_test = get_sample_test(X, time_steps) if batch_size > 0 else X_test
    X_test = np.expand_dims(X_test, axis=-1)    # for input format
    y_test = X[X[:, 1] == 2018][:, 2].reshape(-1, 1, 1)
    
    X = get_sample(X, batch_size, time_steps) if batch_size > 0 else X
    X_train = X[X[:, 1] <= 2017][:, 3:]           # without loc_id, year, yield // shape (*, 393 + time_steps)
    X_train = np.expand_dims(X_train, axis=-1)   # for input format
    y_train = X[X[:, 1] <= 2017][:, 2].reshape(-1, 1, 1)

    print('Std %.2f and mean %.2f  of test ' %(np.std(y_test[:]), np.mean(y_test[:])))
    print("train data", X_train.shape)
    print("test data", X_test.shape)

    X_train_in = {
        f'w{i}': X_train[:, 52*i:52*(i+1), :] for i in range(6)
    }
    X_train_in.update({
        f's{i}': X_train[:, 312+6*i:312+6*(i+1), :] for i in range(11)
    })
    X_train_in['p'] = X_train[:, 378:392, :]
    X_train_in['avg_yield'] = X_train[:, -time_steps:, :]

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
    
    print('INPUT SHAPES')
    for key, value in X_train_in.items():
        print(f"{key}: {value.shape}")
    print('Y_train shape:', y_train.shape)
    
    train_loss = []
    t1=time.time()
    
    model.fit(X_train_in, y_train, epochs=Max_it, callbacks=[lr_scheduler])
    
    t2=time.time()  
    train_loss.append(model.history.history['loss'])
      
    y_pred = model.predict(X_train_in)
    y_train = y_train.reshape(-1, 1)
    y_pred = y_pred.reshape(-1, 1)
    
    rmse_tr = np.sqrt(np.mean((y_train - y_pred) ** 2))
    print('RMSE of training data:', rmse_tr)

    #############################################################################
    # Validation
    #############################################################################
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
    y_test_p = y_test_p.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    
    rmse_te = np.sqrt(np.mean((y_test - y_test_p) ** 2))
    print('RMSE of test data:', rmse_te)
    
    print(f"the training time was {round(t2-t1, 2)} seconds.")
    model.save('./model_soybean.keras')  # Saving the model

    return  rmse_tr, train_loss, rmse_te


#################################################################################################################################################################
#################################################################################################################################################################

# Next Steps:
#     5. Hyperparameter tuning
#     6. Add attention mechanism (optional)
#     7. Add LRP
#     8. Add visualization

#################################################################################################################################################################
#################################################################################################################################################################

BigX = np.load('data/soybean_data_compressed.npz') ## order: locID, year, yield, W(52*6), S(6*11), P(14)
X=BigX['data'] 

del BigX

# Parameters
Max_it = 100                 #150000 could also be used with early stopping
learning_rate = 0.003        # Learning rate

batch_size = 50               # traning batch size
time_steps = 5                # Number of time steps for the RNN

alpha = 1                     # Weight of loss for final time step
beta = 0.1                      # Weight of loss for years before final time steps

num_units = 64                # Number of hidden units for LSTM celss
num_layers = 1                # Number of layers of LSTM cell
dropout = 0.0                 # Dropout rate

rmse_tr, train_loss, rmse_te = main_program(X, Max_it, learning_rate,
                                            batch_size, time_steps, 
                                            alpha, beta, 
                                            num_units, num_layers, dropout)