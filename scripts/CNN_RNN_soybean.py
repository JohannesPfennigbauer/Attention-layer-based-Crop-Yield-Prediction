import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.layers import Input, Flatten, Dense, Conv1D, ReLU, AveragePooling1D, Concatenate, Reshape, LSTM
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

def conv_W(input_shape):
    inputs = Input(shape=input_shape)
    print(f"inputs: {inputs.shape}")
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


def full_model(num_units, num_layers, dropout):
    w_inputs = {f'w{i}': Input(shape=(52, 1), dtype=tf.float32, name=f'w{i}') for i in range(6)}
    w_out = [conv_W((52,1))(w) for w in w_inputs.values()]
    
    s_inputs = {f's{i}': Input(shape=(6, 1), dtype=tf.float32, name=f's{i}') for i in range(11)}
    s_out = [conv_S((6,1))(s) for s in s_inputs.values()]

    m_input = Input(shape=(14, 1), dtype=tf.float32, name='p')
    avg_yield_input = Input(shape=(1, 1), dtype=tf.float32, name='avg_yield')
    
    concatenated = Concatenate()(w_out + s_out + [Flatten()(m_input), Flatten()(avg_yield_input)])

    # Create RNN part
    x = Dense(128, activation='relu')(concatenated)
    x = Reshape((1, -1))(x)
    for _ in range(num_layers):
        x = LSTM(num_units, return_sequences=True, dropout=dropout)(x)
    x = LSTM(num_units, return_sequences=True, dropout=dropout)(x)
    
    output = Dense(1, activation=None, name='yield')(x)

    inputs = {**w_inputs, **s_inputs, 'p': m_input, 'avg_yield': avg_yield_input}
    model = Model(inputs=inputs, outputs=output)
    return model


def cost_function(le, l):
    def loss(Y, Yhat):
        # Y, Y_2 = tf.split(Y, [1, 4], axis=-1)
        # Yhat, Yhat_2 = tf.split(Yhat, [1, 4], axis=-1)
        
        huber_loss = Huber(delta=5.0)
        Loss1 = huber_loss(Y, Yhat)
        # Loss2 = huber_loss(Y_2, Yhat_2)
        # Loss = le * Loss1 + l * Loss2
        return Loss1    
    return loss


def get_sample(dic, A, avg, batch_size, time_steps, num_features):
    A_tr = A[:-1, :]
    out = np.zeros(shape=[batch_size, time_steps, num_features])

    for i in range(batch_size):
        r1 = np.random.randint(A_tr.shape[0])
        years = A_tr[r1, :]

        for j, y in enumerate(years):
            X = dic[str(y)]     # Get the data of year y (multiple observations)
            ym = avg[str(y)]    # Get the average normalized yield of year y
            r2 = np.random.randint(X.shape[0])
            X_r2 = X[r2, :].reshape(1, -1) # Get a random observation of year y
            out[i, j, :] = np.concatenate((X_r2, np.array([[ym]])), axis=1).flatten()
    return out # shape (batch_size=25, time_steps=5, features=396)

def get_sample_te(dic, mean_last, avg, batch_size, time_steps, num_features):
    out = np.zeros(shape = [batch_size, time_steps, num_features])
    X = dic[str(2018)]
    
    # Fill the first 4 time steps with mean_last
    out[:, 0:4, :] += mean_last.reshape(1, 4, 3 + 6*52 + 66 + 14 + 1)
    # Fill the last time step with the data of 2018
    ym = np.zeros(shape=[batch_size, 1]) + avg['2018']
    out[:, 4 ,:] = np.concatenate((X,ym),axis=1)

    return out


#
##
### Main program 
##
#

def main_program(X, Index, Max_it, batch_size, time_steps, learning_rate, le, l, num_units, num_layers, dropout):
    
    model = full_model(num_units, num_layers, dropout)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=cost_function(le, l))
       
    total_parameters = 0
    for variable in model.trainable_weights:
        shape = variable.shape
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim
        total_parameters += variable_parameters
    print("total_parameters",total_parameters)
                

    # Prepare the data
    A = np.array([[i - 4, i - 3, i - 2, i - 1, i] for i in range(4, 39)]) + 1980
    A = np.vstack(A) # shape (35, 5)
    
    years = np.arange(1980, 2019)
    dic = {str(year): X[X[:, 1] == year] for year in years}

    avg_values = {str(year): np.mean(X[X[:, 1] == year][:, 2]) for year in years}
    
    avg2 = list(avg_values.values())
    mm = np.mean(avg2)
    ss = np.std(avg2)

    avg = {str(year): (value - mm) / ss for year, value in avg_values.items()}
    avg['2018'] = avg['2017']

    #a2 = np.concatenate((np.mean(dic['2008'], axis=0), [avg['2008']]))
    #a3 = np.concatenate((np.mean(dic['2009'], axis=0), [avg['2009']]))
    #a4 = np.concatenate((np.mean(dic['2010'], axis=0), [avg['2010']]))
    #a5 = np.concatenate((np.mean(dic['2011'], axis=0), [avg['2011']]))
    #a6 = np.concatenate((np.mean(dic['2012'], axis=0), [avg['2012']]))
    #a7 = np.concatenate((np.mean(dic['2013'], axis=0), [avg['2013']]))
    a8 = np.concatenate((np.mean(dic['2014'], axis=0), [avg['2014']]))
    a9 = np.concatenate((np.mean(dic['2015'], axis=0), [avg['2015']]))
    a10 = np.concatenate((np.mean(dic['2016'], axis=0), [avg['2016']]))
    a11 = np.concatenate((np.mean(dic['2017'], axis=0), [avg['2017']]))
    mean_last = np.concatenate((a8, a9, a10,a11))

    out_tr = get_sample(dic, A, avg, batch_size, time_steps=5, num_features=3+312+66+14+1)
    Batch_X_e = out_tr[:, :, 3:].reshape(-1, 6*52+66+14+1) # without loc_id, year, yield # shape (25*5, 393)
    Batch_X_e = np.expand_dims(Batch_X_e, axis=-1)
    
    X_train = {
        f'w{i}': Batch_X_e[:, 52*i:52*(i+1), :] for i in range(6)
    }
    X_train.update({
        f's{i}': Batch_X_e[:, 312+6*i:312+6*(i+1), :] for i in range(11)
    })
    X_train['p'] = Batch_X_e[:, 378:392, :]
    X_train['avg_yield'] = Batch_X_e[:, -1, :]
    X_train['avg_yield'] = np.expand_dims(X_train['avg_yield'], axis=-1)

    
    for key, x in X_train.items():
        assert x.shape[0] == Batch_X_e.shape[0]

    Y_train = out_tr[:, :, 3].reshape(-1, 1, 1)
    
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
    for key, value in X_train.items():
        print(f"{key}: {value.shape}")
    print('Y_train shape', Y_train.shape)
    
    train_loss = []
    
    t1=time.time()
    model.fit(X_train, Y_train, epochs=Max_it, callbacks=[lr_scheduler])
    t2=time.time()
        
    train_loss.append(model.history.history['loss'])
    
    Y_true = out_tr[:, :, 3].reshape(-1, 1)
    Y_true = Y_true * ss + mm
    
    Y_pred = model.predict(X_train)
    Y_pred = Y_pred * ss + mm
    Y_pred = Y_pred.reshape(-1, 1)
    
    rmse_tr = np.sqrt(np.mean((Y_true - Y_pred) ** 2))
    print('RMSE of training data', rmse_tr)

    #############################################################################
    # Validation
    out_te = get_sample_te(dic, mean_last, avg, np.sum(Index), time_steps=5, num_features=3+312+66+14+1)

    Batch_X_te = out_te[:, :, 3:].reshape(-1, 312 + 66 + 14 + 1)
    Batch_X_te = np.expand_dims(Batch_X_te, axis=-1)
    
    X_test = {
        f'w{i}': Batch_X_te[:, 52*i:52*(i+1), :] for i in range(6)
    }
    X_test.update({
        f's{i}': Batch_X_te[:, 312+6*i:312+6*(i+1), :] for i in range(11)
    })
    X_test['p'] = Batch_X_te[:, 378:392, :]
    X_test['avg_yield'] = Batch_X_te[:, -1, :]
    X_test['avg_yield'] = np.expand_dims(X_test['avg_yield'], axis=-1)

    
    for key, x in X_test.items():
        assert x.shape[0] == Batch_X_te.shape[0]

    Y_te_true = out_te[:, :, 3].reshape(-1, 1)
    Y_te_true = Y_te_true * ss + mm
    
    Y_te_pred = model.predict(X_test)
    Y_te_pred = Y_te_pred * ss + mm
    Y_te_pred = Y_te_pred.reshape(-1, 1)
    
    rmse_te = np.sqrt(np.mean((Y_te_true - Y_te_pred) ** 2))
    print('RMSE of test data', rmse_te)
    
    
    print('the training time was %f' % (round(t2-t1, 2)))
    model.save('./model_soybean.keras')  # Saving the model

    return  rmse_tr, train_loss, rmse_te



#################################################################################################################################################################
#################################################################################################################################################################

# Next Steps:
#     1. Rewrite train/test split ("Prepare data" part)
#     2. Use 2007-2017 for training with average_yield_last_year as input
#     3. Use 2018 for testing
#     4. Review loss function, what about previous years?
#     5. Hyperparameter tuning
#     6. Add attention mechanism (optional)
#     7. Add LRP
#     8. Add visualization

#################################################################################################################################################################
#################################################################################################################################################################

BigX = np.load('data/soybean_data_compressed.npz') ##order W(52*6) S(6*11) P(14)
X=BigX['data']

X_tr=X[X[:,1]<=2017]
X_tr=X_tr[:,3:]

M=np.mean(X_tr,axis=0,keepdims=True)
S=np.std(X_tr,axis=0,keepdims=True)
X[:,3:]=(X[:,3:]-M)/S

X=np.nan_to_num(X)
index_low_yield=X[:,2]<5
print('low yield observations',np.sum(index_low_yield))
print(X[index_low_yield][:,1])
X=X[np.logical_not(index_low_yield)]
del BigX

Index=X[:,1]==2018  #validation year

print('Std %.2f and mean %.2f  of test ' %(np.std(X[Index][:,2]),np.mean(X[Index][:,2])))
print("train data",np.sum(np.logical_not(Index)))
print("test data",np.sum(Index))

# Parameters
Max_it = 1000               #150000 could also be used with early stopping
learning_rate = 0.0003        # Learning rate
batch_size = 50               # traning batch size
time_steps = 5                # Number of time steps for the RNN

# LSTM parameters
le = 0.0                      # Weight of loss for years before final time steps
l = 1.0                       # Weight of loss for final time step
num_units = 64                # Number of hidden units for LSTM celss
num_layers = 1                # Number of layers of LSTM cell
dropout = 0.0                 # Dropout rate

print(X.shape)
rmse_tr, train_loss, rmse_te = main_program(X, Index, Max_it,
                                            batch_size, time_steps, learning_rate,
                                            le, l, num_units, num_layers, dropout)