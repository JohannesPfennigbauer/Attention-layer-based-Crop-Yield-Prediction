import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Conv1D, ReLU, AveragePooling1D, Concatenate, Reshape, LSTM
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.losses import MeanSquaredError, Huber

def conv_res_part_P(P_t, f, is_training, var_name):
    epsilon=0.0001
    f0=5
    s0=1
    #############stage 1
    X = Flatten()(P_t)
    print('conv2 out P', X)

    return X



def conv_res_part_E(E_t, f, is_training, var_name):
    epsilon=0.0001
    f0=5
    s0=1
    
    # Layer 1
    X = Conv1D(filters=8, kernel_size=9, strides=1, padding='valid',
               kernel_initializer=GlorotUniform(), activation=None,
               name='Conv00' + var_name, data_format='channels_last')(E_t)

    X = ReLU()(X)
    X = AveragePooling1D(pool_size=2, strides=2, data_format='channels_last', name='average_pool')(X)
    
    # Layer 2
    X = Conv1D(filters=12, kernel_size=3, strides=1, padding='valid',
               kernel_initializer=GlorotUniform(), activation=None,
               name='Conv0' + var_name, data_format='channels_last')(X)
    X = ReLU()(X)
    X = AveragePooling1D(pool_size=2, strides=2, data_format='channels_last', name='average_pool')(X)
    
    # Layer 3
    X = Conv1D(filters=16, kernel_size=3, strides=1, padding='valid',
               kernel_initializer=GlorotUniform(), activation=None,
               name='Conv1' + var_name, data_format='channels_last')(X)

    X = ReLU()(X)
    X = AveragePooling1D(pool_size=2, strides=2, data_format='channels_last', name='average_pool')(X)
    
    # Layer 4
    X = Conv1D(filters=20, kernel_size=3, strides=s0, padding='valid',
               kernel_initializer=GlorotUniform(), activation=None,
               name='Conv2'+var_name, data_format='channels_last')(X)
    X = ReLU()(X)
    X = AveragePooling1D(pool_size=2, strides=2, data_format='channels_last', name='average_pool')(X)

    print('E outttt',X)
    return X



def conv_res_part_S(S_t, f, is_training, var_name):
    # Layer 1
    X = Conv1D(filters=4, kernel_size=2, strides=1, padding='same',
               kernel_initializer=GlorotUniform(), activation=None,
               name='Conv1' + var_name, data_format='channels_last')(S_t)
    X = ReLU()(X)
    X = AveragePooling1D(pool_size=2, strides=2, padding='same', data_format='channels_last', name='average_pool')(X)

    # Layer 2
    X = Conv1D(filters=8, kernel_size=2, strides=1, padding='same',
               kernel_initializer=GlorotUniform(), activation=None,
               name='Conv2' + var_name, data_format='channels_last')(X)
    X = ReLU()(X)

    # Layer 3
    X = Conv1D(filters=10, kernel_size=2, strides=1, padding='valid',
               kernel_initializer=GlorotUniform(), activation=None,
               name='Conv3' + var_name, data_format='channels_last')(X)
    X = ReLU()(X)

    # Layer 4 (to reduce length to 1)
    X = Conv1D(filters=10, kernel_size=2, strides=1, padding='valid',
               kernel_initializer=GlorotUniform(), activation=None,
               name='Conv4' + var_name, data_format='channels_last')(X)
    X = ReLU()(X)
    
    return X


def main_proccess(E_t1, E_t2, E_t3, E_t4, E_t5, E_t6, S_t1, S_t2, S_t3, S_t4, S_t5, S_t6, S_t7, S_t8, S_t9, S_t10, S_t11, P_t, Ybar, f, is_training, num_units, num_layers, dropout):

    #g_out=genetic_proccessing(G_t,filter_number,kernel_size,stride,padding,bn_G)
    #g_out=conv_res_part(G_t, f, is_training)
    #print("Conv output-----",g_out)
    #g_out=tf.contrib.layers.flatten(g_out)
    #print("Conv output_flatten---",g_out)

    e_out1 = conv_res_part_E(E_t1, f, is_training=is_training, var_name='v1')
    e_out1 = Flatten()(e_out1)
    e_out2 = conv_res_part_E(E_t2, f, is_training=is_training, var_name='v1')
    e_out2 = Flatten()(e_out2)
    e_out3 = conv_res_part_E(E_t3, f, is_training=is_training, var_name='v1')
    e_out3 = Flatten()(e_out3)
    e_out4 = conv_res_part_E(E_t4, f, is_training=is_training, var_name='v1')
    e_out4 = Flatten()(e_out4)
    e_out5 = conv_res_part_E(E_t5, f, is_training=is_training, var_name='v1')
    e_out5 = Flatten()(e_out5)
    e_out6 = conv_res_part_E(E_t6, f, is_training=is_training, var_name='v1')
    e_out6 = Flatten()(e_out6)

    e_out = Concatenate(axis=1)([e_out1, e_out2, e_out3, e_out4, e_out5, e_out6])
    print('after concatenate',e_out)

    e_out = Dense(units=43, activation='relu', kernel_initializer=GlorotUniform(), bias_initializer='zeros')(e_out)
    e_out = ReLU()(e_out)
    print('e_out_*************',e_out)

    s_out1 = conv_res_part_S(S_t1, f, is_training=is_training, var_name='v1S')
    s_out1 = Flatten()(s_out1)
    s_out2 = conv_res_part_S(S_t2, f, is_training=is_training, var_name='v1S')
    s_out2 = Flatten()(s_out2)
    s_out3 = conv_res_part_S(S_t3, f, is_training=is_training, var_name='v1S')
    s_out3 = Flatten()(s_out3)
    s_out4 = conv_res_part_S(S_t4, f, is_training=is_training, var_name='v1S')
    s_out4 = Flatten()(s_out4)
    s_out5 = conv_res_part_S(S_t5, f, is_training=is_training, var_name='v1S')
    s_out5 = Flatten()(s_out5)
    s_out6 = conv_res_part_S(S_t6, f, is_training=is_training, var_name='v1S')
    s_out6 = Flatten()(s_out6)
    s_out7 = conv_res_part_S(S_t7, f, is_training=is_training, var_name='v1S')
    s_out7 = Flatten()(s_out7)
    s_out8 = conv_res_part_S(S_t8, f, is_training=is_training, var_name='v1S')
    s_out8 = Flatten()(s_out8)
    s_out9 = conv_res_part_S(S_t9, f, is_training=is_training, var_name='v1S')
    s_out9 = Flatten()(s_out9)
    s_out10 = conv_res_part_S(S_t10, f, is_training=is_training, var_name='v1S')
    s_out10 = Flatten()(s_out10)
    s_out11 = conv_res_part_S(S_t11, f, is_training=is_training, var_name='v1S')
    s_out11 = Flatten()(s_out11)

    s_out = Concatenate(axis=1)([s_out1, s_out2, s_out3, s_out4, s_out5, s_out6, s_out7, s_out8, s_out9, s_out10, s_out11])
    print('soil after concatenate', s_out)

    s_out = Dense(units=43, activation=None, kernel_initializer=GlorotUniform(), bias_initializer='zeros')(s_out)
    s_out = ReLU()(s_out)
    print('soil after FC layer', s_out)
    
    
    p_out = conv_res_part_P(P_t,f,is_training,var_name='P')
    p_out = Flatten()(p_out)

    print('p outtttttt',p_out)
    
    
    e_out = Concatenate(axis=1)([e_out, s_out, p_out])
    print('soil + Weather after concatante', e_out)

    time_step=5


    e_out = Reshape(target_shape=[time_step, e_out.shape[1] // time_step])(e_out)
    print('e_out_after_reshapeeeee',e_out)
    
    e_out =  Concatenate(axis=2)([e_out, Ybar])
    cells = []
    
    for _ in range(num_layers):
        cell = LSTM(num_units, return_sequences=True, return_state=False, dropout=dropout, recurrent_dropout=dropout)
        cells.append(cell)

    # # Create a stacked LSTM layer
    # cell = tf.contrib.rnn.MultiRNNCell(cells)
    # output, _= tf.nn.dynamic_rnn(cell, e_out, dtype=tf.float32)
    cell = tf.keras.Sequential(cells)
    output = cell(e_out)
    print('RNN output',output)


    output = Reshape(target_shape=[-1, output.shape[-1]])(output)
    output = Dense(units=1, activation=None, kernel_initializer=GlorotUniform(), bias_initializer='zeros')(output)
    print(output)

    output = Reshape(target_shape=[-1, 5])(output)
    print("output of all time steps", output)
    
    ## WRONG SHAPE OF YHAT1 AND YHAT2 ##
    #######
    ##
    #
    #
    #
    #
    ##
    ## !!!!!!!!!!!!!!!!!!
    Yhat1 = output[:, 4:5, :]
    print('Yhat1111111111', Yhat1)

    Yhat2 = output[:, 0:4, :]
    print('Yhat222222', Yhat2)

    return Yhat1,Yhat2



def Cost_function(Y, Yhat):
    mse = MeanSquaredError()
    huber = Huber(delta=5.0)

    E = Y - Yhat
    E2 = tf.square(E)

    MSE = tf.reduce_mean(E2)
    RMSE = tf.sqrt(MSE)
    Loss = huber(Y, Yhat)

    return RMSE, MSE, E, Loss



def get_sample(dic, L, avg, batch_size, time_steps, num_features):
    L_tr = L[:-1,:]
    out = np.zeros(shape=[batch_size,time_steps,num_features])

    for i in range(batch_size):
        r1 = np.random.randint(L_tr.shape[0])
        years = L_tr[r1, :]

        for j, y in enumerate(years):
            X = dic[str(y)]
            ym = avg[str(y)]
            r2 = np.random.randint(X.shape[0])
            out[i, j, :] = np.concatenate((X[r2, :], np.array([[ym]])), axis=1)

    return out



def get_sample_te(dic, mean_last, avg, batch_size_te, time_steps, num_features):
    out = np.zeros(shape = [batch_size_te, time_steps, num_features])
    X = dic[str(2018)]
    
    # Fill the first 4 time steps with mean_last
    out[:, 0:4, :] += mean_last.reshape(1, 4, 6*52 + 66 + 14)
    # Fill the last time step with the data of 2018
    ym = np.zeros(shape=[batch_size_te,1]) + avg['2018']
    out[:,4,:] = np.concatenate((X,ym),axis=1)

    return out



#
##
### Main program 
##
#

def main_program(X, Index, num_units, num_layers, Max_it, learning_rate, batch_size_tr, le, l, is_training, dropout):

    with tf.device('/cpu:0'):
        E_t1 = tf.keras.Input(shape=(52, 1), dtype=tf.float32, name='E_t1')
        E_t2 = tf.keras.Input(shape=(52, 1), dtype=tf.float32, name='E_t2')
        E_t3 = tf.keras.Input(shape=(52, 1), dtype=tf.float32, name='E_t3')
        E_t4 = tf.keras.Input(shape=(52, 1), dtype=tf.float32, name='E_t4')
        E_t5 = tf.keras.Input(shape=(52, 1), dtype=tf.float32, name='E_t5')
        E_t6 = tf.keras.Input(shape=(52, 1), dtype=tf.float32, name='E_t6')

        S_t1 = tf.keras.Input(shape=(6, 1), dtype=tf.float32, name='S_t1')
        S_t2 = tf.keras.Input(shape=(6, 1), dtype=tf.float32, name='S_t2')
        S_t3 = tf.keras.Input(shape=(6, 1), dtype=tf.float32, name='S_t3')
        S_t4 = tf.keras.Input(shape=(6, 1), dtype=tf.float32, name='S_t4')
        S_t5 = tf.keras.Input(shape=(6, 1), dtype=tf.float32, name='S_t5')
        S_t6 = tf.keras.Input(shape=(6, 1), dtype=tf.float32, name='S_t6')
        S_t7 = tf.keras.Input(shape=(6, 1), dtype=tf.float32, name='S_t7')
        S_t8 = tf.keras.Input(shape=(6, 1), dtype=tf.float32, name='S_t8')
        S_t9 = tf.keras.Input(shape=(6, 1), dtype=tf.float32, name='S_t9')
        S_t10 = tf.keras.Input(shape=(6, 1), dtype=tf.float32, name='S_t10')
        S_t11 = tf.keras.Input(shape=(6, 1), dtype=tf.float32, name='S_t11')

        P_t = tf.keras.Input(shape=(14, 1), dtype=tf.float32, name='P_t')

        Ybar = tf.keras.Input(shape=(5, 1), dtype=tf.float32, name='Ybar')

        Y_t = tf.keras.Input(shape=(1,), dtype=tf.float32, name='Y_t')
        Y_t_2 = tf.keras.Input(shape=(4,), dtype=tf.float32, name='Y_t_2')

        # is_training = tf.keras.Input(shape=(), dtype=tf.bool, name='is_training')
        # lr = tf.keras.Input(shape=(), dtype=tf.float32, name='learning_rate')
        # dropout = tf.keras.Input(shape=(), dtype=tf.float32, name='dropout')

        f = 3
        Yhat1,Yhat2 = main_proccess(E_t1, E_t2, E_t3, E_t4, E_t5, E_t6,
                                    S_t1, S_t2, S_t3, S_t4, S_t5, S_t6, S_t7, S_t8, S_t9, S_t10, S_t11,
                                    P_t, Ybar, f, is_training, num_units, num_layers, dropout)
        
        print('Yhatttttttttt',Yhat1)
        # Yhat2 is the prediction we got before the final time step (year t)
        total_parameters = 0
        for variable in tf.compat.v1.trainable_variables():
            # shape is an array of tf.Dimension
            print(variable)
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print("total_parameters",total_parameters)

        with tf.name_scope('loss_function'):

            RMSE,_,_,Loss1 = Cost_function(Y_t, Yhat1)
            _, _, _, Loss2 = Cost_function(Y_t_2, Yhat2)

            #Yhat2 is the prediction we got before the final time step (year t)
            Tloss=tf.constant(l, dtype = tf.float32) * Loss1 + tf.constant(le, dtype = tf.float32) * Loss2

        RMSE=tf.identity(RMSE,name='RMSE')
        
        with tf.name_scope('train'):
            optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
            train_op = optimizer.minimize(Tloss, var_list = tf.compat.v1.trainable_variables())

        # Initialize variables
        init = tf.compat.v1.global_variables_initializer()
        
        sess = tf.compat.v1.Session()
        sess.run(init)
        
        # Create a summary writer
        writer = tf.summary.create_file_writer("./tensorboard")
        with writer.as_default():
            tf.summary.graph(sess.graph)

        t1=time.time()
        
        A = []
        for i in range(4, 39):
            A.append([ i - 4, i - 3, i - 2, i - 1, i])
        A = np.vstack(A)
        A += 1980
        print(A.shape)

        dic = {}
        for i in range(39):
            dic[str(i + 1980)] = X[X[:, 1] == i + 1980]

        avg = {}
        avg2 = []
        for i in range(39):
            avg[str(i + 1980)] = np.mean(X[X[:, 1] == i + 1980][:, 2])
            avg2.append(np.mean(X[X[:, 1] == i + 1980][:, 2]))
        print('avgggggg', avg)

        mm = np.mean(avg2)
        ss = np.std(avg2)

        avg = {}
        for i in range(39):
            avg[str(i + 1980)] = (np.mean(X[X[:, 1] == i + 1980][:, 2]) - mm) / ss
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

        validation_loss=[]
        train_loss=[]

        for i in range(Max_it):

            out_tr = get_sample(dic, A, avg, batch_size_tr, time_steps=5, num_features=316+66+14)
            # I = np.random.randint(m_tr, size=batch_size_tr)
            # Batch_X_g = X_training_g[I]
            Ybar_tr=out_tr[:, :, -1].reshape(-1,5,1)

            Batch_X_e = out_tr[:, :, 3:-1].reshape(-1, 6*52+66+14)
            Batch_X_e = np.expand_dims(Batch_X_e,axis=-1)
            Batch_Y = out_tr[:, -1, 2]
            Batch_Y = Batch_Y.reshape(len(Batch_Y), 1)
            Batch_Y_2 = out_tr[:, np.arange(0,4), 2]

            #I = np.random.randint(m_tr, size=batch_size_tr)
            #Batch_X_g = X_training_g[I]
            #Batch_X_e = np.expand_dims(out[:,3:],axis=-1)
            #Batch_Y = out[:,2]
            #Batch_Y = Batch_Y.reshape(len(Batch_Y), 1)

            if i == 60000:
                learning_rate = learning_rate / 2
                print('learningrate1',learning_rate)
            elif i == 120000:
                learning_rate = learning_rate / 2
                print('learningrate2', learning_rate)
            elif i == 180000:
                learning_rate = learning_rate / 2
                print('learningrate3', learning_rate)


            # Run training operation
            sess.run(train_op, feed_dict={ E_t1: Batch_X_e[:,0:52,:], E_t2: Batch_X_e[:,52*1:2*52,:], E_t3: Batch_X_e[:,52*2:3*52,:],
                                           E_t4: Batch_X_e[:, 52 * 3:4 * 52, :], E_t5: Batch_X_e[:,52*4:5*52,:], E_t6: Batch_X_e[:,52*5:52*6,:],
                                           S_t1: Batch_X_e[:, 312:318, :], S_t2: Batch_X_e[:, 318:324, :], S_t3: Batch_X_e[:, 324:330, :],
                                           S_t4: Batch_X_e[:, 330:336, :], S_t5: Batch_X_e[:, 336:342, :], S_t6: Batch_X_e[:, 342:348, :],
                                           S_t7: Batch_X_e[:, 348:354, :], S_t8: Batch_X_e[:, 354:360, :], S_t9: Batch_X_e[:, 360:366, :], 
                                           S_t10: Batch_X_e[:, 366:372, :], S_t11: Batch_X_e[:, 372:378 , :], P_t: Batch_X_e[:, 378:392, :], 
                                           Ybar:Ybar_tr, Y_t: Batch_Y, Y_t_2: Batch_Y_2})

            if i % 1000 == 0:
                out_tr = get_sample(dic, A, avg, batch_size=800, time_steps=5, num_features= 312 + 66 + 14)
                # I = np.random.randint(m_tr, size=batch_size_tr)
                # Batch_X_g = X_training_g[I]
                Ybar_tr = out_tr[:, :, -1].reshape(-1, 5, 1)
                
                Batch_X_e = out_tr[:, :, 3:-1].reshape(-1, 312 + 66 + 14)
                Batch_X_e = np.expand_dims(Batch_X_e, axis=-1)
                Batch_Y = out_tr[:, -1, 2]
                Batch_Y = Batch_Y.reshape(len(Batch_Y), 1)
                Batch_Y_2 = out_tr[:, np.arange(0, 4), 2]

                out_te = get_sample_te(dic, mean_last, avg,np.sum(Index), time_steps=5, num_features= 312 + 66 + 14)
                print(out_te.shape)

                Ybar_te = out_te[:, :, -1].reshape(-1, 5, 1)
                Batch_X_e_te = out_te[:, :, 3:-1].reshape(-1, 312 + 66 + 14)
                Batch_X_e_te = np.expand_dims(Batch_X_e_te, axis=-1)
                Batch_Y_te = out_te[:, -1, 2]
                Batch_Y_te = Batch_Y_te.reshape(len(Batch_Y_te), 1)
                Batch_Y_te2 = out_te[:, np.arange(0,4), 2]

                rmse_tr,yhat1_tr,loss_tr = sess.run([RMSE,Yhat1,Tloss],
                                                    feed_dict={ E_t1: Batch_X_e[:,0:52,:], E_t2: Batch_X_e[:,52*1:2*52,:], E_t3: Batch_X_e[:,52*2:3*52,:],
                                                               E_t4: Batch_X_e[:, 52 * 3:4 * 52, :], E_t5: Batch_X_e[:,52*4:5*52,:], E_t6: Batch_X_e[:,52*5:52*6,:],
                                                               S_t1: Batch_X_e[:, 312:318, :], S_t2: Batch_X_e[:, 318:324, :], S_t3: Batch_X_e[:, 324:330, :],
                                                               S_t4: Batch_X_e[:, 330:336, :], S_t5: Batch_X_e[:, 336:342, :], S_t6: Batch_X_e[:, 342:348, :],
                                                               S_t7: Batch_X_e[:, 348:354, :], S_t8: Batch_X_e[:, 354:360, :], S_t9: Batch_X_e[:, 360:366, :], 
                                                               S_t10: Batch_X_e[:, 366:372, :], S_t11: Batch_X_e[:, 372:378 , :], P_t: Batch_X_e[:, 378:392, :], 
                                                               Ybar:Ybar_tr, Y_t: Batch_Y, Y_t_2: Batch_Y_2})

                rc_tr = np.corrcoef(np.squeeze(Batch_Y), np.squeeze(yhat1_tr))[0, 1]


                rmse_te,yhat1_te,loss_val = sess.run([RMSE,Yhat1,Tloss], 
                                                     feed_dict={ E_t1: Batch_X_e[:,0:52,:], E_t2: Batch_X_e[:,52*1:2*52,:], E_t3: Batch_X_e[:,52*2:3*52,:],
                                                                E_t4: Batch_X_e[:, 52 * 3:4 * 52, :], E_t5: Batch_X_e[:,52*4:5*52,:], E_t6: Batch_X_e[:,52*5:52*6,:],
                                                                S_t1: Batch_X_e[:, 312:318, :], S_t2: Batch_X_e[:, 318:324, :], S_t3: Batch_X_e[:, 324:330, :],
                                                                S_t4: Batch_X_e[:, 330:336, :], S_t5: Batch_X_e[:, 336:342, :], S_t6: Batch_X_e[:, 342:348, :],
                                                                S_t7: Batch_X_e[:, 348:354, :], S_t8: Batch_X_e[:, 354:360, :], S_t9: Batch_X_e[:, 360:366, :], 
                                                                S_t10: Batch_X_e[:, 366:372, :], S_t11: Batch_X_e[:, 372:378 , :], P_t: Batch_X_e[:, 378:392, :], 
                                                                Ybar:Ybar_tr, Y_t: Batch_Y, Y_t_2: Batch_Y_2})
                
                rc = np.corrcoef(np.squeeze(Batch_Y_te), np.squeeze(yhat1_te))[0, 1]


                print("Iteration %d , The training RMSE is %f and Cor train is %f  and test RMSE is %f and Cor is %f " % (i, rmse_tr, rc_tr, rmse_te, rc))

                validation_loss.append(loss_val)
                train_loss.append(loss_tr)
                print(loss_tr, loss_val)

    out_te = get_sample_te(dic, mean_last, avg,np.sum(Index), time_steps=5, num_features=316+100+14+4)

    Batch_X_e_te = out_te[:, :, 3:-1].reshape(-1, 312 + 66 + 14)
    Ybar_te = out_te[:, :, -1].reshape(-1, 5, 1)
    Batch_X_e_te = np.expand_dims(Batch_X_e_te, axis=-1)
    Batch_Y_te = out_te[:, -1, 2]
    Batch_Y_te = Batch_Y_te.reshape(len(Batch_Y_te), 1)
    Batch_Y_te2 = out_te[:, np.arange(0, 4), 2]

    rmse_te, yhat1 = sess.run([RMSE,Yhat1], feed_dict={ E_t1: Batch_X_e[:,0:52,:], E_t2: Batch_X_e[:,52*1:2*52,:], E_t3: Batch_X_e[:,52*2:3*52,:],
                                                        E_t4: Batch_X_e[:, 52 * 3:4 * 52, :], E_t5: Batch_X_e[:,52*4:5*52,:], E_t6: Batch_X_e[:,52*5:52*6,:],
                                                        S_t1: Batch_X_e[:, 312:318, :], S_t2: Batch_X_e[:, 318:324, :], S_t3: Batch_X_e[:, 324:330, :],
                                                        S_t4: Batch_X_e[:, 330:336, :], S_t5: Batch_X_e[:, 336:342, :], S_t6: Batch_X_e[:, 342:348, :],
                                                        S_t7: Batch_X_e[:, 348:354, :], S_t8: Batch_X_e[:, 354:360, :], S_t9: Batch_X_e[:, 360:366, :], 
                                                        S_t10: Batch_X_e[:, 366:372, :], S_t11: Batch_X_e[:, 372:378 , :], P_t: Batch_X_e[:, 378:392, :], 
                                                        Ybar:Ybar_tr, Y_t: Batch_Y, Y_t_2: Batch_Y_2})
                
    print("The training RMSE is %f  and test RMSE is %f " % (rmse_tr, rmse_te))
    t2=time.time()

    print('the training time was %f' % (round(t2-t1, 2)))
    tf.keras.Model.save('./model_soybean')  # Saving the model

    return  rmse_tr,rmse_te,train_loss,validation_loss



BigX = np.load('data/soybean_data_compressed.npz') ##order W(52*6) S(100) P(14) S_extra(4)
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

Max_it=150000      #150000 could also be used with early stopping
learning_rate=0.0003   # Learning rate
dropout=0.0  # Dropout rate
batch_size_tr=25  # traning batch size
le=0.0  # Weight of loss for prediction using times before final time steps
l=1.0    # Weight of loss for prediction using final time step
num_units=64  # Number of hidden units for LSTM celss
num_layers=1  # Number of layers of LSTM cell

rmse_tr,rmse_te,train_loss,validation_loss = main_program(X, Index,num_units,num_layers,Max_it, learning_rate, batch_size_tr,le,l, is_training=True, dropout=dropout)