import tensorflow as tf
from tensorflow.keras.layers import Input, Flatten, Dense, Conv1D, AveragePooling1D, Concatenate, Reshape, LSTM, MultiHeadAttention
from tensorflow.keras.initializers import GlorotUniform
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


def attention_model(time_steps, num_units, num_layers, dropout, num_heads, key_dim):
    
    multi_head_attention = MultiHeadAttention(num_heads, key_dim)
    
    w_inputs = {f'w{i}': Input(shape=(52, 1), dtype=tf.float32, name=f'w{i}') for i in range(6)}
    w_out = [conv_W((52,1))(w) for w in w_inputs.values()]
    w_out = Concatenate()(w_out)
    print("\n--- Model Architecture ---")
    print(" - CNN for Weather data - ")
    print("Output W-CNN:", w_out.shape)
    w_out = Reshape((w_out.shape[1], 1))(w_out)
    print(" - Multi-Head Attention for Weather data - ")
    w_out, w_attention = multi_head_attention(w_out, w_out, return_attention_scores=True)
    print("Output W-Attention:", w_out.shape)
    w_out = Flatten()(w_out)
    
    print(" - CNN for Soil data - ")
    s_inputs = {f's{i}': Input(shape=(6, 1), dtype=tf.float32, name=f's{i}') for i in range(11)}
    s_out = [conv_S((6,1))(s) for s in s_inputs.values()]
    s_out = Concatenate()(s_out)
    print("Output S-CNN:", s_out.shape)
    s_out = Reshape((s_out.shape[1], 1))(s_out)
    print(" - Multi-Head Attention for Soil data - ")
    s_out, s_attention = multi_head_attention(s_out, s_out, return_attention_scores=True)
    print("Output S-Attention:", s_out.shape)
    s_out = Flatten()(s_out)
    
    
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
    combined = Concatenate()([Flatten()(x), static_features])
    print("Combined:", combined.shape)
    combined = Dense(16, activation='relu')(combined)
    print("Combined Dense:", combined.shape)
    output = Dense(1, activation=None, name='yield')(combined)
    print("Output:", output.shape)

    inputs = {**w_inputs, **s_inputs, 'p': m_input, 'avg_yield': avg_yield_input}
    model = Model(inputs=inputs, outputs=[output, w_attention, s_attention])
    return model



