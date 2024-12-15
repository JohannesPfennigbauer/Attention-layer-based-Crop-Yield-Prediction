import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Flatten, Dense, Conv1D, AveragePooling1D, Concatenate, Reshape, LSTM, Attention, MultiHeadAttention
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, LearningRateScheduler, EarlyStopping

class baseModel:
    """
    Base Model class for both the original and attention-based models
    """
    def __init__(self, learning_rate, time_steps):
        self.model = None
        self.history = None
        self.learning_rate = learning_rate
        self.time_steps = time_steps
    
    def conv_W(self, input_shape):
        """
        Convolutional Neural Network for weather data, input shape (52, 1), output shape (11,)
        """
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

    def conv_S(self, input_shape):
        """
        Convolutional Neural Network for soil data, input shape (6, 1), output shape (4,)
        """
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
    
    def scheduler(self, epoch, learning_rate):
        """
        Learning rate scheduler for the model training. Halves the learning rate at epochs 60, 120, and 180.
        """
        if epoch == 60:
            return learning_rate / 2
        elif epoch == 120:
            return learning_rate / 2
        elif epoch == 180:
            return learning_rate / 2
        else:
            return learning_rate
    
    def total_parameters(self):
        """
        Calculate the total number of trainable parameters in the model.
        """
        total_parameters = 0
        for variable in self.model.trainable_weights:
            shape = variable.shape
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim
            total_parameters += variable_parameters
        print(f"Total parameters: {total_parameters}")
        return total_parameters
    
   
    def compile(self):
        """
        Compiles the instantiated model with Adam optimizer and Huber loss function.
        Huber loss is used to reduce the impact of outliers in the training data.
        """
        self.model = self.full_model() 
        self.model.compile(optimizer = Adam(learning_rate = self.learning_rate), loss = Huber(delta=5.0))
        self.total_parameters()
        return self.model
    
    def fit(self, X_train, y_train, X_val, y_val, epochs):
        """
        Fits the model on the training data and validates on the validation data. Uses Early Stopping on validation loss and Learning Rate Scheduler.
        """
        lr_scheduler = LearningRateScheduler(self.scheduler)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, min_delta=0.001)
        self.history = self.model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, callbacks=[lr_scheduler, early_stopping])
        return self.model
    
    def fit_wrapper(self, X_train, y_train, X_val, y_val, epochs):
        """
        Wrapper function to handle the input data, format it as dictionary and fit the model.
        """
        X_train = np.expand_dims(X_train, axis=-1)
        X_train_in = {f'w{i}': X_train[:, 52*i:52*(i+1), :] for i in range(6)}
        X_train_in.update({f's{i}': X_train[:, 312+6*i:312+6*(i+1), :] for i in range(11)})
        X_train_in['p'] = X_train[:, 378:392, :]
        X_train_in['avg_yield'] = X_train[:, -self.time_steps:, :]
        
        X_val = np.expand_dims(X_val, axis=-1)
        X_val_in = {f'w{i}': X_val[:, 52*i:52*(i+1), :] for i in range(6)}
        X_val_in.update({f's{i}': X_val[:, 312+6*i:312+6*(i+1), :] for i in range(11)})
        X_val_in['p'] = X_val[:, 378:392, :]
        X_val_in['avg_yield'] = X_val[:, -self.time_steps:, :]
        
        return self.fit(X_train_in, y_train, X_val_in, y_val, epochs)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_wrapper(self, X):
        """
        Wrapper function to handle the input data, format it as dictionary and make predictions.
        """
        X = np.expand_dims(X, axis=-1)
        X_in = {f'w{i}': X[:, 52*i:52*(i+1), :] for i in range(6)}
        X_in.update({f's{i}': X[:, 312+6*i:312+6*(i+1), :] for i in range(11)})
        X_in['p'] = X[:, 378:392, :]
        X_in['avg_yield'] = X[:, -self.time_steps:, :]
        
        return self.predict(X_in)
    
    def plot_training_history(self):
        """
        Visualize the training and validation loss over epochs.
        """
        if self.history is None:
            raise ValueError("No training history found. Train the model first.")
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.history.history['loss'], label='Training Loss', color='#606c38')
        plt.plot(self.history.history['val_loss'], label='Validation Loss', color='#bc6c25')
        plt.title('Training and Validation Loss over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.gca().set_facecolor('#fefae0')
        plt.show()
    


class OgModel(baseModel):
    def __init__(self, learning_rate, time_steps, num_units, num_layers, dropout):
        super().__init__(learning_rate, time_steps)
        self.num_units = num_units
        self.num_layers = num_layers
        self.dropout = dropout

    def full_model(self):
        """
        Full original model architecture with CNN for weather and soil data, LSTM for yield data, and Dense layers for static features.
        """
        print("\n--- Model Architecture ---")
        print(" - CNN for Weather data - ")
        w_inputs = {f'w{i}': Input(shape=(52, 1), dtype=tf.float32, name=f'w{i}') for i in range(6)}
        print("Input W-CNN: 6x", w_inputs['w0'].shape)
        w_out = [self.conv_W((52,1))(w) for w in w_inputs.values()]
        w_out = Concatenate()(w_out)
        print("Output W-CNN:", w_out.shape)
        
        print(" - CNN for Soil data - ")
        s_inputs = {f's{i}': Input(shape=(6, 1), dtype=tf.float32, name=f's{i}') for i in range(11)}
        print("Input S-CNN: 11x", s_inputs['s0'].shape)
        s_out = [self.conv_S((6,1))(s) for s in s_inputs.values()]
        s_out = Concatenate()(s_out)
        print("Output S-CNN:", s_out.shape)
        
        print(" - Concatenate weather, soil and management data - ")
        m_input = Input(shape=(14, 1), dtype=tf.float32, name='p')
        static_features = Concatenate()([w_out, s_out, Reshape((14,))(m_input)])
        print("W+S+M concatenated:", static_features.shape)
        static_features = Dense(64, activation='relu', kernel_regularizer=l2(0.04), name='final_CNN_layer')(static_features)
        print("W+S+M after Dense:", static_features.shape)
        
        print(" - LSTM for yield data - ")
        avg_yield_input = Input(shape=(self.time_steps, 1), dtype=tf.float32, name='avg_yield')
        x = avg_yield_input
        print("LSTM input:", x.shape)
        for _ in range(self.num_layers):
            x = LSTM(self.num_units, return_sequences=True, dropout=self.dropout)(x)
        x = LSTM(self.num_units, return_sequences=False, dropout=self.dropout)(x)
        print("LSTM output:", x.shape)
        
        print(" - Combine static and dynamic features - ")
        combined = Concatenate()([x, static_features])
        print("Combined:", combined.shape)
        combined = Dense(16, activation='relu')(combined)
        print("Combined Dense:", combined.shape)
        output = Dense(1, activation=None, name='yield')(combined)
        print("Output:", output.shape, "\n")

        inputs = {**w_inputs, **s_inputs, 'p': m_input, 'avg_yield': avg_yield_input}
        model = Model(inputs=inputs, outputs=output)
        return model
    

    
class AttModel(baseModel):
    def __init__(self, learning_rate, time_steps, num_units, num_layers, dropout, num_heads, key_dim, multi_head=False):
        super().__init__(learning_rate, time_steps)
        self.num_units = num_units
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.multi_head = multi_head
        
    def conv_W(self, input_shape):
        """
        Convolutional Neural Network for weather data, input shape (52, 1), output shape (11,), with additional attention layer after the first convolution.
        """
        inputs = Input(shape=input_shape)

        # Layer 1: Extract initial features from attended output
        X = Conv1D(filters=8, kernel_size=5, strides=1, padding='same',
                kernel_initializer=GlorotUniform(), activation='relu')(inputs)
        
        # Simple attention layer to guide models focus on specific weeks
        query = Dense(12)(X)
        value = Dense(12)(X)
        attention_output = Attention()([query, value])    
        X = AveragePooling1D(pool_size=2, strides=2, padding='same')(attention_output)

        # Layer 2: Further feature extraction
        X = Conv1D(filters=12, kernel_size=3, strides=1, padding='same',
                kernel_initializer=GlorotUniform(), activation='relu')(X)
        X = AveragePooling1D(pool_size=2, strides=2, padding='same')(X)

        # Layer 3: Reduce feature map size
        X = Conv1D(filters=16, kernel_size=3, strides=2, padding='same',
                kernel_initializer=GlorotUniform(), activation='relu')(X)

        X = Flatten()(X)
        X = Dense(16, activation='relu', kernel_initializer=GlorotUniform(), kernel_regularizer=l2(0.004))(X)
        X = Dense(11, activation='linear', kernel_initializer=GlorotUniform())(X)

        # Model
        model = Model(inputs=inputs, outputs=X)
        return model


    def full_model(self):
        """
        The full original model architecture with CNN for weather and soil data, LSTM for yield data, and Dense layers for static features, plus an additional Multi-Head Attention layer for weather data. It also uses CNN with attention for weather data.
        """
        print("\n--- Model Architecture ---")
        print(" - CNN for Weather data - ")
        w_inputs = {f'w{i}': Input(shape=(52, 1), dtype=tf.float32, name=f'w{i}') for i in range(6)}
        print("Input W-CNN: 6x", w_inputs['w0'].shape)
        w_out = [self.conv_W((52,1))(w) for w in w_inputs.values()]
        w_out = Concatenate()(w_out)
        print("Output W-CNN:", w_out.shape)
        
        if self.multi_head:
            print(" - Multi-Head Attention for Weather data - ")
            multi_head_attention = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.key_dim, name='multi_head_attention')
            w_out = Reshape((w_out.shape[1], 1))(w_out)
            w_out = multi_head_attention(w_out, w_out)
            w_out = Flatten()(w_out)
            print("Output W-Attention:", w_out.shape)
        
        print(" - CNN for Soil data - ")
        s_inputs = {f's{i}': Input(shape=(6, 1), dtype=tf.float32, name=f's{i}') for i in range(11)}
        print("Input S-CNN: 11x", s_inputs['s0'].shape)
        s_out = [self.conv_S((6,1))(s) for s in s_inputs.values()]
        s_out = Concatenate()(s_out)
        print("Output S-CNN:", s_out.shape)
         
        print(" - Concatenate weather, soil and management data - ")
        m_input = Input(shape=(14, 1), dtype=tf.float32, name='p')
        static_features = Concatenate()([w_out, s_out, Reshape((14,))(m_input)])
        print("W+S+M concatenated:", static_features.shape)
        static_features = Dense(64, activation='relu', kernel_regularizer=l2(0.04))(static_features)
        print("W+S+M after Dense:", static_features.shape)
        
        print(" - LSTM for yield data - ")
        avg_yield_input = Input(shape=(self.time_steps, 1), dtype=tf.float32, name='avg_yield')
        x = avg_yield_input
        print("LSTM input:", x.shape)
        for _ in range(self.num_layers):
            x = LSTM(self.num_units, return_sequences=True, dropout=self.dropout)(x)
        x = LSTM(self.num_units, return_sequences=False, dropout=self.dropout)(x)
        print("LSTM output:", x.shape)
        
        print(" - Combine static and dynamic features - ")
        combined = Concatenate()([Flatten()(x), static_features])
        print("Combined:", combined.shape)
        combined = Dense(16, activation='relu')(combined)
        print("Combined Dense:", combined.shape)
        output = Dense(1, activation=None, name='yield')(combined)
        print("Output:", output.shape, "\n")

        inputs = {**w_inputs, **s_inputs, 'p': m_input, 'avg_yield': avg_yield_input}
        model = Model(inputs=inputs, outputs=output)
        return model
    
    def fit(self, X_train, y_train, X_val, y_val, epochs):
        """
        A specialized fit method for the attention model that includes an additional callback to store attention weights.
        """
        lr_scheduler = LearningRateScheduler(self.scheduler)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, min_delta=0.001)
        if self.multi_head:
            attention_callback = AttentionWeightsCallback(self.model)
            self.history = self.model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, 
                                      callbacks=[lr_scheduler, early_stopping, attention_callback])
            return self.model, attention_callback
        else:
            self.history = self.model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, 
                                      callbacks=[lr_scheduler, early_stopping])
            return self.model
    
               

class AttentionWeightsCallback(Callback):
    """
    Specialized callback to store attention weights during training.
    """
    def __init__(self, model):
        super(AttentionWeightsCallback, self).__init__()
        self.model = model
        self.weights = []

    def on_epoch_end(self, epoch, logs=None):
        """
        Store attention weights at the end of each epoch. Overwrites the weights from the previous epoch.
        """
        attention_layer = self.model.get_layer('multi_head_attention')
        layer_weights = attention_layer.get_weights()
        self.weights = layer_weights
        
    def visualize_attention(self):
        """
        Visualisation of the attention weights for the final epoch of the training.
        """
        if not self.weights:
            raise ValueError("No weather attention scores found.")
        
        final_weights = self.weights[-1]
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(final_weights[0][0], cmap='viridis', ax=ax)
        ax.set_title('Multi-Head Attention Weights')
        ax.set_xlabel('Head')
        ax.set_ylabel('Week')
        plt.show()