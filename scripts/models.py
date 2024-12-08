import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.layers import Input, Flatten, Dense, Conv1D, AveragePooling1D, Concatenate, Reshape, LSTM, MultiHeadAttention
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping

class baseModel: 
    def __init__(self, learning_rate, alpha, beta):
        self.model = None
        self.history = None
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.beta = beta
    
    def conv_W(self, input_shape):
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
    
    def cost_function(self):
        def loss(y_true, y_pred):       
            huber = Huber(delta=5.0)
            weights = tf.concat([self.beta * tf.ones_like(y_true[:, :-1]), self.alpha * tf.ones_like(y_true[:, -1:])], axis=-1)
            per_timestep_loss = huber(y_true, y_pred)
            weighted_loss = tf.reduce_mean(weights * per_timestep_loss)
            return weighted_loss    
        return loss
    
    def scheduler(self, epoch, learning_rate):
        if epoch == 60:
            return learning_rate / 2
        elif epoch == 120:
            return learning_rate / 2
        elif epoch == 180:
            return learning_rate / 2
        else:
            return learning_rate
    
    def total_parameters(self):
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
        self.model = self.full_model()
        self.model.compile(optimizer = Adam(learning_rate = self.learning_rate), loss = self.cost_function())
        self.total_parameters()
        return self.model
    
    def fit(self, X_train, y_train, X_val, y_val, epochs):
        lr_scheduler = LearningRateScheduler(self.scheduler)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, min_delta=0.001)
        self.history = self.model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, callbacks=[lr_scheduler, early_stopping])
        return self.model
    
    def predict(self, X):
        return self.model.predict(X)
    
    def plot_training_history(self):
        if self.history is None:
            raise ValueError("No training history found. Train the model first.")
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
    


class OgModel(baseModel):
    def __init__(self, learning_rate, alpha, beta, time_steps, num_units, num_layers, dropout):
        super().__init__(learning_rate, alpha, beta)
        self.time_steps = time_steps
        self.num_units = num_units
        self.num_layers = num_layers
        self.dropout = dropout

    def full_model(self): 
        w_inputs = {f'w{i}': Input(shape=(52, 1), dtype=tf.float32, name=f'w{i}') for i in range(6)}
        w_out = [self.conv_W((52,1))(w) for w in w_inputs.values()]
        w_out = Concatenate()(w_out)
        print("\n--- Model Architecture ---")
        print(" - CNN for Weather data - ")
        print("Output W-CNN:", w_out.shape)
        
        print(" - CNN for Soil data - ")
        s_inputs = {f's{i}': Input(shape=(6, 1), dtype=tf.float32, name=f's{i}') for i in range(11)}
        s_out = [self.conv_S((6,1))(s) for s in s_inputs.values()]
        s_out = Concatenate()(s_out)
        print("Output S-CNN:", s_out.shape)
        
        print(" - Concatenate weather, soil and management data - ")
        m_input = Input(shape=(14, 1), dtype=tf.float32, name='p')
        static_features = Concatenate()([w_out, s_out, Reshape((14,))(m_input)])
        print("W+S+M concatenated:", static_features.shape)
        static_features = Dense(64, activation='relu', kernel_regularizer=l2(0.01), name='final_CNN_layer')(static_features)
        print("W+S+M after Dense:", static_features.shape)
        
        print(" - LSTM for yield data - ")
        avg_yield_input = Input(shape=(self.time_steps, 1), dtype=tf.float32, name='avg_yield')
        x = avg_yield_input
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
    def __init__(self, learning_rate, alpha, beta, time_steps, num_units, num_layers, dropout, num_heads, key_dim):
        super().__init__(learning_rate, alpha, beta)
        self.time_steps = time_steps
        self.num_units = num_units
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_heads = num_heads
        self.key_dim = key_dim

    def full_model(self):
        multi_head_attention = MultiHeadAttention(self.num_heads, self.key_dim)
        
        print("\n--- Model Architecture ---")
        print(" - CNN for Weather data - ")
        w_inputs = {f'w{i}': Input(shape=(52, 1), dtype=tf.float32, name=f'w{i}') for i in range(6)}
        print("Input:", w_inputs['w0'].shape)
        w_out = []
        w_attention_scores = []

        for key, w in w_inputs.items():
            conv_output = self.conv_W((52, 1))(w)
            reshaped_output = Reshape((conv_output.shape[1], 1))(conv_output)
            attended_output, attention_scores = multi_head_attention(reshaped_output, reshaped_output, return_attention_scores=True)
            
            w_out.append(Flatten()(attended_output))
            w_attention_scores.append(attention_scores)

        w_out = Concatenate()(w_out)
        print("Output:", w_out.shape)
        w_out = Flatten()(w_out)
        
        print(" - CNN for Soil data - ")
        s_inputs = {f's{i}': Input(shape=(6, 1), dtype=tf.float32, name=f's{i}') for i in range(11)}
        print("Input:", s_inputs['s1'].shape)
        s_out = []
        s_attention_scores = []
        
        for key, s in s_inputs.items():
            conv_output = self.conv_S((6, 1))(s)
            reshaped_output = Reshape((conv_output.shape[1], 1))(conv_output)
            attended_output, attention_scores = multi_head_attention(reshaped_output, reshaped_output, return_attention_scores=True)
            
            s_out.append(Flatten()(attended_output))
            s_attention_scores.append(attention_scores)
        
        s_out = Concatenate()(s_out)
        print("Output:", s_out.shape)
        s_out = Flatten()(s_out)
         
        print(" - Concatenate weather, soil and management data - ")
        m_input = Input(shape=(14, 1), dtype=tf.float32, name='p')
        static_features = Concatenate()([w_out, s_out, Reshape((14,))(m_input)])
        print("W+S+M concatenated:", static_features.shape)
        static_features = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(static_features)
        print("W+S+M after Dense:", static_features.shape)
        
        print(" - LSTM for yield data - ")
        avg_yield_input = Input(shape=(self.time_steps, 1), dtype=tf.float32, name='avg_yield')
        x = avg_yield_input
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
        model = Model(inputs=inputs, outputs=[output, w_attention_scores, s_attention_scores])
        return model
    
    def visualize_attention(self, X, type='weather'):
        outputs = self.model.predict(X)
        if len(outputs) < 2:
            raise ValueError("Model is not configured to output attention scores.")

        predictions, w_attention_scores, s_attention_scores = outputs

        if type == 'weather':
            if w_attention_scores is None:
                raise ValueError("No weather attention scores found.")
            for i, scores in enumerate(w_attention_scores):
                avg_scores = tf.reduce_mean(scores, axis=0)  # Average over observations
                avg_scores = tf.reduce_mean(avg_scores, axis=0) # Average over attention heads
                sns.heatmap(avg_scores.numpy(), annot=False, cmap="viridis")
                plt.title(f"Attention Scores for Weather Input w{i}")
                plt.xlabel("Key Positions")
                plt.ylabel("Query Positions")
                plt.show()
        elif type == 'soil':
            if s_attention_scores is None:
                raise ValueError("No soil attention scores found.")
            for i, scores in enumerate(s_attention_scores):
                avg_scores = tf.reduce_mean(scores, axis=0)  # Average over attention heads
                sns.heatmap(avg_scores.numpy(), annot=False, cmap="viridis")
                plt.title(f"Attention Scores for Soil Input w{i}")
                plt.xlabel("Key Positions")
                plt.ylabel("Query Positions")
                plt.show()
        else:
            raise ValueError("Invalid type. Choose 'weather' or 'soil'.")

            


class ModelWithLRP(baseModel):
    def __init__(self, learning_rate, alpha, beta, time_steps, num_units, num_layers, dropout):
        super().__init__(learning_rate, alpha, beta)
        self.time_steps = time_steps
        self.num_units = num_units
        self.num_layers = num_layers
        self.dropout = dropout
        
    def conv_W(self, input_shape):
        inputs = Input(shape=input_shape)

        # Layer 1: Extract initial features
        X = Conv1D(filters=8, kernel_size=5, strides=1, padding='same',
                   kernel_initializer=GlorotUniform(), activation=None)(inputs)
        X = AveragePooling1D(pool_size=2, strides=2, padding='same')(X)
        
        # Layer 2: Further feature extraction
        X = Conv1D(filters=12, kernel_size=3, strides=1, padding='same',
                   kernel_initializer=GlorotUniform(), activation=None)(X)
        X = AveragePooling1D(pool_size=2, strides=2, padding='same')(X)
        
        # Layer 3: Reduce feature map size
        X = Conv1D(filters=16, kernel_size=3, strides=2, padding='same',
                   kernel_initializer=GlorotUniform(), activation=None)(X)

        X = Flatten()(X)
        X = Dense(16, activation=None, kernel_initializer=GlorotUniform(), kernel_regularizer=l2(0.001))(X)
        X = Dense(11, activation=None, kernel_initializer=GlorotUniform())(X)
        
        model = Model(inputs=inputs, outputs=X)
        return model

    def conv_S(self, input_shape):
        inputs = Input(shape=input_shape)
        
        # Layer 1: Reduce temporal dimension with convolution
        X = Conv1D(filters=4, kernel_size=2, strides=1, padding='same',
                   kernel_initializer=GlorotUniform(), activation=None)(inputs)
        X = AveragePooling1D(pool_size=2, strides=2, padding='same')(X)

        # Layer 2: Additional convolution for feature extraction
        X = Conv1D(filters=8, kernel_size=2, strides=1, padding='same',
                   kernel_initializer=GlorotUniform(), activation=None)(X)

        X = Flatten()(X)
        X = Dense(4, activation=None, kernel_initializer=GlorotUniform(), kernel_regularizer=l2(0.001))(X)
        
        model = Model(inputs=inputs, outputs=X)
        return model

    def full_model(self):
        w_inputs = {f'w{i}': Input(shape=(52, 1), dtype='float32', name=f'w{i}') for i in range(6)}
        w_out = [self.conv_W((52, 1))(w) for w in w_inputs.values()]
        w_out = Concatenate()(w_out)

        s_inputs = {f's{i}': Input(shape=(6, 1), dtype='float32', name=f's{i}') for i in range(11)}
        s_out = [self.conv_S((6, 1))(s) for s in s_inputs.values()]
        s_out = Concatenate()(s_out)

        m_input = Input(shape=(14, 1), dtype='float32', name='p')
        static_features = Concatenate()([w_out, s_out, Reshape((14,))(m_input)])
        static_features = Dense(64, activation=None, kernel_regularizer=l2(0.01), name="final_CNN_layer")(static_features)

        avg_yield_input = Input(shape=(self.time_steps, 1), dtype=tf.float32, name='avg_yield')
        x = avg_yield_input
        for _ in range(self.num_layers):
            x = LSTM(self.num_units, return_sequences=True, dropout=self.dropout)(x)
        x = LSTM(self.num_units, return_sequences=False, dropout=self.dropout)(x)

        combined = Concatenate()([x, static_features])
        combined = Dense(16, activation=None)(combined)
        output = Dense(1, activation=None, name='yield')(combined)

        inputs = {**w_inputs, **s_inputs, 'p': m_input, 'avg_yield': avg_yield_input}
        model = Model(inputs=inputs, outputs=output)
        return model

