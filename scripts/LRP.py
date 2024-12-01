import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv1D, Dense, LSTM, Flatten, Concatenate, AveragePooling1D

class LRPConv1D(Conv1D):
    def call(self, inputs):
        return super(LRPConv1D, self).call(inputs)

    def relevance_propagation(self, R, inputs):
        Z = self(inputs)
        R = tf.cast(R, dtype=Z.dtype)  # Ensure R has the same dtype as Z
        S = R / (Z + 1e-8)
        C = tf.gradients(Z, inputs, grad_ys=S)[0]
        return inputs * C

class LRPDense(Dense):
    def call(self, inputs):
        return super(LRPDense, self).call(inputs)

    def relevance_propagation(self, R, inputs):
        Z = self(inputs)
        R = tf.cast(R, dtype=Z.dtype)  # Ensure R has the same dtype as Z
        S = R / (Z + 1e-8)
        C = tf.gradients(Z, inputs, grad_ys=S)[0]
        return inputs * C

class LRPLSTM(LSTM):
    def build(self, input_shape):
        super(LRPLSTM, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return super(LRPLSTM, self).call(inputs, **kwargs)
    
    def compute_output_shape(self, input_shape):
        return super(LRPLSTM, self).compute_output_shape(input_shape)

    def relevance_propagation(self, R, inputs):
        Z = self(inputs)
        R = tf.cast(R, dtype=Z.dtype)  # Ensure R has the same dtype as Z
        S = R / (Z + 1e-8)
        C = tf.gradients(Z, inputs, grad_ys=S)[0]
        return inputs * C
    
class LRPFlatten(Flatten):
    def call(self, inputs):
        return super(LRPFlatten, self).call(inputs)

    def relevance_propagation(self, R, inputs):
        R = tf.cast(R, dtype=inputs.dtype)  # Ensure R has the same dtype as inputs
        return tf.reshape(R, tf.shape(inputs))

class LRPConcatenate(Concatenate):
    def call(self, inputs):
        return super(LRPConcatenate, self).call(inputs)

    def relevance_propagation(self, R, inputs):
        R = tf.cast(R, dtype=inputs[0].dtype)  # Ensure R has the same dtype as inputs
        split_sizes = [tf.shape(input)[-1] for input in inputs]
        return tf.split(R, split_sizes, axis=-1)

class LRPAveragePooling1D(AveragePooling1D):
    def call(self, inputs):
        return super(LRPAveragePooling1D, self).call(inputs)

    def relevance_propagation(self, R, inputs):
        Z = self(inputs)
        R = tf.cast(R, dtype=Z.dtype)  # Ensure R has the same dtype as Z
        S = R / (Z + 1e-8)
        C = tf.gradients(Z, inputs, grad_ys=S)[0]
        return inputs * C
