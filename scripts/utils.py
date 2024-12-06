import tensorflow as tf
from tensorflow.keras.losses import Huber

def cost_function(alpha, beta):
    def loss(y_true, y_pred):       
        huber = Huber(delta=5.0)
        weights = tf.concat([beta * tf.ones_like(y_true[:, :-1]), alpha * tf.ones_like(y_true[:, -1:])], axis=-1)
        per_timestep_loss = huber(y_true, y_pred)
        weighted_loss = tf.reduce_mean(weights * per_timestep_loss)
        return weighted_loss    
    return loss

def scheduler(epoch, lr):
        if epoch == 60:
            return lr / 2
        elif epoch == 120:
            return lr / 2
        elif epoch == 180:
            return lr / 2
        else:
            return lr
        
def total_parameters(model):
    total_parameters = 0
    for variable in model.trainable_weights:
        shape = variable.shape
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim
        total_parameters += variable_parameters
    return total_parameters