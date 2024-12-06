import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class EvaluateModel:
    def __init__(self, model, X_train, y_train, X_val, y_val, s=1, m=0, att=False):
        """
        Initialize with the model, training and validation data, and optional scaling factors.
        """
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.s = s
        self.m = m
        self.att = att

    def rmse(self, y_true, y_pred):
        """Calculate Root Mean Squared Error."""
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    def mae(self, y_true, y_pred):
        """Calculate Mean Absolute Error."""
        return mean_absolute_error(y_true, y_pred)
    
    def r2(self, y_true, y_pred):
        """Calculate R² Score."""
        return r2_score(y_true, y_pred)


    def plot_results(self, history, y_train, y_train_pred, y_val, y_val_pred):
        """Plot training/validation loss and actual vs predicted values side by side."""       
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Plot training and validation loss
        axes[0].plot(history.history['loss'], label='Training Loss')
        if 'val_loss' in history.history:
            axes[0].plot(history.history['val_loss'], label='Validation Loss')
        axes[0].set_title('Training and Validation Loss over Epochs')
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid()

        # Plot actual vs predicted
        axes[1].scatter(y_train, y_train_pred, label='Training Data', alpha=0.6, color='orange')
        axes[1].scatter(y_val, y_val_pred, label='Validation Data', alpha=0.6, color='blue')
        min_val = min(y_train.min(), y_val.min())
        max_val = max(y_train.max(), y_val.max())
        axes[1].plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
        axes[1].set_title('Actual vs Predicted')
        axes[1].set_xlabel('Actual')
        axes[1].set_ylabel('Predicted')
        axes[1].legend()
        axes[1].grid()

        plt.tight_layout()
        plt.show()


    def make_predictions(self):
        """Make predictions on training and validation data."""
        self.y_train = self.y_train.reshape(-1,1) * self.s + self.m
        self.y_val = self.y_val.reshape(-1,1) * self.s + self.m
        
        if self.att:
            y_train_pred, _, _ = self.model.predict(self.X_train)
            y_val_pred, _, _ = self.model.predict(self.X_val)
        else:
            y_train_pred = self.model.predict(self.X_train)
            y_val_pred = self.model.predict(self.X_val)
        
        y_train_pred = y_train_pred.reshape(-1,1) * self.s + self.m
        y_val_pred = y_val_pred.reshape(-1,1) * self.s + self.m

        return y_train_pred, y_val_pred


    def evaluate(self, history):
        """Evaluate model performance."""
        y_train_pred, y_val_pred = self.make_predictions()
        
        print("Training RMSE:", self.rmse(self.y_train, y_train_pred))
        print("Validation RMSE:", self.rmse(self.y_val, y_val_pred))
        
        print("Training MAE:", self.mae(self.y_train, y_train_pred))
        print("Validation MAE:", self.mae(self.y_val, y_val_pred))
        
        print("Training R²:", self.r2(self.y_train, y_train_pred))
        print("Validation R²:", self.r2(self.y_val, y_val_pred))
        
        self.plot_results(history, self.y_train, y_train_pred, self.y_val, y_val_pred)

        
    def visualize_attention(self, X):
        """Visualize attention weights."""
        _, att_weights_w, att_weights_s = self.model.predict(X)
        
        # Plot both attention weights next to each other
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        axes[0].imshow(att_weights_w, cmap='hot', interpolation='nearest')
        axes[0].set_title('Attention Weights Weather')
        axes[0].set_xlabel('Encoder Time Steps')
        axes[0].set_ylabel('Decoder Time Steps')
        axes[0].grid()

        axes[1].imshow(att_weights_s, cmap='hot', interpolation='nearest')
        axes[1].set_title('Attention Weights Soil')
        axes[1].set_xlabel('Encoder Time Steps')
        axes[1].set_ylabel('Decoder Time Steps')
        axes[1].grid()

        fig.colorbar(axes[0].images[0], ax=axes, orientation='vertical', fraction=.1)
        plt.show()
