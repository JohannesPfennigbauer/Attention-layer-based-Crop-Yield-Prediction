import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class EvaluateModel:
    def __init__(self, model, X_train, y_train, X_val, y_val, s=1, m=0):
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
    
    def rmse(self, y_true, y_pred):
        """Calculate Root Mean Squared Error."""
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    def mae(self, y_true, y_pred):
        """Calculate Mean Absolute Error."""
        return mean_absolute_error(y_true, y_pred)
    
    def r2(self, y_true, y_pred):
        """Calculate R² Score."""
        return r2_score(y_true, y_pred)


    def plot_actual_vs_predicted(self, y_train, y_train_p, y_val, y_val_p):
        """Plots actual values versus predicted for training and validation data."""
        
        plt.figure(figsize=(10, 6))
        plt.scatter(y_train, y_train_p, label='Training Data', alpha=0.6, color='orange')
        plt.scatter(y_val, y_val_p, label='Validation Data', alpha=0.6, color='blue')
        plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], color='red', linestyle='--')
        plt.title('Actual vs Predicted')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.legend()
        plt.show()
        

    def make_predictions(self):
        """Make predictions on training and validation data."""
        self.y_train = self.y_train.reshape(-1,1) * self.s + self.m
        self.y_val = self.y_val.reshape(-1,1) * self.s + self.m
        
        output_train = self.model.predict(self.X_train)
        output_val = self.model.predict(self.X_val)
        y_train_pred = output_train[0] if len(output_train) == 3 else output_train
        y_val_pred = output_val[0] if len(output_val) == 3 else output_val

        y_train_pred = y_train_pred.reshape(-1,1) * self.s + self.m
        y_val_pred = y_val_pred.reshape(-1,1) * self.s + self.m

        return y_train_pred, y_val_pred


    def evaluate(self):
        """Evaluate model performance."""
        y_train_pred, y_val_pred = self.make_predictions()
        
        print("Training RMSE:", self.rmse(self.y_train, y_train_pred))
        print("Validation RMSE:", self.rmse(self.y_val, y_val_pred), "\n")
        
        print("Training MAE:", self.mae(self.y_train, y_train_pred))
        print("Validation MAE:", self.mae(self.y_val, y_val_pred), "\n")
        
        print("Training R²:", self.r2(self.y_train, y_train_pred))
        print("Validation R²:", self.r2(self.y_val, y_val_pred))
        
        self.plot_actual_vs_predicted(self.y_train, y_train_pred, self.y_val, y_val_pred)

