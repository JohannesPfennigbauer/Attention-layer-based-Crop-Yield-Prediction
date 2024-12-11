import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lime.lime_tabular import LimeTabularExplainer

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
        
        output_train = self.model.predict_wrapper(self.X_train)
        output_val = self.model.predict_wrapper(self.X_val)
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





class ExplainModel:
    def __init__(self, model, X_train, s, m, time_steps):
        """
        Initialize with the input data and scaling factors.
        """
        self.model = model
        self.X_train = X_train
        self.s = s
        self.m = m
        self.time_steps = time_steps
        self.features = self.feature_names(time_steps)
        self.explainer = self.initialize_explainer()
        
        
    def feature_names(self, time_steps):
        """
        Return feature names.
        """
        w_vars = ['precipitation', 'solar-rad.', 'snow-water-equivalent', 'max temp', 'min temp', 'vapor pressure']
        w_names = [f"{var}_{j}" for var in w_vars for j in range(1, 53)]
        s_vars = ['bulk density', 'cation exchange', 'coarse fragments', 'clay', 'nitrogen', 'org. carbon density', 'org. carbon stock', 'pH', 'sand', 'silt', 'organic carbon']
        s_depths = ['0-5', '5-15', '15-30', '30-60', '60-100', '100-120']
        s_names = [f"{var}_{depth}" for var in s_vars for depth in s_depths]
        m_names = [f"M{i}" for i in range(1, 15)]
        y_names = [f"Y-{i}" for i in range(1, time_steps+1)]
        features = w_names + s_names + m_names + y_names
        assert len(features) == 392 + time_steps
        return features
    
    
    def initialize_explainer(self):
        """
        Initialize the explainer with the model.
        """
        X_train_rescaled = self.X_train * self.s + self.m
        explainer = LimeTabularExplainer(
            X_train_rescaled,
            mode="regression",
            feature_names=self.features,
            verbose=True,
            random_state=42
        )
        self.explainer = explainer
        return explainer
        
        
    def explain_observation(self, X, index, top=10, show_table=True):
        """
        Explain a single observation.
        """
        explanation = self.explainer.explain_instance(
            X[index],
            self.model.predict_wrapper,
            num_features=top
        )
        top_list = explanation.as_list()
        
        if show_table:
            explanation.show_in_notebook(show_table=True)
        
        return top_list[:top]
    
    
    def explain_many_observations(self, X, indices, top=10):
        """
        Explain multiple observations and sum the weights of the top features.
        """
        explanations = []
        for index in indices:
            explanation = self.explain_observation(X, index, top=top, show_table=False)
            explanations.append(explanation)
        
        top_features = {}
        for explanation in explanations:
            for feature, weight in explanation:
                if feature in top_features:
                    top_features[feature] += weight
                else:
                    top_features[feature] = weight
        
        top_features = sorted(top_features.items(), key=lambda x: x[1], reverse=True)[:top]
        return top_features