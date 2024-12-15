import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lime.lime_tabular import LimeTabularExplainer

class EvaluateModel:
    def __init__(self, model, X_train, y_train, X_val, y_val, m, s):
        """
        Initialize with the model, training and validation data, and optional scaling factors.
        """
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.m = m
        self.s = s

    
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
        plt.scatter(y_train, y_train_p, label='Training Data', alpha=0.6, color='#606c38')
        plt.scatter(y_val, y_val_p, label='Validation Data', alpha=0.8, color='#bc6c25')
        plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], color='#283618', linestyle='--')
        plt.title('Actual vs Predicted')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.legend()
        plt.gca().set_facecolor('#fefae0')
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
    def __init__(self, model, X_train, M, S, time_steps):
        """
        Initialize with the input data and scaling factors.
        """
        self.model = model
        self.X_train = X_train
        self.M = M
        self.S = S
        self.time_steps = time_steps
        self.features, self.features_seasons_dict = self._initialize_feature_names(time_steps)
        self.explainer = self._initialize_explainer()
        self.feature_impact = None
        
        
    def _initialize_feature_names(self, time_steps):
        """
        Return feature names and dictionary with seasons for weather attributes.
        """
        w_vars = ['precipitation', 'solar-rad.', 'snow-water-equivalent', 'max temp', 'min temp', 'vapor pressure']
        w_names = [f"{var}_{j}" for var in w_vars for j in range(1, 53)]
        s_vars = ['bulk density', 'cation exchange', 'coarse fragments', 'clay', 'nitrogen', 'org. carbon density', 'org. carbon stock', 'pH', 'sand', 'silt', 'organic carbon']
        s_depths = ['0-5', '5-15', '15-30', '30-60', '60-100', '100-120']
        s_names = [f"{var}_({depth})" for var in s_vars for depth in s_depths]
        m_names = [f"M{i}" for i in range(1, 15)]
        y_names = [f"Y-{i}" for i in range(1, time_steps+1)]
        features = w_names + s_names + m_names + y_names
        assert len(features) == 392 + time_steps
        
        pattern = re.compile(r"_\d{1,2}")
        weather_features = [feature for feature in features if pattern.search(feature)]
        seasons_dict = {}
        for feature in weather_features:
            week_no = int(pattern.search(feature).group()[1:])
            if week_no < 11:
                seasons_dict[feature] = "Pre Planting"
            elif week_no < 16:
                seasons_dict[feature] = "Planting"
            elif week_no < 26:
                seasons_dict[feature] = "Early Growth"
            elif week_no < 36:
                seasons_dict[feature] = "Reproductive Growth"
            elif week_no < 46:
                seasons_dict[feature] = "Maturation/Harvest"
            else:
                seasons_dict[feature] = "Post Harvest"
        
        return features, seasons_dict
    
    
    def _initialize_explainer(self):
        """
        Initialize the explainer with the model.
        """
        X_train_rescaled = self.X_train #* self.S + self.M
        explainer = LimeTabularExplainer(
            X_train_rescaled,
            mode="regression",
            feature_names=self.features,
            verbose=True,
            random_state=42
        )
        self.explainer = explainer
        return explainer
        
        
    def explain_observation(self, X, index, top=None, show_table=True):
        """
        Explain a single observation.
        """
        if top is not None:
            explanation = self.explainer.explain_instance(
                X[index],
                self.model.predict_wrapper,
                num_features=top
            )
        else:
            explanation = self.explainer.explain_instance(
                X[index],
                self.model.predict_wrapper
            )
        feature_values = {feature: abs(value) for feature, value in explanation.as_list()}
        feature_values = sorted(feature_values.items(), key=lambda x: x[1], reverse=True)
        feature_values = feature_values[:top] if top is not None else feature_values
        
        if show_table:
            explanation.show_in_notebook(show_table=True)
        
        return feature_values
    
    
    def explain_many_observations(self, X, indices, top=None):
        """
        Explain multiple observations and sum the weights of the top features.
        """
        explanations = []
        for index in indices:
            explanation_dic = self.explain_observation(X, index, top=top, show_table=False)
            explanations.append(explanation_dic)
        
        feature_impact = {}
        for explanation_dic in explanations:
            for feature, weight in explanation_dic:
                if feature in feature_impact:
                    feature_impact[feature] += abs(weight)
                else:
                    feature_impact[feature] = abs(weight)
        
        self.feature_impact = sorted(feature_impact.items(), key=lambda x: x[1], reverse=True)
        
    
    def feature_importance_table(self, model_name):
        """
        Returns a table with the feature importance scores for weather attributes
        """
        feature_importance_table = pd.DataFrame(columns=["Model", "Feature", "Impact", "Season"])
        pattern = re.compile(r"_\d{1,2}")
        
        for feature, impact in self.feature_impact:
            if pattern.search(feature):
                attribute = feature.split("<")[1].strip() if feature.count("<") == 2 \
                    else feature.split(">")[1].strip() if feature.count(">") == 2 \
                        else re.split(r"[<>=]", feature)[0].strip()
                row = {"Model": model_name, "Feature": attribute, "Impact": impact, "Season": self.features_seasons_dict[attribute]}
                feature_importance_table.loc[len(feature_importance_table)+1] = row
        
        return feature_importance_table

def compare_feature_impact(og_w_impact, att_w_impact):
    """
    Plot feature importance for weather attributes.
    """
    combined_w_impact = pd.concat([og_w_impact, att_w_impact], axis=0)
    m = combined_w_impact["Impact"].max()
    merged_w_impact = pd.merge(og_w_impact, att_w_impact, on=["Feature", "Season"], suffixes=("_og", "_att"))
    palette1 = {"original": "#606c38", "attention": "#bc6c25"}
    palette2 = {"Pre Planting": "#A9AAA9", "Planting": "#BBE387", "Early Growth": "#606c38", "Reproductive Growth": "#283618", "Maturation/Harvest": "#dda15e", "Post Harvest": "#bc6c25"}
    plot_data = combined_w_impact.groupby(["Model", "Season"]).sum("Impact").reset_index()
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    sns.barplot(ax=axes[0], x="Season", y="Impact", hue="Model", data=plot_data, palette=palette1, order=palette2.keys())
    axes[0].set_title("Sum of Feature Impact per Model and Season")
    axes[0].legend(title="Model", loc='upper right')
    for tick in axes[0].get_xticklabels():
        tick.set_rotation(45)

    sns.scatterplot(ax=axes[1], x="Impact_og", y="Impact_att", hue="Season", data=merged_w_impact, palette=palette2, hue_order=palette2.keys())
    axes[1].plot([0, m], [0, m], color="grey")
    axes[1].set_title("Feature Impact Comparison between Original and Attention Model")
    axes[1].set_xlabel("Original Model")
    axes[1].set_xlim(0, m)
    axes[1].set_ylabel("Attention Model")
    axes[1].set_ylim(0, m)
    axes[1].legend(title="Season", loc='upper left')
    plt.tight_layout()
    axes[0].set_facecolor('#fefae0')
    axes[1].set_facecolor('#fefae0')
    plt.show()