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
        Initialize with the model, training and validation data, and rescaling factors.
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


    def plot_actual_vs_predicted(self, y_train, y_train_p, y_val, y_val_p, save=None):
        """
        Scatterplot of actual vs predicted values for training and validation data.
        Helps to visualize the model's performance.
        """
        
        plt.figure(figsize=(10, 6))
        plt.scatter(y_train, y_train_p, label='Training Data', alpha=0.6, color='#606c38')
        plt.scatter(y_val, y_val_p, label='Validation Data', alpha=0.8, color='#bc6c25')
        plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], color='#283618', linestyle='--')
        plt.title(f'{save} model: Actual vs Predicted')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.legend()
        plt.gca().set_facecolor('#fefae0')
        if save != None:
            plt.savefig(f"../assets/{save}_actual_vs_predicted.png")
        plt.show()
        

    def make_predictions(self):
        """
        Make predictions on training and validation data.
        Rescales the target values and predictions to their original scale and returns predictions.
        """
        self.y_train = self.y_train.reshape(-1,1) * self.s + self.m
        self.y_val = self.y_val.reshape(-1,1) * self.s + self.m
        
        output_train = self.model.predict_wrapper(self.X_train)
        output_val = self.model.predict_wrapper(self.X_val)
        y_train_pred = output_train[0] if len(output_train) == 3 else output_train
        y_val_pred = output_val[0] if len(output_val) == 3 else output_val

        y_train_pred = y_train_pred.reshape(-1,1) * self.s + self.m
        y_val_pred = y_val_pred.reshape(-1,1) * self.s + self.m

        return y_train_pred, y_val_pred


    def evaluate(self, save=None):
        """
        Evaluate model performance in terms of RMSE, MAE and R² for training and validation data.
        Also plots the actual vs predicted values.
        """
        y_train_pred, y_val_pred = self.make_predictions()
        
        print("Training RMSE:", self.rmse(self.y_train, y_train_pred))
        print("Validation RMSE:", self.rmse(self.y_val, y_val_pred), "\n")
        
        print("Training MAE:", self.mae(self.y_train, y_train_pred))
        print("Validation MAE:", self.mae(self.y_val, y_val_pred), "\n")
        
        print("Training R²:", self.r2(self.y_train, y_train_pred))
        print("Validation R²:", self.r2(self.y_val, y_val_pred))
        
        self.plot_actual_vs_predicted(self.y_train, y_train_pred, self.y_val, y_val_pred, save=save)





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
        
        self.features, self.feature_groups, self.feature_subgroups, self.w_feature_seasons = None, None, None, None
        self._initialize_feature_names(time_steps)
        
        self.explainer = self._initialize_explainer()
        self.feature_impact = None

        
        
    def _initialize_feature_names(self, time_steps):
        """
        Initialize feature names and categorizes them into groups, subgroups and growing seasons. The latter is only for weather features.
        """
        w_vars = ['precipitation', 'solar-rad.', 'snow-water-eq', 'max temp', 'min temp', 'vapor pressure']
        w_names = [f"{var}_{j}" for var in w_vars for j in range(1, 53)]
        s_vars = ['bulk density', 'cation exchange', 'coarse fragments', 'clay', 'nitrogen', 'org-c density', 'org. carbon stock', 'pH', 'sand', 'silt', 'organic carbon']
        s_depths = ['0-5', '5-15', '15-30', '30-60', '60-100', '100-120']
        s_names = [f"{var}_({depth})" for var in s_vars for depth in s_depths]
        m_names = [f"M{i}" for i in range(1, 15)]
        y_names = [f"Y-{i}" for i in range(1, time_steps+1)]
        self.features = w_names + s_names + m_names + y_names
        assert len(self.features) == 392 + time_steps
        
        self.feature_groups = {}
        self.feature_subgroups = {}
        self.w_feature_seasons = {}
        for feature in self.features:
            if feature.startswith("Y"):
                self.feature_groups[feature] = "yield"
                self.feature_subgroups[feature] = feature
            elif feature.startswith("M"):
                self.feature_groups[feature] = "management"
                self.feature_subgroups[feature] = feature
            else:
                subgroup = feature.split("_")[0]
                if subgroup in s_vars:
                    self.feature_groups[feature] = "soil"
                    self.feature_subgroups[feature] = subgroup
                elif subgroup in w_vars:
                    self.feature_groups[feature] = "weather"
                    self.feature_subgroups[feature] = subgroup
                    week_no = int(feature.split("_")[1])
                    if week_no < 11:
                        self.w_feature_seasons[feature] = "Pre Planting"
                    elif week_no < 16:
                        self.w_feature_seasons[feature] = "Planting"
                    elif week_no < 26:
                        self.w_feature_seasons[feature] = "Early Growth"
                    elif week_no < 36:
                        self.w_feature_seasons[feature] = "Reproductive Growth"
                    elif week_no < 46:
                        self.w_feature_seasons[feature] = "Maturation/Harvest"
                    else:
                        self.w_feature_seasons[feature] = "Post Harvest"                     
                else:
                    raise ValueError(f"Invalid feature: {feature}")
    
    
    def _initialize_explainer(self):
        """
        Initialize the explainer with the model's training data.
        """
        explainer = LimeTabularExplainer(
            self.X_train,
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
        
    
    def get_feature_importance_table(self, model_name):
        """
        Returns a table with the feature importance scores (as a percentage of total), group, subgroup and growing season for all attributes.
        """
        feature_importance_table = pd.DataFrame(columns=["Model", "Group", "Subgroup", "Season", "Feature", "Impact"])

        for condition, impact in self.feature_impact:
            feature = condition.split("<")[1].strip() if condition.count("<") == 2 \
                else condition.split(">")[1].strip() if condition.count(">") == 2 \
                    else re.split(r"[<>=]", condition)[0].strip()

            season = self.w_feature_seasons[feature] if feature in self.w_feature_seasons else None
            row = {"Model": model_name,
                   "Group": self.feature_groups[feature], 
                   "Subgroup": self.feature_subgroups[feature],
                   "Season": season,
                   "Feature": feature, 
                   "Impact": impact}

            feature_importance_table.loc[len(feature_importance_table)+1] = row
            feature_importance_table["Impact"] = feature_importance_table["Impact"] / feature_importance_table["Impact"].sum()
            
        return feature_importance_table

def compare_feature_impact(importance_table1, importance_table2, save=True):
    """
    Plot feature importance for groups, subgroups and seasons for two different models.
    """
    plt.figure(figsize=(12, 6))
    gs = plt.GridSpec(2, 4, height_ratios=[1, 1])
    
    ax1 = plt.subplot(gs[0, :2])  # Big plot on the top left
    ax2 = plt.subplot(gs[0, 2:])  # Big plot on the top right
    ax3 = plt.subplot(gs[1, 0])  # Bottom-left plot 1
    ax4 = plt.subplot(gs[1, 1])  # Bottom-left plot 2
    ax5 = plt.subplot(gs[1, 2])  # Bottom-left plot 3
    ax6 = plt.subplot(gs[1, 3])  # Bottom-left plot 4
    axes = [ax1, ax2, ax3, ax4, ax5, ax6]
    
    palette = {"original": "#606c38", "attention": "#bc6c25"}
    palette2 = {"Impact_og": "#606c38", "Impact_att": "#bc6c25"}
    order = ["Pre Planting", "Planting", "Early Growth", "Reproductive Growth", "Maturation/Harvest", "Post Harvest"]
    
    # Plot groups
    og_group_impact = importance_table1.groupby(["Model", "Group"]).sum("Impact").reset_index()
    att_group_impact = importance_table2.groupby(["Model", "Group"]).sum("Impact").reset_index()
    plot1_data = pd.concat([og_group_impact, att_group_impact], axis=0)
        
    sns.barplot(ax=axes[0], x="Impact", y="Group", hue="Model", data=plot1_data, palette=palette, orient='h')
    axes[0].set_title("Impact per model and featue group")
    axes[0].get_legend().remove()

    # Plot weather features
    og_w_impact = importance_table1[importance_table1["Group"] == "weather"]
    att_w_impact = importance_table2[importance_table2["Group"] == "weather"]
    combined_w_impact = pd.concat([og_w_impact, att_w_impact], axis=0)
    plot2_data = combined_w_impact.groupby(["Model", "Season"]).sum("Impact").reset_index()
    plot2_data["Model"] = pd.Categorical(plot2_data["Model"], categories=["original", "attention"], ordered=True)
    
    sns.barplot(ax=axes[1], x="Impact", y="Season", hue="Model", data=plot2_data, palette=palette, order=order)
    axes[1].set_title("Weather features impact per model and season")
    axes[1].legend(title="Model", loc='upper right')

    # Plot subgroups
    og_subgroup_impact = importance_table1.groupby(["Model", "Group", "Subgroup"]).sum("Impact").reset_index()
    att_subgroup_impact = importance_table2.groupby(["Model", "Group", "Subgroup"]).sum("Impact").reset_index()
    merged_data = pd.merge(og_subgroup_impact, att_subgroup_impact, on=["Group", "Subgroup"], suffixes=("_og", "_att"), how='outer')
    groups = merged_data["Group"].unique()

    for i, group in enumerate(groups):
        ax = axes[i + 2]
        group_data = merged_data[merged_data["Group"] == group]
        group_data = group_data.melt(id_vars=["Subgroup"], value_vars=["Impact_og", "Impact_att"], var_name="Model", value_name="Impact")
        sns.barplot(ax=ax, x="Impact", y="Subgroup", hue="Model", data=group_data, palette=palette2)
        ax.set_title(f"{group}")
        ax.set_ylabel('')
        ax.get_legend().remove()
        ax.set_facecolor('#fefae0')
    axes[0].set_facecolor('#fefae0')
    axes[1].set_facecolor('#fefae0')
    plt.tight_layout()
    if save:
        plt.savefig(f"../assets/Feature_groups_impact_comparison.png")
    plt.show()