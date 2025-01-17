import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lime.lime_tabular import LimeTabularExplainer
from utils import predict_wrapper

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
        
    def predict_wrap(self, X):
        """
        Wrapper function for the model's predict method.
        """
        return predict_wrapper(self.model, X)
    
    def explain_observation(self, X, index):
        """
        Explain a single observation.
        """
        explanation = self.explainer.explain_instance(
            X[index],
            self.predict_wrap,
            num_features=30
        )
        feature_values = {feature: abs(value) for feature, value in explanation.as_list()}
        feature_values = sorted(feature_values.items(), key=lambda x: x[1], reverse=True)

        return feature_values
    
    
    def explain_many_observations(self, X, indices):
        """
        Explain multiple observations and sum the weights of the top features.
        """
        explanations = []
        if indices == 'all':
            np.random.seed(42)
            indices = np.random.choice(range(X.shape[0]), 5)
            for index in indices:
                explanations.append(self.explain_observation(X, index))
        else:
            for index in indices:
                for i in range(3):
                    explanations.append(self.explain_observation(X, index))
            
        
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
