import streamlit as st
import pandas as pd
import numpy as np
from utils import (
    create_map_plot, create_actual_vs_predicted_plot, create_featuregroups_plot, create_bottom_plot
)
# import scripts.evaluation
# from scripts.evaluation import ExplainModel
import tensorflow as tf

# Load data
train_df = pd.read_csv('data/train_df.csv')
val_df = pd.read_csv('data/val_df.csv')
data = np.load('data/app_data.npz')
X_test = data['X_test']
X_train = data['X_train']
M = data['M']
S = data['S']

og_feature_importance = pd.read_csv('data/og_feature_importance.csv')
att_feature_importance = pd.read_csv('data/att_feature_importance.csv')

# Load plots
fig_map = create_map_plot(val_df)
fig_actual_vs_predicted = create_actual_vs_predicted_plot(train_df, val_df)
fig_groups, fig_weather_subgroup = create_featuregroups_plot(og_feature_importance, att_feature_importance)
fig_bottom = create_bottom_plot(og_feature_importance, att_feature_importance)

# Load models
og_model = tf.keras.models.load_model('models/og_model.keras')
att_model = tf.keras.models.load_model('models/standard_att_model.keras')

# Model explanation methods:
#og_Explainer = ExplainModel(og_model, X_train, M[:,1:], S[:,1:], time_steps=5)
#att_Explainer = ExplainModel(att_model, X_train, M[:,1:], S[:,1:], time_steps=5)


# Streamlit app layout
st.set_page_config(layout="wide")

# Region selection and Predict button
col1, col2, col3, _ = st.columns([1, 1, 3, 2])
with col1:
    selected_county = st.selectbox('Select a region', val_df['County_State'].unique())
with col2:
    if st.button("Predict"):
        st.write(f"Running prediction for {selected_county}...")
with col3:
    st.markdown("<h2 style='text-align: center; font-size:22px'>Attention-Based Crop Yield Prediction</h1>", unsafe_allow_html=True)

# Main layout
col_left, col_center, col_right = st.columns([1, 1, 1])

## Left column: Observation Regions (Map)
with col_left:
    st.subheader("Prediction Regions")
    st.plotly_chart(fig_map, use_container_width=True)

## Center column: Actual vs Predicted
with col_center:
    st.subheader("Actual vs Predicted")
    st.plotly_chart(fig_actual_vs_predicted, use_container_width=True)

## Right column: Feature Groups and Subgroups
with col_right:
    st.subheader("Impact of feature groups and weather subgroup")
    st.plotly_chart(fig_groups, use_container_width=True)
    st.plotly_chart(fig_weather_subgroup, use_container_width=True)

# Bottom row: Combined Feature Subgroup Analysis
st.subheader("Single feature impact analysis")
st.plotly_chart(fig_bottom, use_container_width=True)