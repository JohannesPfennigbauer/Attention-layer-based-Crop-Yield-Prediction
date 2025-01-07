import streamlit as st
from streamlit_plotly_events import plotly_events
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
val_df['FIPS_cty'] = val_df['FIPS_cty'].astype(str)
data = np.load('data/app_data.npz')
X_test = data['X_test']
X_train = data['X_train']
M = data['M']
S = data['S']

og_feature_importance = pd.read_csv('data/og_feature_importance.csv')
att_feature_importance = pd.read_csv('data/att_feature_importance.csv')

# Load models
og_model = tf.keras.models.load_model('models/og_model.keras')
att_model = tf.keras.models.load_model('models/standard_att_model.keras')

# Model explanation methods:
#og_Explainer = ExplainModel(og_model, X_train, M[:,1:], S[:,1:], time_steps=5)
#att_Explainer = ExplainModel(att_model, X_train, M[:,1:], S[:,1:], time_steps=5)

# Streamlit app layout
st.set_page_config(layout="wide")

# Initialize session state
if "selected_county" not in st.session_state:
    st.session_state["selected_county"] = None
    
if "county_dropdown" not in st.session_state:
    st.session_state["county_dropdown"] = None
    
def update_map():
    st.session_state["selected_county"] = st.session_state["county_dropdown"]
    st.session_state["fig_map"] = create_map_plot(val_df, st.session_state["selected_county"])
    st.toast(f"Map updated for {st.session_state['selected_county']}.", icon="✅")
    
    
# Title
st.markdown("<h2 style='text-align: center; font-size:36px'>Attention-Based Crop Yield Prediction</h2>", unsafe_allow_html=True)

# Main layout
map, scatter, featuregroups = st.columns([1, 1, 1])

# Left column: Map
with map:
    st.subheader("Prediction Regions")
    if "fig_map" not in st.session_state or st.session_state["selected_county"]:
        # Create map with the currently selected county
        st.session_state["fig_map"] = create_map_plot(val_df, st.session_state["selected_county"])
    
    #fig_map = create_map_plot(val_df, st.session_state["selected_county"])
    selected_event = plotly_events(st.session_state["fig_map"], click_event=True, select_event=True)

    if selected_event:
        # Update the selected county based on the map click
        selected_index = selected_event[0]['pointIndex']
        selected_county = val_df.iloc[selected_index]['County_State']

        if selected_county != st.session_state["selected_county"]:
            st.session_state["selected_county"] = selected_county
            st.session_state["county_dropdown"] = selected_county
            
 

    dropdown, update_btn, predict_btn = st.columns([2, 1, 1])
    with dropdown:
        selected_county = st.selectbox(
            'Select a region',
            val_df['County_State'].unique(),
            index=val_df['County_State'].tolist().index(
                st.session_state["selected_county"]) if st.session_state["selected_county"] else 0,
            on_change=update_map,
            key="county_dropdown"
        )

        
    with update_btn:
        if st.button("Update Map"):
            fig_map = create_map_plot(val_df, st.session_state["selected_county"])
            st.toast(f"Map updated for {st.session_state['selected_county']}.", icon="✅")

        
    with predict_btn:
        if st.button("Predict"):
            with st.spinner(f"Running prediction for {selected_county}..."):
                import time
                time.sleep(3)
            st.toast(f"Prediction completed for {selected_county}!", icon="✅")


# Center column: Actual vs Predicted
with scatter:
    st.subheader("Actual vs Predicted")
    fig_actual_vs_predicted = create_actual_vs_predicted_plot(train_df, val_df, st.session_state["selected_county"])
    st.plotly_chart(fig_actual_vs_predicted, use_container_width=True)

# Right column: Feature Groups and Subgroups
with featuregroups:
    st.subheader("Impact of feature groups and weather subgroup")
    fig_groups, fig_weather_subgroup = create_featuregroups_plot(og_feature_importance, att_feature_importance)
    st.plotly_chart(fig_groups, use_container_width=True)
    st.plotly_chart(fig_weather_subgroup, use_container_width=True)

# Bottom row: Combined Feature Subgroup Analysis
st.subheader("Single feature impact analysis")
fig_bottom = create_bottom_plot(og_feature_importance, att_feature_importance)
st.plotly_chart(fig_bottom, use_container_width=True)