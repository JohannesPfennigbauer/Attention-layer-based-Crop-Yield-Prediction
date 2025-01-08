import streamlit as st
from streamlit_plotly_events import plotly_events
import pandas as pd
import numpy as np
from utils import (
    create_map_plot, create_actual_vs_predicted_plot, create_featuregroups_plot, create_bottom_plot, predict_wrapper
)
# import scripts.evaluation
# from scripts.evaluation import ExplainModel
import tensorflow as tf

# Load data
train_df = pd.read_csv('data/train_df.csv')
val_df = pd.read_csv('data/val_df.csv')
val_df['FIPS_cty'] = val_df['FIPS_cty'].astype(str)
test_df = pd.DataFrame(columns=['loc_id', 'year', 'yield', 'yield_pred_og', 'yield_pred_att'])
data = np.load('data/app_data.npz')
X_train = data['X_train']
M = data['M']
S = data['S']
data2 = np.load('data/test_data.npz')
test_data = data2['test_data']

og_feature_importance = pd.read_csv('data/og_feature_importance.csv')
att_feature_importance = pd.read_csv('data/att_feature_importance.csv')

# Load models
og_model = tf.keras.models.load_model('models/og_model.keras')
att_model = tf.keras.models.load_model('models/standard_att_model.keras')

# Model explanation methods:
# og_Explainer = ExplainModel(og_model, X_train, M[:,1:], S[:,1:], time_steps=5)
# att_Explainer = ExplainModel(att_model, X_train, M[:,1:], S[:,1:], time_steps=5)

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
    
def predict_county(selected_county):
    if selected_county == 'all':
        X_sample = test_data
    
    else:
        loc_id = val_df[val_df['County_State'] == selected_county]['loc_id'].values[0]
        X_sample = test_data[test_data[:, 0] == float(loc_id)]
        if X_sample.shape[0] == 0:
            st.error("No data available for the selected region.")
            return
        
    y_pred_og = predict_wrapper(og_model, X_sample[:, 3:])
    y_pred_att = predict_wrapper(att_model, X_sample[:, 3:])
    
    for i in range(X_sample.shape[0]):
        loc_id = X_sample[i, 0]
        year = X_sample[i, 1]
        y_true = X_sample[i, 2] * S[0, 0] + M[0, 0]
        y_pred_og_i = y_pred_og[i][0] * S[0, 0] + M[0, 0]
        y_pred_att_i = y_pred_att[i][0] * S[0, 0] + M[0, 0]
        test_df.loc[len(test_df)] = [loc_id, year, y_true, y_pred_og_i, y_pred_att_i]
    
    
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
            
 

    dropdown, update_btn, predict_btn, predict_all_btn = st.columns([2, 1, 1, 1])
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
            with st.spinner(f"Running prediction for {st.session_state['selected_county']}..."):
                predict_county(st.session_state["selected_county"])
            st.toast(f"Prediction completed for {st.session_state['selected_county']}!", icon="✅")
            
    with predict_all_btn:
        if st.button("Predict All"):
            with st.spinner("Running predictions for all regions..."):
                predict_county(selected_county='all')
            st.toast("All predictions completed!", icon="✅")


# Center column: Actual vs Predicted
with scatter:
    st.subheader("Actual vs Predicted")
    fig_actual_vs_predicted = create_actual_vs_predicted_plot(train_df, val_df, test_df, st.session_state["selected_county"])
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