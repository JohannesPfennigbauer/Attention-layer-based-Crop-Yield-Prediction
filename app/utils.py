import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from urllib.request import urlopen
import json

# Load geojson for map
def load_geojson(url):
    with urlopen(url) as response:
        return json.load(response)
    
# Plot functions
def create_map_plot(val_df):
    counties = load_geojson('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json')
 
    fig = px.choropleth_mapbox(
        val_df,
        geojson=counties,
        locations='FIPS_cty',
        color='yield_pred_att',
        color_continuous_scale="Viridis",
        range_color=(20, 65),
        mapbox_style="carto-positron",
        zoom=4,
        center={"lat": 43, "lon": -94},
        opacity=0.5,
        labels={'yield_pred_att': 'Yield Prediction', 'FIPS_cty': 'County FIPS', 'County_State': 'County, State', 'year': 'Year'},
        hover_data={'County_State': True, 'year': True}
    )
    fig.update_layout(
        height=450,
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        paper_bgcolor='rgba(201, 169, 135, 0.1)',
        plot_bgcolor='rgba(0, 0, 0, 0)'
        )
    return fig

def create_actual_vs_predicted_plot(train_df, val_df):
    combined_df = pd.concat([train_df.assign(dataset='training'), val_df.assign(dataset='validation')])
    palette = ['#606c38','#bc6c25']
    fig = px.scatter(
        combined_df,
        x='yield',
        y='yield_pred_att',
        color='dataset',
        color_discrete_map={key: value for key, value in zip(['training', 'validation'], palette)},
        labels={'yield': 'Actual', 'yield_pred_att': 'Predicted', 'color': 'Dataset', 'year': 'Year'},
        hover_data={'County_State': True, 'year': True},
    )
    
    fig.add_shape(
        type='line',
        x0=combined_df['yield'].min(), y0=combined_df['yield'].min(),
        x1=combined_df['yield'].max(), y1=combined_df['yield'].max(),
        line=dict(color='#87BF7B', dash='dash')
    )
    
    fig.update_layout(
        height=450,
        paper_bgcolor='rgba(201, 169, 135, 0.1)',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        legend=dict(
            title=dict(text='Dataset', font=dict(color='#D4D4D4')),
            orientation='h',
            x=0.5, 
            y=1.02, 
            xanchor='center', 
            yanchor='bottom',
            bgcolor='rgba(241, 241, 241, 0.2)',
            font=dict(color='#D4D4D4'),
            title_side='left'
        ),
        margin={"r": 20, "t": 20, "l": 0, "b": 0},
        xaxis=dict(showgrid=True, gridcolor='#D4D4D4'),
        yaxis=dict(showgrid=True, gridcolor='#D4D4D4')
    )
    return fig

def create_featuregroups_plot(importance_table1, importance_table2):
    """
    Plot the two big feature importance plots for feature groups and weather subgroups.
    """
    # Plot groups
    og_group_impact = importance_table1.groupby(["Model", "Group"]).sum("Impact").reset_index()
    att_group_impact = importance_table2.groupby(["Model", "Group"]).sum("Impact").reset_index()
    plot1_data = pd.concat([og_group_impact, att_group_impact], axis=0)

    fig1 = px.bar(
        plot1_data,
        x="Impact",
        y="Group",
        color="Model",
        barmode="group",
        color_discrete_map={"original": "#606c38", "attention": "#bc6c25"},
        category_orders={"Model": ["original", "attention"]}
    )
    fig1.update_layout(
        height=218,
        paper_bgcolor='rgba(201, 169, 135, 0.1)',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        showlegend=True,
        legend=dict(
            title=dict(text='Model', font=dict(color='#D4D4D4')),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(241, 241, 241, 0.2)',
            font=dict(color='#D4D4D4'),
            title_side="left"
        ),
        margin={"r": 10, "t": 10, "l": 0, "b": 0},
        xaxis=dict(showgrid=True, gridcolor='#D4D4D4')
    )

    # Plot weather features
    og_w_impact = importance_table1[importance_table1["Group"] == "weather"]
    att_w_impact = importance_table2[importance_table2["Group"] == "weather"]
    combined_w_impact = pd.concat([og_w_impact, att_w_impact], axis=0)
    plot2_data = combined_w_impact.groupby(["Model", "Season"]).sum("Impact").reset_index()
    plot2_data["Model"] = pd.Categorical(plot2_data["Model"], categories=["original", "attention"], ordered=True)

    fig2 = px.bar(
        plot2_data,
        x="Impact",
        y="Season",
        color="Model",
        barmode="group",
        category_orders={"Season": ["Pre Planting", "Planting", "Early Growth", "Reproductive Growth", "Maturation/Harvest", "Post Harvest"]},
        color_discrete_map={"original": "#606c38", "attention": "#bc6c25"}
    )
    fig2.update_layout(
        height=216,
        paper_bgcolor='rgba(201, 169, 135, 0.1)',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        showlegend=False,
        margin={"r": 10, "t": 10, "l": 0, "b": 0},
        xaxis=dict(showgrid=True, gridcolor='#D4D4D4')
    )

    return fig1, fig2

def create_bottom_plot(importance_table1, importance_table2):
    """
    Plot the four smaller feature importance plots (horizontally aligned) using Plotly Express.
    Returns the Plotly figure.
    """
    # Plot subgroups
    og_subgroup_impact = importance_table1.groupby(["Model", "Group", "Subgroup"]).sum("Impact").reset_index()
    att_subgroup_impact = importance_table2.groupby(["Model", "Group", "Subgroup"]).sum("Impact").reset_index()
    merged_data = pd.merge(og_subgroup_impact, att_subgroup_impact, on=["Group", "Subgroup"], suffixes=("_og", "_att"), how='outer')
    groups = merged_data["Group"].unique()
    y_max = max(merged_data["Impact_og"].max(), merged_data["Impact_att"].max()) + 0.01

    figs = []
    for group in groups:
        group_data = merged_data[merged_data["Group"] == group]
        group_data = group_data.melt(
            id_vars=["Subgroup"],
            value_vars=["Impact_og", "Impact_att"],
            var_name="Model",
            value_name="Impact"
        )
        fig = px.bar(
            group_data,
            x="Subgroup",
            y="Impact",
            color="Model",
            title=f"{group}",
            color_discrete_map={"Impact_og": "#606c38", "Impact_att": "#bc6c25"}
        )

        figs.append(fig)
        
    fig_bottom = make_subplots(rows=1, cols=len(figs), subplot_titles=[f"{group}" for group in groups])

    for i, fig in enumerate(figs):
        for trace in fig.data:
            fig_bottom.add_trace(trace, row=1, col=i + 1)
    fig_bottom.update_yaxes(range=[0, y_max], row=1)
    fig_bottom.update_layout(
        showlegend=False,
        height=250,
        margin={"r": 0, "t": 25, "l": 0, "b": 0},
        paper_bgcolor='rgba(201, 169, 135, 0.1)',
        plot_bgcolor='rgba(0, 0, 0, 0)',
    )

    for i in range(1, len(figs) + 1):
        fig_bottom.update_yaxes(showgrid=True, gridcolor='#D4D4D4', row=1, col=i)

    return fig_bottom
    
    
    
# Interactivity
def predict_wrapper(model, X, time_steps=5, loc_id=None):
    """
    Wrapper function to handle the input data, format it as dictionary and make predictions.
    """
    if loc_id is not None:
        X = X[X[:, 0] == loc_id]
    X = X[:, 3:] # without loc_id, year and yield
    X = np.expand_dims(X, axis=-1)
    X_in = {f'w{i}': X[:, 52*i:52*(i+1), :] for i in range(6)}
    X_in.update({f's{i}': X[:, 312+6*i:312+6*(i+1), :] for i in range(11)})
    X_in['p'] = X[:, 378:392, :]
    X_in['avg_yield'] = X[:, -time_steps:, :]
    
    return model.predict(X_in)