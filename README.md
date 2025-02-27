# Attention-layer-based-Crop-Yield-Prediction
This repository belongs to a project carried out in the Applied Deep Learning course at TU Vienna. 

The project is of the type "Bring your own method" with the aim to reimplement an existing DL-method and alter or enhance it to achieve better results. 

## Topic: Crop Yield Prediction
In agricultural decision-making, predicting crop yield is essential for optimizing food production, land management, and economic planning. Machine learning models, particularly deep learning approaches, have enabled more accurate and scalable crop yield predictions by integrating spatial and temporal data. One well-established method, the CNN-RNN framework developed by [Khaki et al. (2020)](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2019.01750), has demonstrated strong predictive capabilities using a combination of weather, soil, and management data. However, existing models often overlook the varying importance of different features over time, particularly in relation to plant growth stages.

This project aims to improve the CNN-RNN model by incorporating an attention mechanism that allows the model to dynamically prioritize the weather attributes that are most critical to crop yield at different growth stages. By focusing on key periods such as seeding and early development, when weather conditions have huge impact on crop establishment and future yield potential, the model will be better equipped to predict yields more accurately.

The method will then be further extended by adding explainability to justify its decisions and provide a deeper understanding of the model itself. It will also provide insights into how the attention mechanism contributes to the prediction process, thereby validating whether the improvements are due to the model's increased focus on key weather periods.

## Usability:

1. Install python version 3.10.12
2. Run `pip install -r requirements.txt`
3. Run `streamlit run app/app.py`

#### To train your own model:
4. Download data from [here](https://drive.google.com/file/d/17Qp0cZnQ8fp9Q_q4T9l31QijaYJ5mSW6/view) and save it in a new folder `data`
5. Run `scripts/convert_csv_to_npz.py` (Ensure to run the code from the main folder or change the folder path inside the script accordingly.)
6. Open `scripts/main.ipynb`
7. Specify hyperparameters and run all cells
Trained models will be saved in the folder `models` for further usage.

## Proposed Approach:
Building on the CNN-RNN structure, a multi-head attention mechanism will be integrated after the W-CNN component, which processes weekly weather data throughout the year. This attention layer will enable the model to dynamically assign greater weight to the most relevant meteorological features during critical periods of crop development, such as the early vegetative stages. Research shows that favourable weather during these growth periods is essential for effective crop establishment, weed suppression, and nutrient uptake, all of which directly contribute to biomass production and yield potential. (Butts-Wilmsmeyer et al., [2019](https://doi.org/10.3390/agronomy9010016))

This design mirrors the approach used by [Yi-Ming and Hao (2023)](https://doi.org/10.1016/j.energy.2023.127865), where a multi-head attention mechanism was successfully applied in a comparable time-series prediction task to highlight important time steps and features. The attention mechanism in this project is expected to improve the model's ability to predict crop yield by focusing on the most critical meteorological information at each growth stage.

In addition, the Layer-wise Relevance Propagation (LRP) technique will be incorporated to improve the models transparency. LRP is used to interpret model predictions by assigning relevance scores to input features, allowing for a better understanding of which weather periods and factors played the most significant role in the yield predictions. It uses the topology of the trained model and the backpropagation algorithm to identify the most relevant input neurons, corresponding to high feature importance. (Ullah et al., [2022](https://doi.org/10.3390/app12010136))

## Data
The dataset used will be the same as in the original paper about the CNN-RNN framework (Khaki et al., [2020](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2019.01750)), focusing on the **Corn Belt** region of the United States. It includes detailed weather, location, and crop yield data from 1980-2018:
- **Spatial Data**: Latitude and longitude of the fields.
- **Weather Data**: 52 weeks of features like minimum and maximum temperature, precipitation, solar radiation, snow water equivalent and vapor pressure.
- **Soil Data**: 10 different data attributes, e.g. hydraulic conductivity, pH and organic matter percentage, measured at 9 levels of depth (from 0 to 120cm).
- **Management Data:** Weekly cumulative percentage of planted fields within each state (starting from April each year)
- **Yield Data**: Average yield outcomes for specific locations per year

## Implementation
The initial implementation of the CNN-RNN network presented several challenges due to a lack of following standard coding principles. The original codebase was poorly structured, relied on TensorFlow 1.x, and it was not possible to run the code as is. To address these issues, the model was completely reimplemented in TensorFlow 2, ensuring compatibility with the latest frameworks and improving code clarity and maintainability.  

During the reimplementation, significant effort was put into structuring the code to improve readability and usability. Custom classes were defined for the CNN-RNN models, containing well-organised methods for training, prediction, evaluation and explanation. This modular design not only streamlined experimentation, but also facilitated future extensions to the model. 

To improve model performance, an attention mechanism was incorporated into the framework. This mechanism aimed to highlight the most critical temporal and feature-specific components of the input data. This was done in two steps. First, a standard attention layer was incorporated directly into the W-CNN model after the first convolution to guide the models to focus on the key weeks of the growing season for each weather feature. Subsequently, a multi-head attention layer was integrated after concatenating all W-CNN outputs to assist the model in extracting the most relevant weather attributes.

The layer-wise relevance propagation (LRP) technique was initially considered to address the need for explainability. However, due to the unavailability of TensorFlow 2 compatible libraries for LRP, an alternative method, Local Interpretable Model-agnostic Explanations (LIME), was adopted. The implementation of LIME was straightforward and allowed for detailed analysis of feature importance. This method provided valuable insights into the behaviour of the model, including the identification of critical temporal features and the relative importance of different input attributes.  

## Results
The performance of the implemented models was evaluated using Root Mean Square Error (RMSE) and R² (coefficient of determination) as the primary metrics. RMSE was chosen to quantify the average size of prediction errors, with particular emphasis on penalising larger errors, which is critical for regression tasks. This metric was also used in the original paper and therefore allows for comparison. R² was used to measure the proportion of variance in the target variable explained by the model, providing a complementary perspective on performance.

The initial goal was to exceed the RMSE of the original paper, which was $4.32$ for the validation year 2017. However, due to significant adjustments required in the model definition and data pre-processing, as well as computational constraints, this benchmark was not expected to be achieved. Instead, the target was adjusted to improve the implemented version of the original model by at least $10$%, which translated into achieving an RMSE below $4.759$ on the validation data. Although the multi-head attention model improved the model only slightly, the lighter model using only the standard attention layers proved very beneficial to model performance in most experiments. It achieved an RMSE of $4.610$, an improvement of almost $13$%.

| **Model**                 | **Training RMSE** | **Validation RMSE** | **Training R²** | **Validation R²** |  
|---------------------------|-------------------|---------------------|-----------------|-------------------|  
| Original Model            | 4.125            | 5.288               | 0.845           | 0.684             |  
| Standard Attention Model  | 4.243            | 4.610               | 0.836           | 0.760             |  
| Multi-Head Attention Model| 4.297            | 5.190               | 0.831           | 0.695             |

Across all experiments, the Standard Attention Model consistently outperformed the baseline CNN-RNN model on validation data, achieving better RMSE and R² values. Conversely, the Multi-Head Attention Model exhibited signs of overfitting, performing better on training metrics but worse on validation metrics compared to the Standard Attention Model. Interestingly, the original model occasionally performed as well as, or better than, the attention-based approaches, underscoring the robustness of the baseline architecture. Notably, the Standard Attention Model exhibited remarkable resistance to overfitting, with minimal differences observed between training and validation performance in most experiments.

The comparison of feature importance between the **original** and **standard attention** models using the LIME explainability method highlights the dominance of the weather and soil feature groups. Both models attribute the greatest impact to the weather group. Interestingly, the attention-based model gives it slightly **less** importance (~52%). However, when looking at the seasonal difference, it can be seen that the attention model gives much more importance to the growing seasons, while the original model also focuses on the post-harvest and pre-planting weeks, which can be assumed to be more noisy than useful from a practical perspective. This behaviour is intentional, as the attention mechanism was specifically implemented to focus on key weeks of the growing season. The soil group shows a similar importance in both models at around 35%, while the management and yield characteristics have a marginal impact overall, with the attention model slightly increasing the importance of management characteristics to around 15%. Details are shown in the graph below.

![Impact comparison feature groups](/assets/images/Feature_groups_impact_comparison.png)


## Work-Breakdown Structure (act: 140 / est: 132 hours)
#### 1. **Network Re-Design and Architecture Setup** (60/40 hours)
- Re-implement the CNN-RNN baseline model for crop yield prediction. (25/4 hours)
- Design and implement the attention mechanism into the W-CNN (8/16 hours)
- Implement an explainability method to interpret attention results. (12/16 hours)
- Data preprocessing and adaptation (if needed) (15/4 hours)
#### 2. **Training and Fine-Tuning** (20/20 hours)
- Train the W-CNN + attention model with the crop yield dataset. (4/4 hours)
- Fine-tune and validate the model to optimize the attention mechanisms impact. (16/16 hours)
#### 3. **Application Development** (40/40 hours)
- Develop a basic interface to visualize yield predictions for different fields and time periods. (32/24 hours)
- Build a visualization tool for feature importance according to the LIME explainability analysis. (8/16 hours)
#### 4. **Final Report Writing** (16/24 hours)
- Document model design, development, and results, with a focus on how attention improved performance. (12/12 hours)
- Explain results from LIME and provide insights if certain weather periods were more impactful. (4/12 hours)
#### 5. **Presentation Preparation** (4/8 hours)
- Summarizing the project and highlighting the attention mechanism and its impact. (4/8 hours)


## References
Khaki, S.; Wang, L.; Archontoulis, S.V. A CNN-RNN Framework for Crop Yield Prediction. *Frontiers in Plant Science* **2020**, 10. https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2019.01750

Ullah, I.; Rios, A.; Gala, V.; Mckeever, S. Explaining Deep Learning Models for Tabular Data Using Layer-Wise Relevance Propagation. *Appl. Sci.* **2022**, 12, 136. https://doi.org/10.3390/app12010136

Butts-Wilmsmeyer, C.J.; Seebauer, J.R.; Singleton, L.; Below, F.E. Weather During Key Growth Stages Explains Grain Quality and Yield of Maize. *Agronomy* **2019**, 9, 16. https://doi.org/10.3390/agronomy9010016

Yi-Ming, Z.; Hao, W. Multi-head attention-based probabilistic CNN-BiLSTM for day-ahead wind speed forecasting. *Energy* **2023**, Volume 278, Part A. https://doi.org/10.1016/j.energy.2023.127865.