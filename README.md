# Attention-layer-based-Crop-Yield-Prediction
This repository belongs to a project carried out in the Applied Deep Learning course at TU Vienna. 

The project is of the type "Bring your own method" with the aim to reimplement an existing DL-method and alter or enhance it to achieve better results. 

## Topic: Crop Yield Prediction
In agricultural decision-making, predicting crop yield is essential for optimizing food production, land management, and economic planning. Machine learning models, particularly deep learning approaches, have enabled more accurate and scalable crop yield predictions by integrating spatial and temporal data. One well-established method, the CNN-RNN framework developed by [Khaki et al. (2020)](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2019.01750), has demonstrated strong predictive capabilities using a combination of weather, soil, and management data. However, existing models often overlook the varying importance of different features over time, particularly in relation to plant growth stages.

This project aims to improve the CNN-RNN model by incorporating a multi-head attention mechanism that allows the model to dynamically prioritize the weather attributes that are most critical to crop yield at different growth stages. By focusing on key periods such as seeding and early development, when weather conditions have huge impact on crop establishment and future yield potential, the model will be better equipped to predict yields more accurately.

The method will then be further extended by adding explainability to justify its decisions and provide a deeper understanding of the model itself. It will also provide insights into how the attention mechanism contributes to the prediction process, thereby validating whether the improvements are due to the model's increased focus on key weather periods.

## Approach:
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

## Work-Breakdown Structure
#### 1. **Network Re-Design and Architecture Setup** (5 days)
- Re-implement the CNN-RNN baseline model for crop yield prediction. (< 1 day)
- Design and implement the attention mechanism into the W-CNN (2 days)
- Implement the LRP explainability method to interpret attention results. (2 days)
- Data preprocessing and adaptation (if needed) (< 1 day)
#### 2. **Training and Fine-Tuning** (2-3 days)
- Train the W-CNN + attention model with the crop yield dataset. (< 1 day)
- Fine-tune and validate the model to optimize the attention mechanisms impact. (2 days)
#### 3. **Application Development** (5 days)
- Develop a basic interface to visualize yield predictions for different fields and time periods. (3 days)
- Build a visualization tool for feature importance according to the LRP explainability analysis. (2 days)
#### 4. **Final Report Writing** (3 days)
- Document model design, development, and results, with a focus on how attention improved performance. (1-2 days)
- Explain results from LRP and provide insights if certain weather periods were more impactful. (1-2 days)
#### 5. **Presentation Preparation** (1 days)
- Summarizing the project and highlighting the attention mechanism and its impact. (1 day)

## References
Khaki, S.; Wang, L.; Archontoulis, S.V. A CNN-RNN Framework for Crop Yield Prediction. *Frontiers in Plant Science* **2020**, 10. https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2019.01750

Ullah, I.; Rios, A.; Gala, V.; Mckeever, S. Explaining Deep Learning Models for Tabular Data Using Layer-Wise Relevance Propagation. *Appl. Sci.* **2022**, 12, 136. https://doi.org/10.3390/app12010136

Butts-Wilmsmeyer, C.J.; Seebauer, J.R.; Singleton, L.; Below, F.E. Weather During Key Growth Stages Explains Grain Quality and Yield of Maize. *Agronomy* **2019**, 9, 16. https://doi.org/10.3390/agronomy9010016

Yi-Ming, Z.; Hao, W. Multi-head attention-based probabilistic CNN-BiLSTM for day-ahead wind speed forecasting. *Energy* **2023**, Volume 278, Part A. https://doi.org/10.1016/j.energy.2023.127865.
