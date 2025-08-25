# __<center>Harnessing Covid-19 Public Data for Informed Health Decision in Australia</center>__
This project delivers key insights on the COVID-19 pandemic to support the Agency for Clinical Innovation (ACI) at NSW Health in crafting data-informed public health strategies. By analyzing extensive publicly available COVID-19 data, both nationally and internationally, the project applies descriptive analysis, K-Means clustering, hypothesis testing to map the pandemic’s trajectory and assess the effectiveness of interventions.

*Full report can be found [here](https://github.com/VivianNg9/Harnessing-Covid-19-Public-Data-for-Infomred-Health-Decisions-in-Australia-/blob/main/Harnessing%20Covid-19%20Public%20Data%20for%20Informed%20Health%20Decisions%20in%20Australia.pdf)*

## Tools 
- **Programming Languages**: Python (Pandas, NumPy, Matplotlib, Scikit-learn, SciPy).
- **Data Sources**: [Our World in Data (OWID)](https://github.com/owid/covid-19-data/tree/master/public/data)
- **Statistical Methods**: Correlation analysis, hypothesis testing, and regression models.
- **Machine Learning Models**: K-Means clustering, SHAP for interpretability, XGBoost for mortality predictions.
  
## Objectives

1. Explore global and national trends to understand pandemic dynamics.
2. Analyse demographic, socioeconomic, and policy-related factors influencing COVID-19 outcomes.
3. Identify key predictors of mortality, vaccination impacts, and healthcare outcomes using statistical and machine learning techniques.

## Analytical Methods

### 1. Descriptive Analysis
- Conducted a detailed exploration of pandemic trends:
  - Case counts, death rates, and vaccination trajectories in Australia and globally.
  - Temporal analysis of the pandemic waves and their characteristics.
- Highlighted shifts in mortality and infection rates due to interventions (e.g., lockdowns, vaccinations).

### 2. K-Means Clustering
- Grouped countries based on pandemic-related indicators (e.g., GDP per capita, vaccination rates, healthcare capacity):
  - Discovered clusters representing distinct pandemic experiences (e.g., high-income nations vs. low-resource countries).
- Unveiled patterns linking economic/demographic factors to pandemic outcomes.

### 3. Hypothesis Testing
- Performed statistical hypothesis testing to validate relationships between variables:
  - Correlations between policy stringency and infection/mortality rates.
  - Lagged effects in case reporting between countries.
- Used hypothesis testing to assess vaccine efficacy and the role of comorbidities in mortality.

## Key Insights

1. **Global and National Trends**:

![Pandemic Period](https://github.com/VivianNg9/Harnessing-Covid-19-Public-Data-for-Infomred-Health-Decisions-in-Australia-/blob/main/image%20/pandemic%20period.png)
  
Australia experienced four major COVID-19 waves, with early border closures and a robust vaccination drive keeping initial infection rates low. The Delta variant prompted policy shifts, while the Omicron surge in 2021 tested healthcare systems despite high vaccination coverage, leading to increased fatalities and evolving pandemic management strategies.

2. **Clustering Analysis**:

![clustering](https://github.com/VivianNg9/Harnessing-Covid-19-Public-Data-for-Infomred-Health-Decisions-in-Australia-/blob/main/image%20/clustering.png)

A world map visualisation of clustering analysis highlighted geographic distributions, with high-income countries (Cluster 0) concentrated in Europe, North America, and parts of Asia and Oceania, while Cluster 1 included countries across Africa, Asia, and Latin America. Cluster 2 was limited to China and India, emphasizing their unique demographic and population dynamics. The analysis demonstrated how economic capabilities and demographics shaped countries’ pandemic experiences and responses. These findings underscore the importance of tailoring pandemic strategies to specific cluster needs and fostering collaboration and resource sharing among similar nations. Future research should investigate evolving factors like vaccination uptake and policy adaptations for a comprehensive understanding of pandemic responses.

3. **Policy Effectiveness**:

![policy](https://github.com/VivianNg9/Harnessing-Covid-19-Public-Data-for-Infomred-Health-Decisions-in-Australia-/blob/main/image%20/policy.png)

There is a slight negative correlation between public health measure stringency and COVID-19's R-value, but the model's low explanatory power suggests significant other factors at play.
   
4. **Vaccination and Mortality**:

4.1. Impact of Vaccination on Death Rates:

![death rates](https://github.com/VivianNg9/Harnessing-Covid-19-Public-Data-for-Infomred-Health-Decisions-in-Australia-/blob/main/image%20/death%20rate.png)

The XGB model is trained using features related to vaccination coverage, full vaccination rate, extreme poverty rate, population density, and the two underlying health indicators including cardiovascular effect and diabetes prevalence to predict death rates. Root Mean Squared Error (RMSE) equals approximately 258.86 which is considered the XGB model is appropriate. As in Figure 10, features with higher absolute SHAP values have a greater impact on the predictions. It helps in understanding the direction and magnitude of the effect. A low full vaccination rate increases the death rate when the COVID-19 wave comes up.

4.2 Impact of vaccination on age group greater than 70

![age group](https://github.com/VivianNg9/Harnessing-Covid-19-Public-Data-for-Infomred-Health-Decisions-in-Australia-/blob/main/image%20/age%20group.png)

The result of the OLS model suggests rejecting H0 and accept H1. The model demonstrates a significant relationship between vaccination and cases among the elderly (age group greater than 70). This underscores the importance of prioritising vaccinations for this vulnerable demographic.

4.3. Impact of vaccination in ICU patients 

![ICU](https://github.com/VivianNg9/Harnessing-Covid-19-Public-Data-for-Infomred-Health-Decisions-in-Australia-/blob/main/image%20/ICU.png)

The result of the OLS model suggests rejecting H0 and accept H1. The model strongly supports the notion that vaccination plays a crucial role in reducing the number of ICU patients per million. This suggests a substantial decrease in severe cases.

## Conclusion

Key findings include the effectiveness of Australia's vaccination campaign and the importance of bespoke public health strategies informed by clustering and correlation analyses with global counterparts. The analysis indicates that while Australia's strict measures have impacted virus transmission, the complexity of the pandemic's dynamics necessitates more localized and nuanced analysis for policy formulation.
Statistical methods have shown a clear benefit of vaccinations in reducing mortality rates and an association between higher vaccination rates and improved health outcomes. However, a multifaceted approach is needed, as public health measures alone show a limited correlation with changes in the reproduction rate of the virus.
