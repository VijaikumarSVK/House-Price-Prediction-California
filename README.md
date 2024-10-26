
# House Price Prediction California

**Executive Summary:** This project utilizes machine learning techniques to predict housing prices in California. Using the California housing dataset, the project explores data preprocessing, feature engineering, model selection, and evaluation. The blog details the process, highlighting key insights and challenges encountered. Finally, deployment strategies and ongoing maintenance considerations are discussed.

Link for my [Portfolio(House Price Prediction California)](https://vijaikumarsvk.github.io/House-Price-Prediction-California/)


### Fetching Data
The first step in any machine learning project is gathering the data. For this project, we'll be using the California housing dataset, which contains information about various housing characteristics and their corresponding prices.

We load the data into a pandas DataFrame and explore its structure using **info()** and **describe()** methods. This allows us to understand the data types of each attribute, identify missing values, and gain insights into the data's distribution.

### Data Visualization
Visualizing the data is crucial for understanding its underlying patterns. We create histograms to examine the distribution of each numerical attribute. This helps us identify potential issues like skewed distributions or capped values, which may impact model performance.

![alt text](https://res.cloudinary.com/dqqjik4em/image/upload/v1729822514/attribute_histogram_plots.png)

The histograms reveal interesting insights:
1. **Median income:** The data is capped, with the highest value at 15, representing $150,000. This might affect the model's ability to predict prices accurately for high-income areas.
2. **Housing median value:** The housing prices are also capped, possibly limiting the model's prediction range.
3. **Skewed Distributions:** Many histograms exhibit right-skewness, potentially challenging certain machine learning algorithms in detecting patterns.


### Creating a Stratified Train-Test Split
To avoid data snooping bias, we create a stratified train-test split based on the **median_income** attribute. This ensures that both the training and testing sets have representative proportions of different income categories, improving the model's generalizability.

We compare the income category proportions in the overall dataset, stratified test set, and random test set. The stratified split maintains a near-perfect representation of the original proportions, minimizing potential bias

![alt text](https://res.cloudinary.com/dqqjik4em/image/upload/v1729824366/compare_table.png)

## Exploratory Data Analysis (EDA)
Now, we dive deeper into the training data to explore relationships between attributes and identify potential patterns.

### Visualizing Geographical Data
We begin by visualizing the geographical distribution of housing data using scatter plots. This provides a visual representation of housing density and price variations across different regions.


![alt text](https://res.cloudinary.com/dqqjik4em/image/upload/v1729824996/Geographical_scatter_plot.png)

The plot highlights areas with high housing density, like the Bay Area, Los Angeles, and San Diego, suggesting potential price premiums in these locations.

#### Housing Price as per Location and Population
We enhance the scatter plot by incorporating housing prices and population density. This allows us to observe correlations between these factors and geographical location.
![alt text](https://res.cloudinary.com/dqqjik4em/image/upload/v1729826728/housing_prices_scatterplot.png)

The plot reveals a strong correlation between housing prices and location, with coastal areas exhibiting higher prices. Population density also seems to play a role, with denser areas generally having higher housing costs.

![alt text](https://res.cloudinary.com/dqqjik4em/image/upload/v1729826867/california_housing_prices_plot.png)

This visualization reinforces the previous findings, showing a clear concentration of high-priced homes near the coast and in densely populated areas.


## Correlations
We explore correlations between different attributes and the target variable (median house value) using a correlation matrix. This helps us identify potentially valuable features for our model.

![alt text](https://res.cloudinary.com/dqqjik4em/image/upload/v1729827553/scatter_matrix_plot.png)

Since we are having 11 numerical column, for visual manner we will get lot of plot(121). To avoid that we are focusing on few attributes that seem most correlated

![alt text](https://res.cloudinary.com/dqqjik4em/image/upload/v1729827659/income_vs_house_value_scatterplot.png)

The correlation matrix reveals strong positive correlations between 'median_income' and 'median_house_value', suggesting its potential importance in predicting housing prices. Other attributes like 'total_rooms' and 'housing_median_age' also exhibit some correlation with the target variable.


### Feature Engineering and Experimenting their combination
To improve model performance, we experiment with feature engineering by creating new features based on existing ones. We analyze the correlation of these new features with the target variable to assess their usefulness.

![alt text](https://res.cloudinary.com/dqqjik4em/image/upload/v1729914845/corr_image.png)

The new features, like 'rooms_per_house' and 'bedroom_ratio', provide additional information about the housing characteristics and contribute to a better understanding of the dataset.


## Preparing Data for Machine Learning
Before training our model, we need to prepare the data by addressing missing values, handling categorical attributes, and scaling numerical features.

### Data Cleaning  
We address missing values in the 'total_bedrooms' attribute by replacing them with the median value. However, to handle potential missing values in other columns with new datasets, we implement the **Imputer** function.

### Handling Text and Categorical columns
We handle the categorical attribute **'ocean_proximity'** by converting it into numerical values using both **Ordinal Encoder** and **OneHot Encoder**. These techniques enable the model to process categorical data effectively.

### Feature Scaling and Transformation
We explore different feature scaling techniques, namely **MinMaxScaler** and **StandardScaler**, to standardize the numerical attributes and improve model performance.

## Transformation Pipelines
To streamline the data preprocessing steps and ensure consistent execution order, we utilize **Pipeline** and **ColumnTransformer.**

We demonstrate the use of **make_pipeline()** and **column_transformer()** for simplified pipeline creation and highlight the advantages of using these techniques for complex data transformations.
We then combine all the preprocessing steps into a single **ColumnTransformer**, efficiently handling both numerical and categorical attributes

## Select and Train a Model

We start by training a Linear Regression model and evaluating its performance using RMSE. The model shows signs of underfitting, prompting us to explore more powerful models.

We then train a Decision Tree Regressor, which overfits the training data, indicating the need for model validation techniques.

### Evaluation using Cross - Validation
To address overfitting and obtain a more reliable performance estimate, we utilize cross-validation with 10 folds. This technique provides a robust evaluation of the Decision Tree Regressor, revealing its limitations in generalizing to unseen data.

Next, we explore a Random Forest Regressor, which exhibits significantly better performance compared to the previous models.

### Model Fine-Tune
To further enhance the Random Forest Regressor's performance, we perform model fine-tuning using Grid Search and Randomized Search. These techniques explore different hyperparameter combinations to identify the optimal model configuration.

We analyze the results of these searches, highlighting the best hyperparameters and their impact on model performance.

### Analyzing best models and their errors
We examine the feature importances of the best model, providing insights into the most influential features in predicting housing prices. This helps us understand the model's decision-making process and identify key factors driving price variations.

### Evaluating the Test Set
Finally, we evaluate the final model on the test set, using RMSE and confidence intervals to assess its generalizability and prediction accuracy.

```
X_test = strat_test_set.drop('median_house_value', axis = 1)
y_test = strat_test_set['median_house_value'].copy()
final_predictions = final_model.predict(X_test)
final_rmse = mean_squared_error(y_test, final_predictions, squared = False)
print(final_rmse)
// Output --> 41424.4

// We can compute a 95% confidence interval for the test RMSE
from scipy import stats
confidence = 0.95
squared_errors = (final_predictions - y_test)**2
np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                        loc = squared_errors.mean(),
                        scale = stats.sem(squared_errors)))
//Output -->39275.4, 43467.3

// Showing how to compute a confidence interval for the RMSE
m = len(squared_errors)
mean = squared_errors.mean()
tscore = stats.t.ppf((1+confidence)/2, df = m -1)
tmargin = tscore*squared_errors.std(ddof=1)/np.sqrt(m)
np.sqrt(mean - tmargin), np.sqrt(mean+tmargin)
// Output --> 39275.4, 43467.3

//Alternatively, we can use Z-score rather than t-score. Since the test is too small, it won't make a huge difference.
zscore = stats.norm.ppf((1+confidence)/2)
zmargin = zscore * squared_errors.std(ddof = 1)/np.sqrt(m)
np.sqrt(mean - zmargin), np.sqrt(mean+zmargin)
// Output --> 39276.1, 43466.7
```


## Launch, Monitor and Maintain our System
We discuss deployment strategies, highlighting the importance of monitoring the model's live performance, ensuring data quality, and maintaining backups for rollback capabilities.

We showcase how to load the saved model and use it to predict housing prices for new data, demonstrating the model's practical applicability.

**new_data**
![alt text](https://res.cloudinary.com/dqqjik4em/image/upload/v1729912426/new_data.png)

**Original value** of housing_labels.iloc[:5]
![alt text](https://res.cloudinary.com/dqqjik4em/image/upload/v1729912643/original_value_housing_label.png)

**Predicted values**
![alt text](https://res.cloudinary.com/dqqjik4em/image/upload/v1729912752/predicted.png)

By following these steps, we successfully built a machine learning model to predict housing prices in California. The project demonstrates the importance of data exploration, preprocessing, feature engineering, model selection, evaluation, and deployment considerations for creating robust and reliable machine learning solutions.
