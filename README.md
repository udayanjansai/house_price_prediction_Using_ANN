#House Price Prediction: ANN vs. Regularized Linear Models
Project Overview
This project aims to predict house prices using a dataset containing various housing features. We explore and compare the performance of three different machine learning models: an Artificial Neural Network (ANN), Ridge Regression, and Lasso Regression. The goal is to identify which model provides the best generalization on unseen data.

Dataset
The dataset used is housing.csv, which contains information about houses, including longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income, median_house_value (target variable), and ocean_proximity.

Methodology
1. Data Loading and Initial Exploration
Loaded the dataset into a Pandas DataFrame.
Performed initial checks for null values and data types (df.info(), df.isnull().sum()).
2. Data Preprocessing
Missing Values: Imputed missing values in total_bedrooms with the mean.
Duplicate Rows: Checked for and confirmed no duplicate rows.
Feature and Target Split: Separated the target variable (median_house_value) from the features (X).
Categorical Feature Encoding: Applied OneHotEncoder to the ocean_proximity categorical column.
Feature Scaling: Used MinMaxScaler to scale all numerical features (including the one-hot encoded features) to a range between 0 and 1.
Data Split: Split the processed data into training and testing sets (80% training, 20% testing).
3. Model Building and Training
a. Artificial Neural Network (ANN)
Architecture: A Sequential Keras model with an input layer, three dense hidden layers (128, 64, 32 neurons with ReLU activation), and a linear output layer.
Compilation: Compiled with the Adam optimizer (learning_rate=0.001) and Mean Squared Error (mse) loss function.
Callbacks: Utilized EarlyStopping (patience=10) and ReduceLROnPlateau (patience=5, factor=0.5) to prevent overfitting and adjust the learning rate during training.
Training: Trained for 100 epochs with a batch size of 32.
b. Ridge Regression
Model: Implemented using sklearn.linear_model.Ridge.
Hyperparameter Tuning: Used GridSearchCV to find the optimal alpha (regularization strength) from [0.01, 0.1, 1, 10, 100] with 5-fold cross-validation.
c. Lasso Regression
Model: Implemented using sklearn.linear_model.Lasso.
Hyperparameter Tuning: Used GridSearchCV to find the optimal alpha from [0.01, 0.1, 1, 10, 100] with 5-fold cross-validation.
4. Model Evaluation
All models were evaluated on the test set using the following metrics:

Mean Absolute Error (MAE)
Mean Squared Error (MSE)
Root Mean Squared Error (RMSE)
Scatter plots of predicted vs. actual house prices were generated for each model to visualize their performance.

5. Model Comparison
A summary table (performance_df.csv) was created to compare the MAE, MSE, and RMSE values across all three models.

Key Findings
Based on the evaluation metrics:

ANN Model: MAE: 46083.67, MSE: 4.376369e+09, RMSE: 66154.13
Ridge Model: MAE: 50686.34, MSE: 4.901200e+09, RMSE: 70008.57
Lasso Model: MAE: 50700.67, MSE: 4.903969e+09, RMSE: 70028.34
The Artificial Neural Network (ANN) model demonstrated the best generalization performance on the test set, achieving the lowest MAE, MSE, and RMSE among the three models.

How to Run the Notebook
Open in Google Colab: Upload the .ipynb notebook file to Google Colab.
Ensure Data is Present: Make sure housing.csv is uploaded to the /content/ directory or update the data loading path accordingly.
Run All Cells: Execute all cells in the notebook sequentially.
Dependencies
Key libraries used in this project include:

pandas
numpy
scikit-learn (for preprocessing and linear models)
tensorflow (for the ANN model)
matplotlib (for plotting)
seaborn (for plotting)
These libraries can typically be installed via pip if not already available in your environment:

pip install pandas numpy scikit-learn tensorflow matplotlib seaborn
