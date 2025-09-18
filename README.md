Car Price Prediction — What Drives the Price of a Car?
Overview
This Jupyter notebook analyzes a large dataset of used cars to identify which features most strongly impact the price. The goal is to provide actionable insights to a used car dealership about what vehicle characteristics consumers value most.

The methodology follows the CRISP-DM framework:

Business Understanding

Data Understanding

Data Preparation

Modeling

Evaluation

Deployment (Recommendations)

Key Steps
1. Data Loading & Exploration
Loads and previews a cars dataset (vehicles.csv)

Examines the structure, summary statistics, unique values, and missing data by column

2. Data Cleaning & Preprocessing
Drops non-informative columns (e.g., region, VIN)

Handles missing and extreme values (removes outliers, fills/drops NaNs)

Encodes categorical columns with OneHotEncoder and scales numerics with StandardScaler

Defines features (X) and target (y='price'), then splits into train/test

3. Exploratory Data Analysis (EDA)
Explores price distributions by year, manufacturer, and other categories using seaborn/matplotlib

Investigates relationships between features and price with scatterplots and boxplots

4. Feature Engineering & Modeling Pipeline
Constructs a preprocessing pipeline using ColumnTransformer for scalars and one-hot encoding

Implements a regularized regression (RidgeCV), tuning alpha by cross-validation

5. Ridge Regression Baseline
Trains a RidgeCV model on the full feature set

Evaluates with MAE, RMSE, R² on the test set

Extracts and ranks feature importance by absolute coefficient value to reveal key price drivers

6. Sequential Feature Selection (SFS)
Reduces feature space using SequentialFeatureSelector (SFS) with LinearRegression

Selects a limited subset of features that optimize cross-val performance

Evaluates a new model on these selected features, interprets and prints the coefficients

7. Insights & Recommendations
Summarizes which features (specific models, luxury manufacturers, age, mileage, etc.) most drive prices

Clearly interprets model coefficients for both Ridge and SFS models

Offers recommendations to dealerships on vehicle acquisition and pricing based on feature importance

Example Results
Ridge Regression: ~70% of price variance explained, MAE ~$4,346, major drivers are luxury/performance brands/models and year/mileage.

SFS: Simpler, interpretable model with top 7 features (year, odometer, specific model, cylinders, fuel, drive type, truck).

Data/plots show positive price impact for newer cars, low mileage, diesel, 4WD, and certain luxury/performance models.

How to Use
Ensure you have all dependencies (pandas, numpy, matplotlib, seaborn, scikit-learn).

Place your data file (e.g., vehicles.csv) in the working directory.

Run the notebook cells sequentially for data cleaning, EDA, modeling, and result interpretation.

Recommendations
Focus acquisitions on vehicles with newer model years and lower mileage.

Luxury and specific high-performance models command large price premiums.

Consider emphasizing features like diesel, truck body style, and advanced drive types in sales and marketing.
