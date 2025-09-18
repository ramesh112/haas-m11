# Car Price Prediction — What Drives the Price of a Car?

This Jupyter notebook analyzes a large dataset of used cars to identify which features most strongly impact price. The goal is to provide actionable insights to a used car dealership on the vehicle characteristics consumers value most.

## Methodology

The analysis follows the **CRISP-DM framework**:

- **Business Understanding**
- **Data Understanding**
- **Data Preparation**
- **Modeling**
- **Evaluation**
- **Deployment (Recommendations)**

## Key Steps

### Data Loading & Exploration
- Loads `vehicles.csv` and previews the dataset
- Explores structure, summary statistics, unique values, and missingness

### Data Cleaning & Preprocessing
- Drops non-informative columns (e.g., region, VIN)
- Handles missing values, removes outliers
- Encodes categorical data (OneHotEncoder), scales numerics (StandardScaler)
- Defines features (`X`) and target (`y = 'price'`); splits data

### Exploratory Data Analysis (EDA)
- Visualizes price distributions by year, manufacturer, and category (`seaborn`/`matplotlib`)
- Investigates feature-price relationships (scatterplots, boxplots)

### Feature Engineering & Modeling Pipeline
- Pipeline: `ColumnTransformer` (scaling, encoding), regularized regression (`RidgeCV`)
- Alpha tuning via cross-validation

### Baseline Model (Ridge Regression)
- Trains on full feature set, evaluates with MAE, RMSE, R² on test set
- Ranks feature importance via absolute coefficients

### Sequential Feature Selection (SFS)
- Uses SFS with `LinearRegression` to select top features
- Retrains/evaluates simplified model; interprets coefficients

### Insights & Recommendations
- Identifies key drivers: luxury brands, model year, mileage, performance models
- Interprets Ridge and SFS coefficients
- Offers actionable recommendations for dealerships

## Example Results

- **Ridge Regression:**  
  ~70% price variance explained, MAE ~$4,346  
  Key drivers: luxury/performance brands/models, year, mileage

- **SFS:**  
  Interpretable model using top 7 features (year, odometer, model, cylinders, fuel, drive type, truck)

- **Insights:**  
  - Newer, low-mileage cars and luxury/performance models have higher prices  
  - Positive impact for diesel, 4WD, truck body, and certain high-value brands/models

## How to Use

1. Install requirements: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`
2. Place `vehicles.csv` in the working directory
3. Run notebook cells in order: data cleaning → EDA → modeling → result interpretation

## Recommendations

- Focus inventory on newer cars and models with lower mileage
- Target acquisitions of luxury and high-performance models for premium pricing
- Highlight features like diesel, 4WD, and truck styling in sales and marketing
