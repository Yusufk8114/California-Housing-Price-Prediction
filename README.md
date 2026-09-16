# California-Housing-Price-Prediction
Data Analytics project showcasing Machine Learning Workflow using Python (Pandas and NumPy), Scikit-learn and Joblib.

A machine learning project that predicts **median house values for California housing districts** using a preprocessing pipeline and a Random Forest Regression model.

## Project Overview

This project implements an end-to-end machine learning workflow for California housing price prediction.

The project:

- Loads the California housing dataset from `housing.csv`
- Creates income categories for stratified sampling
- Splits the data into training and test sets while preserving income-category distribution
- Handles missing numerical values using median imputation
- Scales numerical features using `StandardScaler`
- Encodes categorical features using `OneHotEncoder`
- Trains a `RandomForestRegressor`
- Saves the trained model and preprocessing pipeline using `joblib`
- Loads the saved model for inference without retraining
- Uses `input.csv` for new data
- Saves predictions to `output.csv`

## Objective

The objective is to build a reusable machine learning pipeline that can process California housing data and predict the `median_house_value` for new observations.

## Dataset

The project uses:

```text
housing.csv
```

The target variable is:

```text
median_house_value
```

The categorical feature used in preprocessing is:

```text
ocean_proximity
```

The remaining features are treated as numerical attributes.

## Project Workflow

```text
housing.csv
    │
    ▼
Create income categories
    │
    ▼
Stratified train/test split
    │
    ├──────────────► input.csv (test data)
    │
    ▼
Separate features and target
    │
    ▼
Preprocessing Pipeline
    ├── Numerical: Median Imputation → Standard Scaling
    └── Categorical: One-Hot Encoding
    │
    ▼
Random Forest Regressor
    │
    ▼
Save model + preprocessing pipeline
    │
    ▼
Load input.csv
    │
    ▼
Transform input data
    │
    ▼
Predict median_house_value
    │
    ▼
output.csv
```

## Data Preprocessing

The preprocessing is implemented using a `ColumnTransformer` and separate pipelines for numerical and categorical attributes.

### Numerical Features

The numerical pipeline performs:

1. **Missing-value imputation** using the median.
2. **Feature scaling** using `StandardScaler`.

### Categorical Features

The categorical pipeline uses `OneHotEncoder` with:

```python
handle_unknown="ignore"
```

This allows the pipeline to handle previously unseen categorical values during inference.

## Stratified Sampling

The project creates an `income_cat` feature from `median_income` using the following bins:

```text
0.0 – 1.5
1.5 – 3.0
3.0 – 4.5
4.5 – 6.0
6.0+
```

`StratifiedShuffleSplit` is then used with:

- Test size: **20%**
- Random state: **42**

This helps preserve the income-category distribution between the training and test data.

## Machine Learning Model

The implemented prediction model is:

**Random Forest Regressor**

```python
RandomForestRegressor(random_state=42)
```

The model is trained on the preprocessed training features and the `median_house_value` target.

## Model Persistence

After training, two files are saved using Joblib:

```text
model.pkl
pipeline.pkl
```

- `model.pkl` stores the trained Random Forest model.
- `pipeline.pkl` stores the preprocessing pipeline.

The program checks whether `model.pkl` exists.

### If the model does not exist

The program:

1. Loads `housing.csv`
2. Prepares the training data
3. Trains the Random Forest model
4. Saves the model
5. Saves the preprocessing pipeline

### If the model already exists

The program:

1. Loads the saved model
2. Loads the saved preprocessing pipeline
3. Reads `input.csv`
4. Transforms the input data
5. Generates predictions
6. Saves the results to `output.csv`

This prevents unnecessary model retraining during inference.

## Input

The inference stage reads:

```text
input.csv
```

The input data is transformed using the saved preprocessing pipeline before being passed to the model.

## Output

Predictions are written to:

```text
output.csv
```

The predicted values are stored in the:

```text
median_house_value
```

column.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data loading and data manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Preprocessing, data splitting, and machine learning |
| Joblib | Saving and loading the model and pipeline |
| CSV | Dataset and prediction file format |

## Project Structure

```text
California-Housing-Price-Prediction/
│
├── housing.csv
├── input.csv
├── output.csv
├── main.py
├── model.pkl
├── pipeline.pkl
└── README.md
```

## Installation

Make sure Python is installed on your system.

Install the required libraries:

```bash
pip install pandas numpy scikit-learn joblib
```

## How to Run

Run the main Python script:

```bash
python main.py
```

### First Run

If `model.pkl` does not exist, the script trains the model and creates:

```text
model.pkl
pipeline.pkl
input.csv
```

### Subsequent Runs

If the model already exists, the script loads the saved model and pipeline, performs inference on `input.csv`, and creates:

```text
output.csv
```

## Future Improvements

Potential improvements include:

- Add explicit evaluation on the held-out test set
- Compare Linear Regression, Decision Tree, and Random Forest models
- Add cross-validation and report RMSE results
- Perform hyperparameter tuning
- Add feature engineering
- Add model performance reporting
- Build a web interface or API for predictions
- Deploy the prediction pipeline to a cloud environment

## Conclusion

This project demonstrates a complete machine learning workflow for California housing price prediction. The implementation combines stratified sampling, robust preprocessing, feature transformation, Random Forest regression, and model persistence into a reusable prediction pipeline.

The trained model and preprocessing pipeline are saved using Joblib, allowing the system to avoid unnecessary retraining. New housing data can be processed through `input.csv`, and the predicted `median_house_value` values are saved in `output.csv`.

Overall, the project provides a practical foundation for developing and deploying a reusable California housing price prediction system.
