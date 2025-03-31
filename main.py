import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Initial data cleaning
# Remove outliers
train = train[train['GrLivArea'] < 4000]
print("Data after removing outliers:")
print(train.describe())

# Fill missing values for 'LotFrontage'
train.loc[:, 'LotFrontage'] = train['LotFrontage'].fillna(train['LotFrontage'].median())

# General imputation for remaining missing values
for col in train.columns:
    if train[col].dtype == "object":
        train.loc[:, col] = train[col].fillna(train[col].mode()[0])
    else:
        train.loc[:, col] = train[col].fillna(train[col].median())

print("Data after filling missing values:")
print(train.head())

# Separate features and target
X = train.drop(['SalePrice', 'Id'], axis=1)  # Exclude 'Id' if present
y = train['SalePrice']

# Define numerical and categorical columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object']).columns

# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)])

# Define the models
linear_model = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', LinearRegression())])
ridge_model = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', Ridge())])
rf_model = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', RandomForestRegressor())])

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Plotting function
def plot_predictions(model_name, y_val, preds):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_val, preds, alpha=0.3)
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], '--r', linewidth=2)
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.title(f'{model_name} - Actual vs Predicted Prices')
    plt.show()

# Evaluation function
def evaluate_model(model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    plot_predictions(model.__class__.__name__, y_val, preds)
    return rmse

# Evaluate models
results = {
    'Linear Regression': evaluate_model(linear_model, X_train, y_train, X_val, y_val),
    'Ridge Regression': evaluate_model(ridge_model, X_train, y_train, X_val, y_val),
    'Random Forest Regression': evaluate_model(rf_model, X_train, y_train, X_val, y_val)
}

# Print results
for model_name, rmse in results.items():
    print(f"{model_name} RMSE: {rmse}")

# Random Forest feature importance analysis
if 'Random Forest Regression' in results:
    feature_importances = rf_model.named_steps['model'].feature_importances_
    features = np.array(numerical_cols.tolist() +
                        list(rf_model.named_steps['preprocessor']
                             .named_transformers_['cat']
                             .named_steps['onehot']
                             .get_feature_names_out()))
    importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    print(importance_df.head(10))
