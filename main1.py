import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Basic EDA and data understanding
print("Original train shape:", train.shape)
print("SalePrice summary:")
print(train['SalePrice'].describe())

# Visualize distribution of SalePrice
plt.figure(figsize=(8, 5))
sns.histplot(train['SalePrice'], bins=40, kde=True)
plt.title("Distribution of SalePrice")
plt.xlabel("SalePrice")
plt.ylabel("Frequency")
plt.show()

# Outlier removal
train = train[train['GrLivArea'] < 4000]
print("Data after removing outliers:")
print(train.describe())

# Feature Engineering: Convert YearBuilt to Age
train['Age'] = train['YrSold'] - train['YearBuilt']
test['Age'] = test['YrSold'] - test['YearBuilt']

# Handle missing values
for col in train.columns:
    if train[col].dtype == "object":
        train[col] = train[col].fillna(train[col].mode()[0])
    else:
        train[col] = train[col].fillna(train[col].median())

print("Preview after filling missing values:")
print(train.head())

# Separate features and target
X = train.drop(['SalePrice', 'Id'], axis=1)
y = train['SalePrice']

# Identify categorical and numerical columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object']).columns

# Preprocessing
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# Models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Random Forest": RandomForestRegressor(random_state=42)
}

# Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


def evaluate_model(name, model):
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', model)])
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))

    plt.figure(figsize=(6, 4))
    plt.scatter(y_val, preds, alpha=0.3)
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
    plt.title(f"{name} - Actual vs Predicted")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.show()

    print(f"{name} RMSE: {rmse:.2f}")
    return pipeline, rmse


results = {}
for name, model in models.items():
    pipe, score = evaluate_model(name, model)
    results[name] = (pipe, score)

# Check for multicollinearity
X_num = train[numerical_cols].copy()
X_num = X_num.fillna(X_num.median())
vif_data = pd.DataFrame()
vif_data["feature"] = X_num.columns
vif_data["VIF"] = [variance_inflation_factor(X_num.values, i)
                   for i in range(len(X_num.columns))]
print("\nVIF Scores (Multicollinearity Check):")
print(vif_data.sort_values(by="VIF", ascending=False).head(10))

# Feature Importance from RF
best_model = results["Random Forest"][0]
rf_model = best_model.named_steps['model']
onehot_cols = best_model.named_steps['preprocessor'].named_transformers_['cat'].named_steps[
    'onehot'].get_feature_names_out(categorical_cols)
all_features = list(numerical_cols) + list(onehot_cols)
importances = rf_model.feature_importances_
feat_imp = pd.DataFrame({'Feature': all_features, 'Importance': importances})
feat_imp = feat_imp.sort_values('Importance', ascending=False)
print("\nTop 10 Important Features:")
print(feat_imp.head(10))

# Plot feature importance
plt.figure(figsize=(8, 6))
sns.barplot(data=feat_imp.head(10), x="Importance", y="Feature")
plt.title("Top 10 Feature Importances (Random Forest)")
plt.show()
