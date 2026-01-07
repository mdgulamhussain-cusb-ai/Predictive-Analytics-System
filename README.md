# Predictive-Analytics-System
Tech Stack: Python, Pandas, NumPy, Scikit-learn  🔹 What this system does - Loads dataset | Handles missing values | Performs feature engineering | Scales numerical features | Trains Regression &amp; Classification models | Evaluates performance | Ready for deployment / production use.

# Install libraries 
pip install pandas numpy scikit-learn matplotlib seaborn

# Import Libraries 
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report

import warnings
warnings.filterwarnings("ignore")

# Load Dataset
data = pd.read_csv("data.csv")

print("Dataset Shape:", data.shape)
print(data.head())

# Feature Engineering & Preprocessing
# Separate target variable
TARGET_COLUMN = "target"   # change if needed

X = data.drop(columns=[TARGET_COLUMN])
y = data[TARGET_COLUMN]

# Identify column types
numerical_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

# Preprocessing steps
numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

