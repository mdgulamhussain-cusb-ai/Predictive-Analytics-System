# Predictive-Analytics-System
Tech Stack: Python, Pandas, NumPy, Scikit-learn   What this system does - Loads dataset | Handles missing values | Performs feature engineering | Scales numerical features | Trains Regression & Classification models | Evaluates performance | Ready for deployment / production use.

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
#Separate target variable
TARGET_COLUMN = "target"   # change if needed

X = data.drop(columns=[TARGET_COLUMN])
y = data[TARGET_COLUMN]

#Identify column types
numerical_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

#Preprocessing steps
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
#Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#Regression Model (Predictive Analytics)
regression_pipeline = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("model", LinearRegression())
])

regression_pipeline.fit(X_train, y_train)

y_pred_reg = regression_pipeline.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred_reg))
r2 = r2_score(y_test, y_pred_reg)

print("Regression Model Performance")
print("RMSE:", rmse)
print("R2 Score:", r2)


#Classification Model (Convert target to binary if needed.)
# Example: converting target to binary classification
y_class = (y > y.median()).astype(int)

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_class, test_size=0.2, random_state=42
)

classification_pipeline = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])

classification_pipeline.fit(X_train_c, y_train_c)

y_pred_class = classification_pipeline.predict(X_test_c)

accuracy = accuracy_score(y_test_c, y_pred_class)

print("\n Classification Model Performance")
print("Accuracy:", accuracy)
print(classification_report(y_test_c, y_pred_class))


#Save Trained Model (Deployment Ready)
import joblib
joblib.dump(classification_pipeline, "predictive_model.pkl")
print("Model saved successfully!")

#Load & Predict (Production Usage
loaded_model = joblib.load("predictive_model.pkl")
sample_input = X_test.iloc[:1]
prediction = loaded_model.predict(sample_input)
print("Prediction:", prediction)

Add csv file and go on!!!



