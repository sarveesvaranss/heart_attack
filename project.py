# Section 1: Data Loading & Initial Preprocessing

# Import basic libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv('heart_disease_data.csv')


# Rename columns for clarity
df.columns = [
    "age", "sex", "cp", "trtbps", "chol", "fbs", "rest_ecg", "thalach",
    "exang", "oldpeak", "slope", "ca", "thal", "target"
]

# Basic dataset info
print("Dataset Shape:", df.shape)
print(df.info())
print("Missing Values:\n", df.isnull().sum())

# Section 2: Handling Missing Data & Basic EDA

import missingno

# Visualize missing data
missingno.bar(df, color="blue")
plt.title("Missing Values in Dataset")
plt.show()

# Replace 'thal' value 0 with NaN
df['thal'] = df['thal'].replace(0, np.nan)

# Fill missing 'thal' values with mode
df['thal'].fillna(2, inplace=True)
df['thal'] = pd.to_numeric(df['thal'], downcast='integer')

# Section 3: Visualizations

# Numerical and Categorical Variables
numeric_features = ["age", "trtbps", "chol", "thalach", "oldpeak"]
categorical_features = ["sex", "cp", "fbs", "rest_ecg", "exang", "slope", "ca", "thal", "target"]

# Distribution plots for numeric features
for feature in numeric_features:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[feature], kde=True, bins=20)
    plt.title(f"Distribution of {feature}")
    plt.tight_layout()
    plt.show()

# Count plots for categorical features
for feature in categorical_features:
    plt.figure(figsize=(8, 4))
    sns.countplot(x=feature, data=df, hue="target")
    plt.title(f"{feature} vs Target")
    plt.tight_layout()
    plt.show()

# Section 4: Outlier Detection and Treatment

from scipy.stats.mstats import winsorize

# Winsorize 'trtbps'
df["trtbps_winsorized"] = winsorize(df["trtbps"], limits=(0, 0.01))

# Winsorize 'oldpeak'
df["oldpeak_winsorized"] = winsorize(df["oldpeak"], limits=(0, 0.01))

# Drop old columns
df.drop(["trtbps", "oldpeak"], axis=1, inplace=True)

# Section 5: Feature Engineering & Scaling

from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Log or sqrt transform if needed
df["oldpeak_sqrt"] = np.sqrt(df["oldpeak_winsorized"])

# Drop unnecessary features
df.drop(["chol", "fbs", "rest_ecg"], axis=1, inplace=True)

# Section 6: Define Preprocessing Pipeline

# Features to scale and encode
numeric_features = ["age", "thalach", "trtbps_winsorized", "oldpeak_sqrt"]
categorical_features = ["sex", "cp", "exang", "slope", "ca", "thal"]

# Create the preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ("scaler", RobustScaler())
])

categorical_transformer = Pipeline(steps=[
    ("encoder", OneHotEncoder(drop='first'))
])

# Combine transformers into a preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Section 7: Model Training with Pipeline

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# Train-test split
X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Section 8: Create Pipelines for Logistic Regression, SVM, and Random Forest

# Logistic Regression Pipeline
log_reg_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression())
])

# Random Forest Pipeline
rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier())
])

# Support Vector Machine (SVM) Pipeline
svm_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", SVC(probability=True))
])

# Section 9: Hyperparameter Tuning with GridSearchCV

# Logistic Regression Hyperparameters
param_grid_lr = {
    "classifier__penalty": ["l1", "l2"],
    "classifier__solver": ['liblinear', 'saga']
}

# Random Forest Hyperparameters
param_grid_rf = {
    "classifier__n_estimators": [50, 100, 200],
    "classifier__criterion": ["gini", "entropy"],
    "classifier__bootstrap": [True, False],
    "classifier__max_features": ["sqrt"]
}

# SVM Hyperparameters
param_grid_svm = {
    "classifier__C": [0.1, 1, 10],
    "classifier__kernel": ["linear", "rbf"]
}

# Create GridSearchCV for each model
grid_lr = GridSearchCV(log_reg_pipeline, param_grid_lr, cv=5, n_jobs=-1, scoring="accuracy")
grid_rf = GridSearchCV(rf_pipeline, param_grid_rf, cv=5, n_jobs=-1, scoring="accuracy")
grid_svm = GridSearchCV(svm_pipeline, param_grid_svm, cv=5, n_jobs=-1, scoring="accuracy")

# Fit models with GridSearchCV
grid_lr.fit(X_train, y_train)
grid_rf.fit(X_train, y_train)
grid_svm.fit(X_train, y_train)

# Best parameters found
print("Best Parameters for Logistic Regression:", grid_lr.best_params_)
print("Best Parameters for Random Forest:", grid_rf.best_params_)
print("Best Parameters for SVM:", grid_svm.best_params_)

# Section 10: Model Evaluation & ROC Curve

# Evaluate Logistic Regression
y_pred_lr = grid_lr.predict(X_test)
fpr_lr, tpr_lr, _ = roc_curve(y_test, grid_lr.predict_proba(X_test)[:, 1])
roc_auc_lr = auc(fpr_lr, tpr_lr)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))

# Evaluate Random Forest
y_pred_rf = grid_rf.predict(X_test)
fpr_rf, tpr_rf, _ = roc_curve(y_test, grid_rf.predict_proba(X_test)[:, 1])
roc_auc_rf = auc(fpr_rf, tpr_rf)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

# Evaluate SVM
y_pred_svm = grid_svm.predict(X_test)
fpr_svm, tpr_svm, _ = roc_curve(y_test, grid_svm.predict_proba(X_test)[:, 1])
roc_auc_svm = auc(fpr_svm, tpr_svm)
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))

# Plot ROC Curves
plt.figure(figsize=(8, 6))
plt.plot(fpr_lr, tpr_lr, color='blue', lw=2, label='Logistic Regression (AUC = %0.2f)' % roc_auc_lr)
plt.plot(fpr_rf, tpr_rf, color='green', lw=2, label='Random Forest (AUC = %0.2f)' % roc_auc_rf)
plt.plot(fpr_svm, tpr_svm, color='red', lw=2, label='SVM (AUC = %0.2f)' % roc_auc_svm)
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curves')
plt.legend(loc="lower right")
plt.show()

# Section 11: Save Models

import joblib

# Save the best models
joblib.dump(grid_lr.best_estimator_, "model_lr.pkl")
joblib.dump(grid_rf.best_estimator_, "model_rf.pkl")
joblib.dump(grid_svm.best_estimator_, "model_svm.pkl")
joblib.dump(preprocessor, "preprocessor.pkl")

print("✅ Models and preprocessor saved successfully.")




!pip install dash dash-bootstrap-components
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import joblib
import os

# Step 1: Load dataset (replace with your actual dataset)
df = pd.read_csv('heart_disease_data.csv')  # Replace this with the actual path to your dataset

# Step 2: Feature-target split (Assuming 'target' is the column to predict)
X = df.drop('target', axis=1)  # Features (input variables)
y = df['target']  # Target variable (1 = Heart disease, 0 = No heart disease)

# Step 3: Handle missing values (optional, drop rows with missing values in this case)
df = df.dropna()

# Step 4: Train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Scale the features (important for models like Logistic Regression and SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Train models
model_lr = LogisticRegression()
model_rf = RandomForestClassifier()
model_svm = SVC(probability=True)

# Train the models
model_lr.fit(X_train_scaled, y_train)
model_rf.fit(X_train_scaled, y_train)
model_svm.fit(X_train_scaled, y_train)

# Step 7: Create a directory to save models (if it doesn't exist)
os.makedirs('models', exist_ok=True)

# Step 8: Save models and scaler
joblib.dump(model_lr, 'models/model_lr.pkl')
joblib.dump(model_rf, 'models/model_rf.pkl')
joblib.dump(model_svm, 'models/model_svm.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

print("✅ Models and scaler saved successfully.")

import joblib
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

server = app.server

# Load models and scaler
model_lr = joblib.load('models/model_lr.pkl')
model_rf = joblib.load('models/model_rf.pkl')
model_svm = joblib.load('models/model_svm.pkl')
scaler = joblib.load('models/scaler.pkl')

# Feature abbreviation to full name mapping
feature_labels = {
    'age': 'Age',
    'sex': 'Sex',
    'cp': 'Chest Pain Type',
    'trestbps': 'Resting Blood Pressure',
    'chol': 'Serum Cholesterol',
    'fbs': 'Fasting Blood Sugar',
    'restecg': 'Resting ECG Results',
    'thalach': 'Max Heart Rate Achieved',
    'exang': 'Exercise Induced Angina',
    'oldpeak': 'ST Depression (Oldpeak)',
    'slope': 'Slope of ST Segment',
    'ca': 'Major Vessels Colored',
    'thal': 'Thalassemia'
}

# Feature order
feature_names = list(feature_labels.keys())

# Input component generator
def get_input_component(feature):
    dropdown_options = {
        'sex': [{"label": "Male", "value": 1}, {"label": "Female", "value": 0}],
        'cp': [{"label": "Typical Angina", "value": 1},
               {"label": "Atypical Angina", "value": 2},
               {"label": "Non-anginal Pain", "value": 3},
               {"label": "Asymptomatic", "value": 0}],
        'fbs': [{"label": "True", "value": 1}, {"label": "False", "value": 0}],
        'restecg': [{"label": "Normal", "value": 1},
                    {"label": "ST-T Abnormality", "value": 2},
                    {"label": "Hypertrophy", "value": 0}],
        'exang': [{"label": "Yes", "value": 1}, {"label": "No", "value": 0}],
        'slope': [{"label": "Upsloping", "value": 2},
                  {"label": "Flat", "value": 1},
                  {"label": "Downsloping", "value": 0}],
        'thal': [{"label": "Normal", "value": 2},
                 {"label": "Fixed Defect", "value": 1},
                 {"label": "Reversible Defect", "value": 3}]
    }

    if feature in dropdown_options:
        return dcc.Dropdown(
            id=f"input-{feature}",
            options=dropdown_options[feature],
            className="form-control",
            placeholder="Select"
        )
    else:
        return dcc.Input(
            id=f"input-{feature}",
            type="number",
            step=0.01,
            className="form-control",
            placeholder="Enter value"
        )

# Initialize app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout
app.layout = dbc.Container([
    html.H1("Heart Disease Prediction", className="text-center my-4"),

    dbc.Row([
        dbc.Col([
            dbc.Label(f"{feature.upper()} - {feature_labels[feature]}"),
            get_input_component(feature)
        ], width=6) for feature in feature_names
    ]),

    html.Br(),

    dbc.Row([
        dbc.Col([
            dbc.Label("Select Model"),
            dcc.Dropdown(id="model-dropdown",
                         options=[
                             {"label": "Logistic Regression", "value": "lr"},
                             {"label": "Random Forest", "value": "rf"},
                             {"label": "SVM", "value": "svm"}
                         ],
                         value="lr",
                         className="form-control")
        ], width=6)
    ]),

    dbc.Button("Predict", id="predict-button", color="primary", className="mt-3"),

    html.Div(id="output", className="mt-4")
])

# Callback
@app.callback(
    Output("output", "children"),
    Input("predict-button", "n_clicks"),
    [State(f"input-{feature}", "value") for feature in feature_names] +
    [State("model-dropdown", "value")]
)
def predict(n_clicks, *args):
    if not n_clicks:
        return ""

    values = args[:-1]
    selected_model = args[-1]

    if None in values:
        return dbc.Alert("Please fill in all input values.", color="warning")

    try:
        input_data = scaler.transform([values])
        model = {"lr": model_lr, "rf": model_rf, "svm": model_svm}.get(selected_model)

        if not model:
            return dbc.Alert("Invalid model selected.", color="danger")

        prediction = model.predict(input_data)[0]
        result_text = "Heart Disease" if prediction == 1 else "No Heart Disease"
        result_color = "danger" if prediction == 1 else "success"

        return dbc.Alert(f"Prediction: {result_text}", color=result_color)

    except Exception as e:
        return dbc.Alert(f"Error: {e}", color="danger")

# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=8051, host='0.0.0.0')
