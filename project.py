import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import missingno

from scipy.stats.mstats import winsorize
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, RocCurveDisplay
import joblib

warnings.filterwarnings("ignore")

# Set random seed for reproducibility
random_state = 42

# Load dataset
df = pd.read_csv('heart_disease_data.csv')

# Rename columns
df.columns = [
    "age", "sex", "cp", "trtbps", "chol", "fbs", "rest_ecg", "thalach",
    "exang", "oldpeak", "slope", "ca", "thal", "target"
]

# Dataset overview
print("Dataset Shape:", df.shape)
print(df.info())
print("Missing Values:\n", df.isnull().sum())

# Section 2: Handling Missing Data & Basic EDA

# Visualize missing data
missingno.bar(df, color="blue")
plt.title("Missing Values in Dataset")
plt.show()

# Replace 0 in 'thal' with NaN, then fill with mode
df['thal'] = df['thal'].replace(0, np.nan)
df['thal'].fillna(df['thal'].mode()[0], inplace=True)
df['thal'] = pd.to_numeric(df['thal'], downcast='integer')

# Section 3: Visualizations

numeric_features = ["age", "trtbps", "chol", "thalach", "oldpeak"]
categorical_features = ["sex", "cp", "fbs", "rest_ecg", "exang", "slope", "ca", "thal", "target"]

# Distribution plots
for feature in numeric_features:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[feature], kde=True, bins=20)
    plt.title(f"Distribution of {feature}")
    plt.tight_layout()
    plt.show()

# Count plots
for feature in categorical_features:
    plt.figure(figsize=(8, 4))
    sns.countplot(x=feature, data=df, hue="target")
    plt.title(f"{feature} vs Target")
    plt.tight_layout()
    plt.show()

# Section 4: Outlier Detection and Treatment

df["trtbps"] = winsorize(df["trtbps"], limits=(0, 0.01))
df["oldpeak"] = winsorize(df["oldpeak"], limits=(0, 0.01))

# Section 5: Feature Engineering & Scaling

df["oldpeak_sqrt"] = np.sqrt(df["oldpeak"])
df.drop(["chol", "fbs", "rest_ecg"], axis=1, inplace=True)

# One-hot encoding
categorical_to_encode = ["sex", "cp", "exang", "slope", "ca", "thal"]
df_encoded = pd.get_dummies(df, columns=categorical_to_encode, drop_first=True)

# Scaling
features_to_scale = ["age", "thalach", "trtbps", "oldpeak_sqrt"]
scaler = RobustScaler()
df_encoded[features_to_scale] = scaler.fit_transform(df_encoded[features_to_scale])

# Section 6: Train & Evaluate Multiple Models

# Features and target
X = df_encoded.drop("target", axis=1)
y = df_encoded["target"]

# Save feature names
joblib.dump(X.columns.tolist(), "feature_columns.pkl")

# Split data with a fixed random_state
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

# Define models with random_state set for reproducibility
models = {
    "Logistic Regression": LogisticRegression(random_state=random_state),
    "Support Vector Machine": SVC(probability=True, random_state=random_state),
    "Random Forest": RandomForestClassifier(random_state=random_state)
}

best_models = {}

# Train & evaluate
for name, model in models.items():
    print(f"\nTraining: {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy:.4f}")
    print(f"{name} Classification Report:\n{classification_report(y_test, y_pred)}")

    # ROC Curve
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title(f"{name} - ROC Curve")
    plt.show()

    # Save the trained model
    best_models[name] = model

    # Cross-validation
    cv_score = cross_val_score(model, X_train, y_train, cv=10).mean()
    print(f"{name} Cross-Validation Accuracy: {cv_score:.4f}")

# Section 7: Hyperparameter Tuning

# Logistic Regression
param_grid_lr = {
    "penalty": ["l1", "l2"],
    "solver": ["liblinear", "saga"]
}
grid_lr = GridSearchCV(LogisticRegression(random_state=random_state), param_grid=param_grid_lr, cv=5)
grid_lr.fit(X_train, y_train)
print("\nBest Params (Logistic Regression):", grid_lr.best_params_)

# Random Forest
param_grid_rf = {
    "n_estimators": [50, 100, 200],
    "criterion": ["gini", "entropy"],
    "bootstrap": [True, False],
    "max_features": ["sqrt"]
}
grid_rf = GridSearchCV(RandomForestClassifier(random_state=random_state), param_grid=param_grid_rf, cv=5)
grid_rf.fit(X_train, y_train)
print("Best Params (Random Forest):", grid_rf.best_params_)

# SVM
param_grid_svm = {
    "C": [0.1, 1, 10],
    "kernel": ["linear", "rbf"],
    "gamma": ["scale", "auto"]
}
grid_svm = GridSearchCV(SVC(probability=True, random_state=random_state), param_grid=param_grid_svm, cv=5)
grid_svm.fit(X_train, y_train)
print("Best Params (SVM):", grid_svm.best_params_)

# Section 8: Save Tuned Models

joblib.dump(grid_lr.best_estimator_, "model_logistic.pkl")
joblib.dump(grid_rf.best_estimator_, "model_random_forest.pkl")
joblib.dump(grid_svm.best_estimator_, "model_svm.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\n✅ All models and scaler saved successfully.")




import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
import joblib
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from dash.dependencies import Input, Output, State
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv('heart_disease_data.csv')  # Replace with your actual dataset

# Feature-target split
X = df.drop('target', axis=1)
y = df['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
model_rf = RandomForestClassifier()
model_rf.fit(X_train_scaled, y_train)

# Initialize app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

server = app.server
# Layout
app.layout = dbc.Container([
    html.H1("Heart Disease Prediction", className="text-center my-4"),

    dbc.Row([
        dbc.Col([
            dbc.Label("Age"),
            dcc.Input(id="input-age", type="number", placeholder="Enter age", className="form-control")
        ], width=6),
        dbc.Col([
            dbc.Label("Sex"),
            dcc.Dropdown(
                id="input-sex",
                options=[{"label": "Male", "value": 1}, {"label": "Female", "value": 0}],
                className="form-control",
                placeholder="Select sex"
            )
        ], width=6),
        # Add more input fields for other features...
    ]),

    dbc.Button("Predict", id="predict-button", color="primary", className="mt-3"),

    html.Div(id="prediction-output", className="mt-4"),

    # ROC Curve
    dcc.Graph(id="roc-curve"),

    # Feature Importance
    dcc.Graph(id="feature-importance"),

    # Confusion Matrix
    dcc.Graph(id="confusion-matrix")
])

# Callback for prediction and graphs
@app.callback(
    [Output("prediction-output", "children"),
     Output("roc-curve", "figure"),
     Output("feature-importance", "figure"),
     Output("confusion-matrix", "figure")],
    Input("predict-button", "n_clicks"),
    [State("input-age", "value"),
     State("input-sex", "value")]
    # Add more states for other inputs...
)
def update_output(n_clicks, age, sex):
    if not n_clicks:
        return "", go.Figure(), go.Figure(), go.Figure()

    # Prepare input data
    input_data = np.array([[age, sex]])  # Add other features here
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model_rf.predict(input_scaled)[0]
    prediction_text = "Heart Disease" if prediction == 1 else "No Heart Disease"

    # ROC Curve
    y_pred_prob = model_rf.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    roc_fig = go.Figure()
    roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"ROC curve (AUC = {roc_auc:.2f})"))
    roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random classifier', line=dict(dash='dash')))
    roc_fig.update_layout(title="Receiver Operating Characteristic Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")

    # Feature Importance
    importances = model_rf.feature_importances_
    feature_names = X.columns
    fi_fig = go.Figure()
    fi_fig.add_trace(go.Bar(x=feature_names, y=importances))
    fi_fig.update_layout(title="Feature Importance", xaxis_title="Feature", yaxis_title="Importance")

    # Confusion Matrix
    cm = confusion_matrix(y_test, model_rf.predict(X_test_scaled))
    cm_fig = go.Figure()
    cm_fig.add_trace(go.Heatmap(z=cm, x=['No Heart Disease', 'Heart Disease'], y=['No Heart Disease', 'Heart Disease'], colorscale='Viridis'))
    cm_fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual")

    return f"Prediction: {prediction_text}", roc_fig, fi_fig, cm_fig

if __name__ == '__main__':
    app.run(debug=True, port=8051, host='0.0.0.0')
