
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, roc_curve, auc
import joblib
import os
import plotly.graph_objects as go
import plotly.express as px
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output, State
import seaborn as sns
import matplotlib.pyplot as plt

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
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

server = app.server

# Layout
app.layout = dbc.Container([

    # Title
    html.H1("Heart Disease Prediction", className="text-center my-4"),

    # Input fields
    dbc.Row([
        dbc.Col([ 
            dbc.Label(f"{feature.upper()} - {feature_labels[feature]}"),
            get_input_component(feature)
        ], width=6) for feature in feature_names
    ]),

    html.Br(),

    # Model selection
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

    # Prediction Output
    html.Div(id="output", className="mt-4"),

    # ROC Curve and Feature Importance Graphs
    dcc.Graph(id='roc-curve'),
    dcc.Graph(id='feature-importance'),
    dcc.Graph(id='confusion-matrix')
])

# Callback to predict the result and generate graphs
@app.callback(
    [Output("output", "children"),
     Output('roc-curve', 'figure'),
     Output('feature-importance', 'figure'),
     Output('confusion-matrix', 'figure')],
    Input("predict-button", "n_clicks"),
    [State(f"input-{feature}", "value") for feature in feature_names] +
    [State("model-dropdown", "value")]
)
def predict(n_clicks, *args):
    if not n_clicks:
        return "", {}, {}, {}

    values = args[:-1]
    selected_model = args[-1]

    if None in values:
        return "Please fill in all input values.", {}, {}, {}

    # Scaling the input data
    input_data = scaler.transform([values])

    # Selecting the model
    model = {"lr": model_lr, "rf": model_rf, "svm": model_svm}.get(selected_model)
    if not model:
        return "Invalid model selected.", {}, {}, {}

    # Prediction
    prediction = model.predict(input_data)[0]
    result_text = "Heart Disease" if prediction == 1 else "No Heart Disease"
    result_color = "danger" if prediction == 1 else "success"

    # Confusion Matrix
    y_pred = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    cm_fig = px.imshow(cm, color_continuous_scale='Blues', title="Confusion Matrix",
                       labels={'x': 'Predicted', 'y': 'Actual'})

    # ROC Curve
    y_prob = model.predict_proba(X_test_scaled)[:, 1]  # Probability for class 1
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    roc_fig = go.Figure(data=go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve (area = {roc_auc:.2f})'))
    roc_fig.update_layout(title="Receiver Operating Characteristic (ROC) Curve",
                          xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')

    # Feature Importance (Random Forest only)
    if selected_model == 'rf':
        feature_importance = model.feature_importances_
        feature_importance_fig = px.bar(
            x=feature_names, y=feature_importance, title="Feature Importance",
            labels={'x': 'Features', 'y': 'Importance'}
        )
    else:
        feature_importance_fig = {}

    return dbc.Alert(f"Prediction: {result_text}", color=result_color), roc_fig, feature_importance_fig, cm_fig

# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=8051, host='0.0.0.0')
