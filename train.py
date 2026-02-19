import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("healthcare-dataset-stroke-data.csv")

# Fill missing BMI
df['bmi'] = df['bmi'].fillna(df['bmi'].mean())

# Select features
features = [
    'gender',
    'age',
    'hypertension',
    'heart_disease',
    'ever_married',
    'work_type',
    'Residence_type',
    'avg_glucose_level',
    'bmi',
    'smoking_status'
]

X = df[features]
y = df['stroke']

# Convert categorical columns
X = pd.get_dummies(X)

# Save column names
joblib.dump(X.columns, "columns.pkl")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train model
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train_scaled, y_train)

# Save model and scaler
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model trained and saved successfully!")
