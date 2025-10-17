import pandas as pd
import yaml
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load parameters from params.yaml
params = yaml.safe_load(open("params.yaml"))

# Load dataset
data_path = params["data"]["path"]
data = pd.read_csv(data_path)

# Select features and target
X = data[params["data"]["features"]].copy()
y = data[params["data"]["target"]]

# --- Data Cleaning ---
# Convert 'area' from strings like "9.6 Marla" to numeric values
def convert_area(value):
    if isinstance(value, str):
        value = value.lower().strip()
        try:
            num = float(value.split()[0])  # extract numeric part
        except:
            return None
        # Convert local area units to square feet (optional but consistent)
        if "marla" in value:
            return num * 272.25
        elif "kanal" in value:
            return num * 5445
        else:
            return num
    return value

# Apply cleaning on area column
if "area" in X.columns:
    X["area"] = X["area"].apply(convert_area)

# Fill missing values with 0
X = X.fillna(0)
y = y.fillna(0)

# Convert categorical variables (like city) to numeric using one-hot encoding
X = pd.get_dummies(X, drop_first=True)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=params["test_size"], random_state=params["random_state"]
)

# Train model
if params["model"]["type"] == "LinearRegression":
    model = LinearRegression(fit_intercept=params["model"]["fit_intercept"])
else:
    raise ValueError("Unsupported model type specified in params.yaml")

model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Save metrics
metrics = {"mae": mae, "mse": mse, "r2": r2}
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# Save model
joblib.dump(model, "models/model.joblib")

# Save model columns
joblib.dump(X.columns.tolist(), "models/model_columns.joblib")
print("✅ Training complete. Model and metrics saved.")