from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd

# Load and clean data
data = fetch_openml("autoMPG", version=1, as_frame=True)
X, y = data.data, data.target
X = X.apply(pd.to_numeric, errors='coerce')
clean_data = pd.concat([X, y], axis=1).dropna()
X_clean = clean_data.iloc[:, :-1]
y_clean = clean_data.iloc[:, -1]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)

# Train models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

print("Model Performance:")
print("-" * 30)
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{name}: MAE={mae:.2f}, R²={r2:.2f}")

# Test predictions
print("\nSample Predictions:")
sample_indices = [0, 1, 2]
best_model = models['Random Forest']
for i in sample_indices:
    actual = y_test.iloc[i]
    predicted = best_model.predict([X_test.iloc[i]])[0]
    print(f"Actual: {actual:.1f} MPG, Predicted: {predicted:.1f} MPG")