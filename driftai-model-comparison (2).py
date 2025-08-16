import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Load the data
data = pd.read_excel('drift_racing_data.xlsx')

# Prepare the features and target
features = ['speed_km_h', 'steering_angle_deg', 'throttle_position_percent', 'brake_pressure_bar',
            'gear', 'engine_rpm', 'track_temperature_c', 'ambient_temperature_c', 'humidity_percent',
            'total_tire_pressure_psi', 'average_suspension_travel_mm', 'lateral_g_force',
            'longitudinal_g_force', 'yaw_rate_deg_s', 'boost_pressure_psi']

X = data[features]
y = data['drift_angle_deg']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Deep Neural Network Model
def create_dnn_model(input_shape):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

dnn_model = create_dnn_model(len(features))
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = dnn_model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

# Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate models
def evaluate_model(model, X, y, model_name, is_dnn=False):
    if is_dnn:
        predictions = model.predict(X).flatten()
    else:
        predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    r2 = r2_score(y, predictions)
    print(f"{model_name} - MSE: {mse:.4f}, R2: {r2:.4f}")
    return predictions

print("Validation Set Performance:")
dnn_val_pred = evaluate_model(dnn_model, X_val_scaled, y_val, "DNN", is_dnn=True)
rf_val_pred = evaluate_model(rf_model, X_val, y_val, "Random Forest")

print("\nTest Set Performance:")
dnn_test_pred = evaluate_model(dnn_model, X_test_scaled, y_test, "DNN", is_dnn=True)
rf_test_pred = evaluate_model(rf_model, X_test, y_test, "Random Forest")

# Visualize predictions
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(y_test, dnn_test_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Drift Angle")
plt.ylabel("Predicted Drift Angle")
plt.title("DNN: Actual vs Predicted Drift Angle")

plt.subplot(1, 2, 2)
plt.scatter(y_test, rf_test_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Drift Angle")
plt.ylabel("Predicted Drift Angle")
plt.title("Random Forest: Actual vs Predicted Drift Angle")

plt.tight_layout()
plt.show()

# Feature importance for Random Forest
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': rf_model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.bar(feature_importance['feature'], feature_importance['importance'])
plt.title("Random Forest: Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

print("\nTop 5 important features (Random Forest):")
print(feature_importance.head())
