#Trains a regression model to estimate intent_score based on session behavior.
#Uses 1 - abandoned as a proxy for intent (1 = intent to purchase).
#Outputs predictions and evaluation metrics.
#Saves the updated dataset with a new column: predicted_intent_score.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

# Load the dataset
df = pd.read_csv("cart_abandonment.csv")

# Select features and target
features = [
    'session_duration', 'num_items_in_cart', 'cart_value',
    'time_of_day', 'day_of_week', 'offer_type', 'offer_response'
]

# Encode categorical features
categorical_cols = ['time_of_day', 'day_of_week', 'offer_type', 'offer_response']
for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

X = df[features]
y = 1 - df['abandoned']  # Simulated intent_score: 1 if purchased, 0 if abandoned

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost Regressor
model = XGBRegressor()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# Predict intent scores for all sessions
df['predicted_intent_score'] = model.predict(X)

# Save the updated dataset
df.to_csv("cart_abandonment_with_intent_score.csv", index=False)
