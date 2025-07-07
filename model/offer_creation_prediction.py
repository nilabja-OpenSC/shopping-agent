#Trains a classifier to predict whether an offer should be created (offer_created = 1) based on session context and user intent.
#Uses intent_score as a key feature to guide the decision.
#Outputs predictions and evaluation metrics.
#Saves the updated dataset with a new column: predicted_offer_created.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier

# Load the dataset
df = pd.read_csv("cart_abandonment.csv")

# Define features and target
features = [
    'session_duration', 'num_items_in_cart', 'cart_value',
    'time_of_day', 'day_of_week', 'intent_score',
    'offer_type', 'offer_response'
]
target = 'offer_created'

# Encode categorical features
categorical_cols = ['time_of_day', 'day_of_week', 'offer_type', 'offer_response']
for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

X = df[features]
y = df[target]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost Classifier
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Predict offer_created for all sessions
df['predicted_offer_created'] = model.predict(X)

# Save the updated dataset
df.to_csv("cart_abandonment_with_offer_created.csv", index=False)
