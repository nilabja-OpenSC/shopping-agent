
#XGBoost
#Goal:
#Predict whether a user will abandon their cart based on session behavior and offer-related features.

#What the script does:
#Loads the cart_abandonment.csv dataset.
#Encodes categorical features like time_of_day, day_of_week, offer_type, and offer_response using LabelEncoder.
#Prepares features (X) and target (y) where y = abandoned.
#Splits the data into training and testing sets.
#Trains an XGBoost classifier on the training data.
#Evaluates the model using a classification report (precision, recall, F1-score).
#Saves the trained model as cart_abandonment_model.pkl.





import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import xgboost as xgb
import joblib

df = pd.read_csv("cart_abandonment.csv")
for col in ['time_of_day', 'day_of_week', 'offer_type', 'offer_response']:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

X = df.drop(columns=['abandoned', 'user_id'])
y = df['abandoned']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = xgb.XGBClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Cart Abandonment Prediction Report:")
print(classification_report(y_test, y_pred))

joblib.dump(model, "cart_abandonment_model.pkl")
