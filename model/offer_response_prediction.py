#XGBoost
#Goal:
#Predict how a user will respond to an offer (e.g., accept, ignore, decline).

#What the script does:
#Loads the same cart_abandonment.csv dataset.
#Encodes the same categorical features.
#Prepares features (X) and target (y) where y = offer_response.
#Splits the data into training and testing sets.
#Trains an XGBoost classifier to predict offer response.
#Evaluates the model and prints a classification report.
#Saves the model as offer_response_model.pkl.


df = pd.read_csv("cart_abandonment.csv")
for col in ['time_of_day', 'day_of_week', 'offer_type', 'offer_response']:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

X = df.drop(columns=['offer_response', 'user_id'])
y = df['offer_response']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = xgb.XGBClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Offer Response Prediction Report:")
print(classification_report(y_test, y_pred))

joblib.dump(model, "offer_response_model.pkl")
