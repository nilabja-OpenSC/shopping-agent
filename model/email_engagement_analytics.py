#XGBoost
#Goal:
#Predict whether a user will complete a purchase after receiving a cart reminder email.

#What the script does:
#Loads the transaction_history.csv dataset.
#Encodes the payment_method column.
#Cleans and converts email_sent, email_clicked, and completed_from_email to integers.
#Prepares features (X) and target (y) where y = completed_from_email.
#Splits the data into training and testing sets.
#Trains an XGBoost classifier.
#Evaluates the model and prints a classification report.
#Saves the model as email_engagement_model.pkl.


df = pd.read_csv("transaction_history.csv")
df['email_sent'] = df['email_sent'].astype(int)
df['email_clicked'] = df['email_clicked'].astype(int)
df['completed_from_email'] = df['completed_from_email'].astype(int)
df['payment_method'] = LabelEncoder().fit_transform(df['payment_method'].astype(str))

X = df.drop(columns=['completed_from_email', 'user_id', 'product_id', 'purchase_date'])
y = df['completed_from_email']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = xgb.XGBClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Email Engagement Prediction Report:")
print(classification_report(y_test, y_pred))

joblib.dump(model, "email_engagement_model.pkl")
