#LightFM
#Goal:
#Recommend products to users based on their past interactions.

#What the script does:
#Loads user_interactions.csv and filters for meaningful events (add_to_cart, purchase).
#Extracts unique users and products.
#Initializes a LightFM Dataset and fits it with users and items.
#Builds an interaction matrix from user-product pairs.
#Trains a LightFM model using WARP loss (good for implicit feedback).
#Saves the model and the dataset mapping for future use.


from lightfm import LightFM
from lightfm.data import Dataset

interactions_df = pd.read_csv("user_interactions.csv")
interactions_df = interactions_df[interactions_df['event_type'].isin(['add_to_cart', 'purchase'])]

users = interactions_df['user_id'].unique()
items = interactions_df['product_id'].unique()

dataset = Dataset()
dataset.fit(users, items)

interactions = list(zip(interactions_df['user_id'], interactions_df['product_id']))
(interactions_matrix, _) = dataset.build_interactions(interactions)

model = LightFM(loss='warp')
model.fit(interactions_matrix, epochs=10, num_threads=2)

joblib.dump(model, "lightfm_recommender.pkl")
joblib.dump(dataset.mapping(), "lightfm_mapping.pkl")
