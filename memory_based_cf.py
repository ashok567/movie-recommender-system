# Memory-based models are based on similarity between items or users using cosine-similarity.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

columns = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('data/u.data', sep='\t', names=columns)
titles_df = pd.read_csv('data/Movie_Id_Titles')
df = pd.merge(df, titles_df, on='item_id')

users_counts = df['user_id'].nunique()
movies_counts = df['item_id'].nunique()
print("m: "+str(users_counts), "& n: "+str(movies_counts))

# initialize the metrices
train_data, test_data = train_test_split(df, test_size=0.25)
train_data_matrix = np.zeros((users_counts, movies_counts))
test_data_matrix = np.zeros((users_counts, movies_counts))

for index, row in train_data.iterrows():
    train_data_matrix[row['user_id']-1, row['item_id']-1] = row['rating']

for index, row in test_data.iterrows():
    test_data_matrix[row['user_id']-1, row['item_id']-1] = row['rating']

# train_data_matrix --> users * items(m*n)
user_similarity = cosine_similarity(train_data_matrix)
# train_data_matrix --> items * users(n*m)
movie_similarity = cosine_similarity(train_data_matrix.T)


def predict(ratings, similarity, type1):
    if type1 == 'user':
        mean_user_rating = ratings.mean(axis=1)
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff)/np.array([np.abs(similarity).sum(axis=1)]).T
    elif type1 == 'movie':
        pred = ratings.dot(similarity)/np.array([np.abs(similarity).sum(axis=1)])
    return pred


user_prediction = predict(train_data_matrix, user_similarity, 'user')
item_prediction = predict(train_data_matrix, movie_similarity, 'movie')

print(item_prediction)
print(user_prediction)
