import pandas as pd
import preprocessing
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

df, movies, ratings = preprocessing.data()
pivot_df = pd.pivot_table(df, index='title', columns=[
                          'userId'], values='rating')


def memory_cf(pivot_df, subject):
    pivot_df = pivot_df.fillna(0)
    sparse_pivot = sparse.csr_matrix(pivot_df)
    recommender = cosine_similarity(sparse_pivot)
    recommender_df = pd.DataFrame(
        recommender, columns=pivot_df.index, index=pivot_df.index)
    cosine_df = pd.DataFrame(
        recommender_df[subject].sort_values(ascending=False))
    cosine_df.reset_index(inplace=True)
    return cosine_df


user_id = 262
movie_title = 'Toy Story'

# Item based CF
if movie_title.upper() not in movies['title'].values.tolist():
    print("Mentioned movie is not in our database")
else:
    already_watched = df[df['userId'] == user_id]['title'].values.tolist()
    item_based_cf = memory_cf(pivot_df, movie_title.upper())
    print("Films you might enjoy based that you watched {0}:".format(
        movie_title))
    for movie in item_based_cf['title'].values.tolist()[1:11]:
        if movie not in already_watched:
            print(movie)

print("-------------------------------------------------------------")

# User based CF
if user_id not in ratings['userId'].values.tolist():
    print("Mentioned userid is not in our database")
else:
    already_watched = df[df['userId'] == user_id]['title'].values.tolist()
    similar_users = memory_cf(pivot_df.T, user_id)
    similar_users_list = similar_users['userId'].values.tolist()[1:11]
    similar_user_df = pivot_df[similar_users_list].fillna(0)
    similar_user_df['mean_rating'] = similar_user_df[similar_users_list].mean(
        numeric_only=True, axis=1)
    similar_user_df = similar_user_df.sort_values(
        'mean_rating', ascending=False)
    print("Flims reccomended for you:")
    for movie in similar_user_df.index.tolist()[:10]:
        if movie not in already_watched:
            print(movie)
