import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv('data/movies.csv')
ratings = pd.read_csv('data/ratings.csv')

movies['title'] = movies['title'].apply(lambda x: ' '.join(x.split(' ')[:-1]))
# duplicate movie titles
duplicate_movies = movies.groupby('title').filter(lambda x: len(x) >= 2)
duplicate_movies = duplicate_movies[['movieId', 'title']]

duplicate_ids = duplicate_movies['movieId'].values.tolist()

# review on duplicate movies
duplicate_ratings = pd.DataFrame(ratings[ratings['movieId'].isin(
    duplicate_ids)]['movieId'].value_counts()).reset_index()
duplicate_ratings.columns = ['movieId', 'count']

duplicated_df = pd.merge(duplicate_movies, duplicate_ratings, on='movieId')
duplicated_df.sort_values(by=['title', 'count'], ascending=[True, False])

# Removing duplicated ids with low review count
low_ids = duplicated_df.drop_duplicates(
    subset='title', keep='last', inplace=False)['movieId']
movies = movies.loc[~movies['movieId'].isin(low_ids)]
ratings = ratings.loc[~ratings['movieId'].isin(low_ids)]

df = pd.merge(ratings, movies, on='movieId')
pivot_df = pd.pivot_table(df, index='title', columns=[
                          'userId'], values='rating')
movie = 'Toy Story'


# Item based CF
def item_based_cf(pivot_df, movie):
    pivot_df = pivot_df.fillna(0)
    sparse_pivot = sparse.csr_matrix(pivot_df)
    recommender = cosine_similarity(sparse_pivot)
    recommender_df = pd.DataFrame(
        recommender, columns=pivot_df.index, index=pivot_df.index)
    # print(recommender_df.index.str.upper().tolist())
    cosine_df = pd.DataFrame(
        recommender_df[movie].sort_values(ascending=False))[1:6]
    cosine_df.reset_index(inplace=True)
    print(cosine_df['title'].values.tolist())


# User based CF
def user_based_cf(pivot_df, userid):
    pivot_df = pivot_df.fillna(0)
    sparse_pivot = sparse.csr_matrix(pivot_df)
    recommender = cosine_similarity(sparse_pivot)
    recommender_df = pd.DataFrame(
        recommender, columns=pivot_df.index, index=pivot_df.index)
    cosine_df = pd.DataFrame(
        recommender_df[userid].sort_values(ascending=False))[1:6]
    cosine_df.reset_index(inplace=True)
    print(cosine_df['userId'].values.tolist())


# item_based_cf(pivot_df, movie)
user_based_cf(pivot_df.T, 1)
