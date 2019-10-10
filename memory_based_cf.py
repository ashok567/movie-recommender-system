import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv('data/movies.csv')
ratings = pd.read_csv('data/ratings.csv')

movies['title'] = movies['title'].apply(lambda x: ' '.join(x.split(' ')[:-1]))
movies['title'] = movies['title'].str.upper()
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


def memory_cf(pivot_df, subject):
    pivot_df = pivot_df.fillna(0)
    sparse_pivot = sparse.csr_matrix(pivot_df)
    recommender = cosine_similarity(sparse_pivot)
    recommender_df = pd.DataFrame(
        recommender, columns=pivot_df.index, index=pivot_df.index)
    cosine_df = pd.DataFrame(
        recommender_df[subject].sort_values(ascending=False))[1:6]
    cosine_df.reset_index(inplace=True)
    return cosine_df


# Item based CF
movie_title = 'Toy Story'
if movie_title.upper() not in movies['title'].values.tolist():
    print("Mentioned movie is not in our database")
else:
    item_based_cf = memory_cf(pivot_df, movie_title.upper())
    print("Films you might enjoy based that you watched {0}".format(movie_title))
    for movie in item_based_cf['title'].values.tolist():
        print(movie)

# User based CF
user = 1
if user not in ratings['userId'].values.tolist():
    print("Mentioned userid is not in our database")
else:
    user_based_cf = memory_cf(pivot_df.T, user)
    print(user_based_cf['userId'].values.tolist())
