import pandas as pd


def data():
    movies = pd.read_csv('data/movies.csv')
    ratings = pd.read_csv('data/ratings.csv')

    movies['title'] = movies['title'].apply(
        lambda x: ' '.join(x.split(' ')[:-1]))
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
    return df, movies, ratings
