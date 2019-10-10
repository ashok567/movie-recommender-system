# Model-based CF is based on matrix factorization using SVD
import pandas as pd
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds

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
pivot_df = pivot_df.fillna(0)
sparse_pivot = sparse.csr_matrix(pivot_df)

sparsity = round(
    1.0-len(df)/(sparse_pivot.shape[0]*sparse_pivot.shape[1]), 3)*100
print("shape:", sparse_pivot.shape)
print("sparsity:", str(sparsity)+"%")

# Singular value decomposition(SVD) --> Matrix Factorization
u, s, vt = svds(sparse_pivot, k=20)
s_diag = np.diag(s)
# S(m*n) is a diagonal matrix, Features Matrix U(users-->m*m) & V(movies)(n*n)
recommender = np.dot(np.dot(u, s_diag), vt)  # X(m*n) = USV.T
