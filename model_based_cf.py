# Model-based CF is based on matrix factorization using SVD for user based cf
import pandas as pd
import numpy as np
import preprocessing
from scipy import sparse
from scipy.sparse.linalg import svds

user_id = 2

# Get all basic data
df, movies, ratings = preprocessing.data()
pivot_df = pd.pivot_table(df, index='title', columns=[
    'userId'], values='rating').T
pivot_df = pivot_df.fillna(0)
sparse_pivot = sparse.csr_matrix(pivot_df)

sparsity = round(
    1.0-len(df)/(sparse_pivot.shape[0]*sparse_pivot.shape[1]), 3)*100
print("sparsity:", str(sparsity)+"%")

# Singular value decomposition(SVD) --> Matrix Factorization
u, s, vt = svds(sparse_pivot, k=20)
s_diag = np.diag(s)

# S(m*n) is a diagonal matrix, Features Matrix U(users-->m*m) & V(movies)(n*n)
preds = np.dot(np.dot(u, s_diag), vt)  # X(m*n) = USV.T

# Get user specific data
preds_df = pd.DataFrame(preds, columns=pivot_df.columns, index=pivot_df.index)
sorted_user_preds = preds_df.loc[user_id].sort_values(ascending=False)

# Watched and unwatched movies
already_watched = ratings[ratings['userId'] == user_id]
unwatched_movies = movies[~movies['movieId'].isin(already_watched['movieId'])]

# top 5 movie recommendation
recommendation_df = unwatched_movies.merge(
    pd.DataFrame(sorted_user_preds), how='left', on='title')
recommendation_df = recommendation_df.rename(columns={user_id: 'predictions'})
recommendation_df = recommendation_df.sort_values(
    'predictions', ascending=False)
recommendation_list = recommendation_df['title'].values.tolist()[:5]

print("Flims reccomended for you:")
for movie in recommendation_list:
    print(movie)
