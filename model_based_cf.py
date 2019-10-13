# Model-based CF is based on matrix factorization using SVD
import pandas as pd
import numpy as np
import preprocessing
from scipy import sparse
from scipy.sparse.linalg import svds

movie_title = 'Toy Story'
df, movies, ratings = preprocessing.data()
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
recommender_df = pd.DataFrame(
        recommender, columns=pivot_df.columns, index=pivot_df.index)
cosine_df = pd.DataFrame(
    recommender_df[movie_title].sort_values(ascending=False))
cosine_df.reset_index(inplace=True)
