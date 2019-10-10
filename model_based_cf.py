# Model-based CF is based on matrix factorization using SVD to factorize the matrix.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.sparse.linalg import svds

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

sparsity = round(1.0-len(df)/(users_counts*movies_counts), 3)*100
print("sparsity: "+str(sparsity)+"%")

# Singular value decomposition(SVD) --> matrix factorization(Unsupervise Learning)
u, s, vt = svds(train_data_matrix, k=20)
s_diag = np.diag(s)
# X(m*n) = USV.T (S(m*n) is a diagonal matrix, features matrix U(users-->m*m) & V(movies)(n*n))
x_pred = np.dot(np.dot(u, s_diag), vt)
print(x_pred)
