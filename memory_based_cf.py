import pandas as pd
import preprocessing
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

user_id = 262
movie_title = 'Toy Story'


def get_cosine_df(pivot_df, subject):
    pivot_df = pivot_df.fillna(0)
    sparse_pivot = sparse.csr_matrix(pivot_df)
    recommender = cosine_similarity(sparse_pivot)
    recommender_df = pd.DataFrame(
        recommender, columns=pivot_df.index, index=pivot_df.index)
    cosine_df = pd.DataFrame(
        recommender_df[subject].sort_values(ascending=False))
    cosine_df.reset_index(inplace=True)
    return cosine_df


# Item based CF
def item_based_cf(user_id, movie_title):
    df, movies, ratings = preprocessing.data()
    pivot_df = pd.pivot_table(df, index='title', columns=[
        'userId'], values='rating')
    item_movie_df = []
    if movie_title.upper() in movies['title'].values.tolist():
        already_watched = df[df['userId'] == user_id]['title'].values.tolist()
        item_based_cf = get_cosine_df(pivot_df, movie_title.upper())
        for movie in item_based_cf['title'].values.tolist()[1:]:
            if movie not in already_watched:
                item_movie_df.append(movie)
    return item_movie_df


# User based CF
def user_based_cf(user_id, movie_title):
    df, movies, ratings = preprocessing.data()
    pivot_df = pd.pivot_table(df, index='title', columns=[
        'userId'], values='rating')
    user_movie_df = []
    if user_id in ratings['userId'].values.tolist():
        already_watched = df[df['userId'] == user_id]['title'].values.tolist()
        similar_users = get_cosine_df(pivot_df.T, user_id)
        users_list = similar_users['userId'].values.tolist()[1:11]
        similar_user_df = pivot_df[users_list].fillna(0)
        similar_user_df['mean_rating'] = similar_user_df[users_list].mean(
            numeric_only=True, axis=1)
        similar_user_df = similar_user_df.sort_values(
            'mean_rating', ascending=False)
        for movie in similar_user_df.index.tolist():
            if movie not in already_watched:
                user_movie_df.append(movie)
    return user_movie_df


if __name__ == '__main__':
    item_movie_list = item_based_cf(user_id, movie_title)
    if len(item_movie_list) > 0:
        print("Films you might enjoy based that you watched {0}:".format(
            movie_title))
        for movie in item_movie_list[:6]:
            print(movie)
    else:
        print("Mentioned movie is not our in database")

    print("-----------------------------------------------")
    user_movie_list = user_based_cf(user_id, movie_title)
    if len(user_movie_list) > 0:
        print("Flims reccomended for you:")
        for movie in user_movie_list[:6]:
            print(movie)
    else:
        print("Mentioned user is not our in database")
