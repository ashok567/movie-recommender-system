import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def combine_features(row):
    try:
        feat = row['keywords'] + " "+row['cast'] + \
            " "+row['genres'] + " "+row['director']
        return feat
    except Exception as e:
        print(e)


# Compute the Cosine Similarity
def content_similarity(movie_name):
    similar_movies_list = []
    df = pd.read_csv("data/movie_dataset.csv")
    features = ['keywords', 'cast', 'genres', 'director']
    df[features] = df[features].fillna('')
    df['title'] = df['title'].str.upper()
    df['combined_features'] = df.apply(combine_features, axis=1)
    if movie_name in df['title'].tolist():
        cv = CountVectorizer()
        count_matrix = cv.fit_transform(df["combined_features"])
        cosine_sim = cosine_similarity(count_matrix)
        movie_index = df[df['title'] == movie_name]['index'].values[0]
        similar_movies = list(enumerate(cosine_sim[movie_index]))
        similar_movies = sorted(
            similar_movies, key=lambda x: x[1], reverse=True)
        for element in similar_movies[1:50]:
            similar_movies_list.append(
                df[df.index == element[0]]["title"].values[0])
    return similar_movies_list


movie_name = input("Enter a movie name\n").upper()
similar_movies_list = content_similarity(movie_name)
if len(similar_movies_list) > 0:
    print("Also, add these movies to your watch list: ")
    for movie in similar_movies_list[:6]:
        print(movie)
else:
    print("Mentioned movie is not our in database")
