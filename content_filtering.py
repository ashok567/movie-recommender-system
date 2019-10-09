import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Data Preprocessing
df = pd.read_csv("data/movie_dataset.csv")

features = ['keywords', 'cast', 'genres', 'director']
df[features] = df[features].fillna('')

df['title'] = df['title'].str.upper()


def combine_features(row):
    try:
        feat = row['keywords'] + " "+row['cast'] + \
            " "+row['genres'] + " "+row['director']
        return feat
    except Exception as e:
        print(e)


df['combined_features'] = df.apply(combine_features, axis=1)


# Create count matrix and compute the Cosine Similarity
def similarity(movie_name):
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(df["combined_features"])
    cosine_sim = cosine_similarity(count_matrix)
    movie_index = df[df['title'] == movie_name]['index'].values[0]
    similar_movies = list(enumerate(cosine_sim[movie_index]))
    similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)
    return similar_movies


movie_name = input("Enter a movie name\n").upper()
if movie_name not in df['title'].tolist():
    print("The movie is not listed in our database")
else:
    print("Also add these movies to your watch list: ")
    similar_movies_list = similarity(movie_name)
    i = 0
    for element in similar_movies_list[1:]:
        if i < 10:
            print(df[df.index == element[0]]["title"].values[0])
        else:
            break
        i = i+1
