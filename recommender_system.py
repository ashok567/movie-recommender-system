import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def get_title_from_index(index):
    return df[df.index == index]["title"].values[0]


def get_index_from_title(title):
    return df[df.title == title]["index"].values[0]


df = pd.read_csv("data/movie_dataset.csv")

features = ['keywords', 'cast', 'genres', 'director']

for feature in features:
    df[feature] = df[feature].fillna('')


def combine_features(row):
    try:
        feat = row['keywords'] + " "+row['cast'] + \
            " "+row["genres"] + " "+row["director"]
        return feat
    except Exception as e:
        print("Error:", e)


df["combined_features"] = df.apply(combine_features, axis=1)


# Create count matrix from this new combined column
cv = CountVectorizer()

count_matrix = cv.fit_transform(df["combined_features"])

# Compute the Cosine Similarity based on the count_matrix
cosine_sim = cosine_similarity(count_matrix)
movie_user_likes = "Avatar"

# Get index of this movie from its title
movie_index = get_index_from_title(movie_user_likes)

similar_movies = list(enumerate(cosine_sim[movie_index]))

similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)

i = 0
for element in similar_movies:
    if i < 50:
        print(get_title_from_index(element[0]))
    else:
        break
    i = i+1
