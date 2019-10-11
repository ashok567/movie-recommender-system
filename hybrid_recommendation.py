import content_filtering as cf
import memory_based_cf as mcf

user_id = 262
movie_title = input("Enter a movie name:\n")

item_movie_list = mcf.item_based_cf(user_id, movie_title.upper())
if len(item_movie_list) == 0:
    print("Mentioned movie is not our in database")
else:
    similar_movies_list = cf.content_similarity(movie_title.upper())
    print("Films you might enjoy based that you watched {0}:".format(
        movie_title))
    i = 1
    for movie in item_movie_list:
        if i > 5:
            break
        elif movie in similar_movies_list:
            i += 1
            print(movie)
