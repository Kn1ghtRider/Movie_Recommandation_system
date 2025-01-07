import numpy
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity


movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

# merging two different dataset into one
movies = movies.merge(credits, on="title")

# filtering the appropiate data for our need
movies = movies[["movie_id", "title", "overview", "genres", "keywords", "cast", "crew"]]

# dropping missing data(in our dataset there is only 3 among 5000)
movies.dropna(inplace=True)


# function to get only the genre names
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):  # string of list to list
        L.append(i["name"])
    return L


movies["genres"] = movies["genres"].apply(convert)
movies["keywords"] = movies["keywords"].apply(convert)


def convertdict(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i["name"])
            counter += 1
        else:
            break
    return L


movies["cast"] = movies["cast"].apply(convertdict)


def director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i["job"] == "Director":
            L.append(i["name"])
            break
    return L


movies["crew"] = movies["crew"].apply(director)
movies["overview"] = movies["overview"].apply(lambda x: x.split())

movies["genres"] = movies["genres"].apply(lambda x: [i.replace(" ", "") for i in x])
movies["crew"] = movies["crew"].apply(lambda x: [i.replace(" ", "") for i in x])
movies["cast"] = movies["cast"].apply(lambda x: [i.replace(" ", "") for i in x])
movies["keywords"] = movies["keywords"].apply(lambda x: [i.replace(" ", "") for i in x])

movies["tags"] = (
    movies["overview"]
    + movies["genres"]
    + movies["keywords"]
    + movies["cast"]
    + movies["crew"]
)

# creating a new dataframe to hold only the important info
new_df = movies[["movie_id", "title", "tags"]]

# converting list in tags to string
new_df["tags"] = new_df["tags"].apply(lambda x: " ".join(x))

# using vectoriztion to get most used words
cv = CountVectorizer(max_features=5000, stop_words="english")
vectors = cv.fit_transform(new_df["tags"]).toarray()


# applying stem to covert similar words into one word
ps = PorterStemmer()


def stem(text):
    y = []

    for i in text.split():
        y.append(ps.stem(i))

    return " ".join(y)  # converting back to string


new_df["tags"] = new_df["tags"].apply(stem)

# getting cosine similarity
similarity = cosine_similarity(vectors)
print(similarity.shape)

#function to recommend top 5 closest movies
def recommend(movie):
    movie_index = new_df[new_df["title"] == movie].index[0] # getting the index of the movie
    distances = similarity[movie_index] # getting the similarity of the movie with all other movies
    # sorting the distances in descending order while keeping the index of the movie
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    for i in movies_list:
        print(new_df.iloc[i[0]].title)  

recommend("Batman")

import pickle
pickle.dump(new_df, open("movies.pkl", "wb"))
pickle.dump(similarity, open("similarity.pkl", "wb"))