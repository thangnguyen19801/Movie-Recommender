import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from ast import literal_eval
from surprise import Reader, Dataset, SVD, evaluate

''' Read data to dataframe '''
df1 = pd.read_csv("input/tmdb-movie-metadata/tmdb_5000_credits.csv")
df2 = pd.read_csv("input/tmdb-movie-metadata/tmdb_5000_movies.csv")

df1.columns = ["id", "title", "cast", "crew"]
df2 = df2.merge(df1, on=["id", "title"])
# print(df2.head(10))

# Recommend trending movies
C = df2["vote_average"].mean()
# print(C)

m = df2["vote_count"].quantile(0.9)
# print(m)

q_movies = df2.copy().loc[df2["vote_count"] >= m]
# print(q_movies.shape)

def weighted_rating(x, m=m, C=C):
    v = x["vote_count"]
    R = x["vote_average"]
    return (v*R + m*C)/(v+m)

q_movies["score"] = q_movies.apply(weighted_rating, axis=1)
q_movies = q_movies.sort_values("score", ascending=False)
# print(q_movies[["id", "title", "vote_average", "score"]].head(10))

''' Recommendation based on content '''

# print(df2["overview"].head(10))
tfidf = TfidfVectorizer(stop_words="english")
df2["overview"] = df2["overview"].fillna("")
tfidf_matrix = tfidf.fit_transform(df2["overview"])
#print(tfidf_matrix.shape)

# Cosine Similarity
cos_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
data = pd.Series(df2.index, index=df2["title"]).drop_duplicates()

# Recommender
def get_recommendations(title, cos_sim = cos_sim):
    idx = data[title]
    sim_scores = list(enumerate(cos_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [x[0] for x in sim_scores]
    
    #return df2["title"][movie_indices]
    return df2["title"][movie_indices]

# print(get_recommendations('The Dark Knight Rises'))

''' Recommendation based on director, cast, genres and keywords'''

features = ["cast", "crew", "keywords", "genres"]
for feature in features:
    df2[feature] = df2[feature].apply(literal_eval)

def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan

def get_list(x):
    if isinstance(x, list):
        names = [i["name"] for i in x]
        if(len(names) >= 3):
            names = names[:3]
        return names
    return []

df2["director"] = df2["crew"].apply(get_director)
features = ["cast", "keywords", "genres"]
for feature in features:
    df2[feature] = df2[feature].apply(get_list)
    
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        if(isinstance(x, str)):
            return str.lower(x.replace(" ", ""))
        else:
            return ""
       
features = ["director", "cast", "keywords", "genres"]
for feature in features:
    df2[feature] = df2[feature].apply(clean_data)
    
def create_data(x):
    return x["director"] + " " + " ".join(x["cast"]) + " " + " ".join(x["keywords"]) + " " + " ".join(x["genres"])

df2["data"] = df2.apply(create_data, axis=1)

# Create matrix for cosine_similiraty
count = CountVectorizer(stop_words="english")
count_matrix = count.fit_transform(df2["data"])
cos_sim2 = cosine_similarity(count_matrix, count_matrix)
df2.reset_index()
indices = pd.Series(df2.index, index=df2["title"])

print(get_recommendations("Fantastic 4: Rise of the Silver Surfer", cos_sim2))

'''Collaborative Filtering'''

reader = Reader()
ratings = pd.read_csv('../input/the-movies-dataset/ratings_small.csv')
# print(ratings.head())
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
data.split(n_folds=5)
svd = SVD()
evaluate(svd, data, measures=['RMSE', 'MAE']
trainset = data.build_full_trainset()
svd.fit(trainset)
ratings[ratings['userId'] == 1]
# svd.predict(1, 105, 3)















