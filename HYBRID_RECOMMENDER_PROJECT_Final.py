#############################################

import pandas as pd
pd.pandas.set_option('display.max_columns', None)
pd.pandas.set_option('display.width', 300)


movie = pd.read_csv('Modül_4_Tavsiye_Sistemleri/datasets/movie.csv')
movie.head()
movie.shape


rating = pd.read_csv('Modül_4_Tavsiye_Sistemleri/datasets/rating.csv')
rating.head()
rating.shape
rating["userId"].nunique()


df = movie.merge(rating, how="left", on="movieId")
df.head()
df.shape


comment_counts = pd.DataFrame(df["title"].value_counts())
comment_counts


rare_movies = comment_counts[comment_counts["title"] <= 1000].index
common_movies = df[~df["title"].isin(rare_movies)]
common_movies.shape


user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
user_movie_df.head()



def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
    rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')
    df = movie.merge(rating, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df

user_movie_df = create_user_movie_df()


#############################################

# Random User Select
random_user = 108170

# random_user_df
random_user_df = user_movie_df[user_movie_df.index == random_user]
random_user_df.head()
random_user_df.shape

# movies_watched
movies_watched = random_user_df.columns[random_user_df.notna().any()].to_list()
movies_watched

movie.columns[movie.notna().any()].to_list()


#############################################

# user_movie_df --> movies_watched_df 
movies_watched_df = user_movie_df[movies_watched]
movies_watched_df.head()
movies_watched_df.shape

# user_movie_count 
user_movie_count = movies_watched_df.T.notnull().sum()
user_movie_count = user_movie_count.reset_index()
user_movie_count.columns = ["userId", "movie_count"]
user_movie_count.head(5)

# We consider those who watch 60 percent or more of the movies voted by the selected user as similar users.
# users_same_movies 
perc = len(movies_watched) * 60 / 100
users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]
len(users_same_movies)



# Identifying Users Who Are Most Similar to the User to Suggest

# Filter the movies_watched_df dataframe to find the ids of the users that are similar to the selected user in the user_same_movies list.
final_df = movies_watched_df[movies_watched_df.index.isin(users_same_movies)]
final_df.head()
final_df.shape

# Create a new corr_df dataframe where users' correlations with each other will be found.
corr_df = final_df.T.corr().unstack().sort_values()
corr_df = pd.DataFrame(corr_df, columns=["corr"])
corr_df.index.names = ['user_id_1', 'user_id_2']
corr_df = corr_df.reset_index()

#corr_df[corr_df["user_id_1"] == random_user]



# Create a new dataframe named top_users by filtering out the users with high correlation (over 0.65) with the selected user.
top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][["user_id_2", "corr"]].reset_index(drop=True)
top_users = top_users.sort_values(by='corr', ascending=False)
top_users.rename(columns={"user_id_2": "userId"}, inplace=True)
top_users.shape
top_users.head()

# Merge the top_users dataframe with the rating dataset
top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')
top_users_ratings = top_users_ratings[top_users_ratings["userId"] != random_user]
top_users_ratings["userId"].unique()
top_users_ratings.head()




# Calculating Weighted Average Recommendation Score and Keeping Top 5 Movies

# Create a new variable named weighted_rating, which is the product of each user's corr and rating.
top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']
top_users_ratings.head()

# Create a new dataframe named recommendation_df containing the movie id and the average value of the weighted ratings of all users for each movie.
recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})
recommendation_df = recommendation_df.reset_index()
recommendation_df.head()

# Select movies with a weighted rating greater than 3.5 in recommendation_df and rank them by weighted rating. Save the first 5 observations as movies_to_be_recommend.
recommendation_df[recommendation_df["weighted_rating"] > 3.5]
movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 3.5].sort_values("weighted_rating", ascending=False)


movies_to_be_recommend.merge(movie[["movieId", "title"]])["title"][:5]

# 0    Mystery Science Theater 3000: The Movie (1996)
# 1                               Natural, The (1984)
# 2                             Super Troopers (2001)
# 3                         Christmas Story, A (1983)
# 4                       Sonatine (Sonachine) (1993)




# Item-Based Recommendation

# Make an item-based suggestion based on the name of the movie that the user last watched and gave the highest rating.
user = 108170


movie = pd.read_csv('Modül_4_Tavsiye_Sistemleri/datasets/movie.csv')
rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')


movie_id = rating[(rating["userId"] == user) & (rating["rating"] == 5.0)].sort_values(by="timestamp", ascending=False)["movieId"][0:1].values[0]


movie[movie["movieId"] == movie_id]["title"].values[0]
movie_df = user_movie_df[movie[movie["movieId"] == movie_id]["title"].values[0]]
movie_df


user_movie_df.corrwith(movie_df).sort_values(ascending=False).head(10)


def item_based_recommender(movie_name, user_movie_df):
    movie = user_movie_df[movie_name]
    return user_movie_df.corrwith(movie).sort_values(ascending=False).head(10)


movies_from_item_based = item_based_recommender(movie[movie["movieId"] == movie_id]["title"].values[0], user_movie_df)

movies_from_item_based[1:6].index


# 'My Science Project (1985)',
# 'Mediterraneo (1991)',
# 'Old Man and the Sea,
# The (1958)',
# 'National Lampoon's Senior Trip (1995)',
# 'Clockwatchers (1997)']



