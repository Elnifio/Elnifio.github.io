import pandas as pd
df_movies=pd.read_csv("C:/Users/91832/Desktop/movies/movies.csv")
df_movies['year'] = df_movies.title.str.extract('(\(\d\d\d\d\))',expand=False)
df_movies['year'] = df_movies.year.str.extract('(\d\d\d\d)',expand=False)
df_movies['title'] = df_movies.title.str.replace('(\(\d\d\d\d\))', '')
df_movies['title'] = df_movies['title'].apply(lambda x: x.strip())
df_movies.head()
df_movies["genres"]=df_movies["genres"].apply(lambda x:x.split("|"))
df_movies["title"]=df_movies["title"].apply(lambda x:x.replace(x,x.lower()))
df_moviegenre=df_movies.copy()
for index, row in df_movies.iterrows():
    for genre in row['genres']:
        df_moviegenre.at[index, genre] = 1
df_moviegenre= df_moviegenre.fillna(0)
#df_moviegenre.head()
df_ratings=pd.read_csv("C:/Users/91832/Desktop/movies/ratings.csv")
df_ratings=df_ratings.drop("timestamp",1)
#df_ratings.head()
new=[]
print("If movie starts with 'the' enter movie name as '(name,the)'")
print("NUMBER OF MOVIES SEEN =")
for i in range(int(input())):
    dc={}
    print("MOVIE NAME =")
    dc["title"]=input()
    print("GIVEABLE RATING =")
    dc["rating"]=float(input())
    new.append(dc)
usermovies = pd.DataFrame(new)
#usermovies
inputid=df_movies[df_movies["title"].isin(usermovies["title"].tolist())]
usermovies=pd.merge(inputid,usermovies)
#usermovies
usermovies=usermovies.drop("genres",1).drop("year",1)
#usermovies
movies = df_moviegenre[df_moviegenre['movieId'].isin(usermovies['movieId'].tolist())]
movies=movies.drop("year",1).drop("title",1).drop("genres",1).drop("movieId",1)
movies=movies.reset_index(drop=True)
movies=movies.transpose().dot(usermovies["rating"])
genre = df_moviegenre.set_index(df_moviegenre['movieId'])
genre = genre.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)
df_movierecommend = ((genre*movies).sum(axis=1))/(movies.sum())
df_movierecommend=df_movierecommend.sort_values(ascending=False)
df_movies.loc[df_movies['movieId'].isin(df_movierecommend.head(20).keys())]