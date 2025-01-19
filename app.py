#as it is virtual environment, no libraries installed
import streamlit as st
import pandas as pd

import pickle
movies_dict=pickle.load(open('movies_dict.pkl','rb'))
movies=pd.DataFrame(movies_dict) #create a data frame

similarity=pickle.load(open('similarity.pkl','rb'))


def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]  # similarity matrix index
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended=[]
    for i in movies_list:
        recommended.append(movies.iloc[i[0]].title)  # fetching the movie title from newdf dataframe
    return recommended


st.title('The Movie Recommender')
select_movie_name=st.selectbox('what do you want to watch?',
                   movies['title'].values)

if st.button('Recommend'):
    recommendation=recommend(select_movie_name)
    for i in recommendation:
        st.write(i)


