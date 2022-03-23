import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px








# FRONT END ---------------------------------------------------------
st.title('Top 100 IMBb adventure movies')
st.markdown('This App shows some visualiztions on the top 100 adventure movies on [IMDb](https://www.imdb.com/) and give some recommandations on movies to watch.')

# Glimpse on data
show_data = st.sidebar.checkbox('Glimpse on data?: ')

# Movies recommendations
recommendation = st.sidebar.checkbox('See movie recommendation')
# Plot selector
plot_selected = st.sidebar.selectbox('Select a visualization: ', 
('None', 'rating per year of release', 'Number of movies per actor', 'Number of movies per director'))






# BACK END --------------------------------------------------------

# import data
data = pd.read_csv('data_imdb_adventure.csv')

# converting to datetime
data[['start_filming_date', 'end_filming_date']] = data[[ 'start_filming_date', 'end_filming_date']].apply(pd.to_datetime)


# Show first 10 observations
if show_data == True:
    st.write(data.head(10))


if plot_selected == 'rating per year of release':
    # choose variable for color
    
    fig = px.scatter(data, x="release_date", y="rating", color= 'votes', 
    template='seaborn', 
    title='Movie rating over time', 
    labels={"release_date": "Year of release",
                     "rating": "Movie rating"})

    st.plotly_chart(fig)


elif plot_selected == 'Number of movies per actor':
    # choose variable for color
    
    movie_actors_dict = {}
    for movie, actors in zip(data['movie_name'], data['actors']):
        movie_actors_dict[movie] = actors.split(',')

    movie_actors = pd.DataFrame(movie_actors_dict).melt(var_name= 'movie', value_name='actors')


    # control number of bar
    k = st.sidebar.selectbox('Minimum number of movies played', [2, 3])

    # actors that played at least k number of movies
    nb_movies_per_actor = movie_actors['actors'].value_counts()

    serie_mv_act = nb_movies_per_actor [nb_movies_per_actor >= k]
    actor_movie = pd.DataFrame({'actor': serie_mv_act.index, 'number_movies':serie_mv_act.values})


    fig = px.bar(actor_movie, x="actor", y="number_movies",
    template='seaborn',
    title=f'Actors that played at least {k} movies', 
    labels={"actor": "Actor's name",
            "number_movies": "Number of movies"})

    st.plotly_chart(fig)



