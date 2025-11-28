import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------
# BASIC PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 Movie Recommendation System")
st.write("Get movie recommendations based on what you like.")


# ---------------------------------------------------------
# LOAD DATA (movies)
# ---------------------------------------------------------
@st.cache_resource
def load_movies():
    """
    Loads the movies dataframe from pickle.
    Expects movies_dict.pkl with at least 'title' and 'tags' columns.
    """
    movies_dict = pickle.load(open('movies_dict.pkl', 'rb'))
    movies = pd.DataFrame(movies_dict)
    return movies


movies = load_movies()

# Safety check – just in case
required_cols = {"title", "tags"}
if not required_cols.issubset(set(movies.columns)):
    st.error(
        f"Required columns {required_cols} not found in movies dataframe. "
        f"Found columns: {list(movies.columns)}"
    )
    st.stop()


# ---------------------------------------------------------
# BUILD SIMILARITY MATRIX (NO similarity.pkl NEEDED)
# ---------------------------------------------------------
@st.cache_resource
def build_similarity(movies_df: pd.DataFrame):
    """
    Builds cosine similarity matrix from the 'tags' column.
    Cached so it's computed only once per server.
    """
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(movies_df['tags']).toarray()
    similarity_matrix = cosine_similarity(vectors)
    return similarity_matrix


similarity = build_similarity(movies)


# ---------------------------------------------------------
#  RECOMMENDATION FUNCTION
# ---------------------------------------------------------
def recommend(movie_name: str, top_n: int = 5):
    """
    Given a movie title, returns top_n similar movie titles.
    """
    # Check if movie exists
    if movie_name not in movies['title'].values:
        return []

    movie_index = movies[movies['title'] == movie_name].index[0]
    distances = list(enumerate(similarity[movie_index]))
    # Sort by similarity score (descending), skip the first (itself)
    movies_list = sorted(distances, reverse=True, key=lambda x: x[1])[1: top_n+1]

    recommended_movies = []
    for i in movies_list:
        recommended_movies.append(movies.iloc[i[0]].title)

    return recommended_movies


# ---------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------
st.subheader("Select a movie to get recommendations")

selected_movie_name = st.selectbox(
    "Type or choose a movie:",
    movies['title'].values
)

if st.button("Recommend 🎥"):
    with st.spinner("Finding similar movies for you..."):
        recommendations = recommend(selected_movie_name)

    if not recommendations:
        st.warning("No recommendations found. Please try another movie.")
    else:
        st.success("Here are some movies you might like:")
        for idx, movie in enumerate(recommendations, start=1):
            st.write(f"**{idx}.** {movie}")


