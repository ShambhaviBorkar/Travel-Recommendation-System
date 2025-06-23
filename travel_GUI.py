import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Set page config
st.set_page_config(
    page_title="Travel Recommendation System",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .sidebar .sidebar-content {
        background-color: #e8f4f8;
    }
    h1 {
        color: #1e3d6b;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 24px;
    }
    .stSelectbox, .stTextInput, .stNumberInput {
        border-radius: 5px;
    }
    .recommendation-card {
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        background-color: white;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Load datasets
@st.cache_data
def load_data():
    destinations_df = pd.read_csv("Expanded_Destinations.csv")
    reviews_df = pd.read_csv("Final_Updated_Expanded_Reviews.csv")
    userhistory_df = pd.read_csv("Final_Updated_Expanded_UserHistory.csv")
    users_df = pd.read_csv("Final_Updated_Expanded_Users.csv")
    
    # Merge datasets
    reviews_destinations = pd.merge(reviews_df, destinations_df, on='DestinationID', how='inner')
    reviews_destinations_userhistory = pd.merge(reviews_destinations, userhistory_df, on='UserID', how='inner')
    df = pd.merge(reviews_destinations_userhistory, users_df, on='UserID', how='inner')
    
    return destinations_df, reviews_df, userhistory_df, users_df, df

destinations_df, reviews_df, userhistory_df, users_df, df = load_data()

# Recommendation functions
def get_user_based_recommendations(user_id, num_recommendations=5):
    try:
        # Create user-destination matrix
        user_dest_matrix = userhistory_df.pivot_table(
            index='UserID',
            columns='DestinationID',
            values='ExperienceRating',
            fill_value=0
        )
        
        # Calculate cosine similarity between users
        user_similarity = cosine_similarity(user_dest_matrix)
        user_similarity_df = pd.DataFrame(
            user_similarity,
            index=user_dest_matrix.index,
            columns=user_dest_matrix.index
        )
        
        # Get similar users (excluding the user themselves)
        similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:6].index
        
        # Get destinations rated highly by similar users
        similar_users_ratings = user_dest_matrix.loc[similar_users]
        avg_ratings = similar_users_ratings.mean(axis=0)
        
        # Filter out destinations already visited by the user
        visited_destinations = userhistory_df[userhistory_df['UserID'] == user_id]['DestinationID'].unique()
        recommendations = avg_ratings[~avg_ratings.index.isin(visited_destinations)]
        recommendations = recommendations.sort_values(ascending=False).head(num_recommendations)
        
        return recommendations.index.tolist()
    
    except Exception as e:
        st.error(f"Error in recommendation generation: {str(e)}")
        return []

def get_popular_recommendations(num_recommendations=5):
    return destinations_df.sort_values('Popularity', ascending=False)['DestinationID'].head(num_recommendations).tolist()

def get_content_based_recommendations(destination_id, num_recommendations=5):
    try:
        # Vectorize destination features
        vectorizer = CountVectorizer()
        features = destinations_df['Type'] + ' ' + destinations_df['State'] + ' ' + destinations_df['BestTimeToVisit']
        feature_matrix = vectorizer.fit_transform(features)
        
        # Calculate cosine similarity between destinations
        similarity_matrix = cosine_similarity(feature_matrix)
        
        # Get similar destinations
        destination_idx = destinations_df[destinations_df['DestinationID'] == destination_id].index[0]
        similar_destinations = list(enumerate(similarity_matrix[destination_idx]))
        similar_destinations = sorted(similar_destinations, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]
        
        return [destinations_df.iloc[i[0]]['DestinationID'] for i in similar_destinations]
    
    except Exception as e:
        st.error(f"Error in content-based recommendations: {str(e)}")
        return []

# Streamlit app
def main():
    st.title("✈️ Travel Recommendation System")
    st.markdown("Discover your perfect vacation spot with our AI-powered recommendation engine!")

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    menu = ["Home", "Recommendations", "Data Analysis", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.header("Welcome to the Travel Recommender!")
        st.image("https://images.unsplash.com/photo-1508672019048-805c876b67e2", 
                 width=700, caption="Find your next adventure!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dataset Overview")
            st.write(f"Total Users: {len(users_df)}")
            st.write(f"Total Destinations: {len(destinations_df)}")
            st.write(f"Destination Types: {', '.join(destinations_df['Type'].unique())}")
            
        with col2:
            st.subheader("Top Destinations")
            top_dests = destinations_df.sort_values('Popularity', ascending=False).head(5)
            st.dataframe(top_dests[['Name', 'State', 'Type', 'Popularity']])
        
        st.subheader("Destination Types Distribution")
        fig, ax = plt.subplots(figsize=(4,4))
        destinations_df['Type'].value_counts().plot(
            kind='pie',
            autopct='%1.1f%%',
            ax=ax
        )
        st.pyplot(fig)
    elif choice == "Recommendations":
        st.header("Get Personalized Recommendations")
        
        recommendation_type = st.radio("Select Recommendation Type", 
                                     ["User-Based", "Popularity-Based", "Content-Based"])
        
        if recommendation_type == "User-Based":
            st.subheader("User-Based Recommendations")
            st.write("Get recommendations based on what similar users enjoyed")
            
            user_id = st.selectbox("Select User ID", sorted(userhistory_df['UserID'].unique()))
            num_rec = st.slider("Number of Recommendations", 1, 10, 5)
            
            if st.button("Get Recommendations"):
                with st.spinner('Finding what similar travelers enjoyed...'):
                    recommended_ids = get_user_based_recommendations(user_id, num_rec)
                    
                    if recommended_ids:
                        recommended_destinations = destinations_df[
                            destinations_df['DestinationID'].isin(recommended_ids)
                        ].sort_values('Popularity', ascending=False)
                        
                        st.success("Here are recommendations from similar travelers:")
                        for idx, row in recommended_destinations.iterrows():
                            with st.expander(f"🌟 {row['Name']} - Popularity: {row['Popularity']:.1f}"):
                                st.write(f"**Type:** {row['Type']}")
                                st.write(f"**Location:** {row['State']}")
                                st.write(f"**Best Time to Visit:** {row['BestTimeToVisit']}")
                                if 'Description' in row:
                                    st.write(f"**Description:** {row['Description']}")
                    else:
                        st.warning("No recommendations found. Try a different user ID.")

        elif recommendation_type == "Popularity-Based":
            st.subheader("Popularity-Based Recommendations")
            st.write("Get recommendations based on overall popularity")
            
            num_rec = st.slider("Number of Recommendations", 1, 10, 5)
            
            if st.button("Get Popular Destinations"):
                with st.spinner('Finding most popular destinations...'):
                    popular_ids = get_popular_recommendations(num_rec)
                    popular_destinations = destinations_df[
                        destinations_df['DestinationID'].isin(popular_ids)
                    ].sort_values('Popularity', ascending=False)
                    
                    st.success("Most Popular Destinations:")
                    for idx, row in popular_destinations.iterrows():
                        with st.expander(f"🏆 {row['Name']} - Popularity: {row['Popularity']:.1f}"):
                            st.write(f"**Type:** {row['Type']}")
                            st.write(f"**Location:** {row['State']}")
                            st.write(f"**Best Time to Visit:** {row['BestTimeToVisit']}")
                            if 'Description' in row:
                                st.write(f"**Description:** {row['Description']}")

        elif recommendation_type == "Content-Based":
            st.subheader("Content-Based Recommendations")
            st.write("Get recommendations based on similar destination characteristics")
            
            destination_id = st.selectbox("Select Destination ID", 
                                         sorted(destinations_df['DestinationID'].unique()))
            num_rec = st.slider("Number of Recommendations", 1, 10, 5)
            
            if st.button("Get Similar Destinations"):
                with st.spinner('Finding similar destinations...'):
                    selected_dest = destinations_df[
                        destinations_df['DestinationID'] == destination_id
                    ].iloc[0]
                    
                    st.write(f"Selected Destination: **{selected_dest['Name']}**")
                    st.write(f"Type: {selected_dest['Type']} | State: {selected_dest['State']}")
                    st.write("---")
                    
                    similar_ids = get_content_based_recommendations(destination_id, num_rec)
                    similar_destinations = destinations_df[
                        destinations_df['DestinationID'].isin(similar_ids)
                    ].sort_values('Popularity', ascending=False)
                    
                    st.success("Similar Destinations:")
                    for idx, row in similar_destinations.iterrows():
                        with st.expander(f"🔍 {row['Name']} - Popularity: {row['Popularity']:.1f}"):
                            st.write(f"**Type:** {row['Type']}")
                            st.write(f"**Location:** {row['State']}")
                            st.write(f"**Best Time to Visit:** {row['BestTimeToVisit']}")
                            if 'Description' in row:
                                st.write(f"**Description:** {row['Description']}")

    elif choice == "Data Analysis":
        st.header("Data Analysis")
        
        analysis_option = st.selectbox("Select Analysis", 
                                     ["Dataset Overview", "Rating Distribution", 
                                      "Destination Types", "Popularity Analysis"])
        
        if analysis_option == "Dataset Overview":
            st.write("### Users Dataset")
            st.write(users_df.head())
            st.write(f"Total Users: {len(users_df)}")
            
            st.write("### Destinations Dataset")
            st.write(destinations_df.head())
            st.write(f"Total Destinations: {len(destinations_df)}")
            
            st.write("### Reviews Dataset")
            st.write(reviews_df.head())
            st.write(f"Total Reviews: {len(reviews_df)}")
            
            st.write("### User History Dataset")
            st.write(userhistory_df.head())
            st.write(f"Total History Records: {len(userhistory_df)}")
        
        elif analysis_option == "Rating Distribution":
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.histplot(reviews_df['Rating'], bins=5, kde=True, color='blue', ax=ax)
            ax.set_title('Distribution of Ratings')
            ax.set_xlabel('Rating')
            ax.set_ylabel('Frequency')
            st.pyplot(fig)
            
            st.write("""
            The rating distribution shows how users have rated different destinations.
            Most ratings are centered around 3, with fewer extreme ratings (1 or 5).
            """)
        
        elif analysis_option == "Destination Types":
            type_counts = destinations_df['Type'].value_counts()
            fig, ax = plt.subplots(figsize=(8, 6))
            type_counts.plot(kind='bar', color='skyblue', ax=ax)
            ax.set_title('Distribution of Destination Types')
            ax.set_xlabel('Type')
            ax.set_ylabel('Count')
            st.pyplot(fig)
            
            st.write("""
            This shows the distribution of different types of destinations available.
            The most common types are shown in the bar chart above.
            """)
        
        elif analysis_option == "Popularity Analysis":
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=destinations_df, x='Type', y='Popularity', ax=ax)
            ax.set_title('Popularity by Destination Type')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            st.pyplot(fig)
            
            st.write("""
            This boxplot shows the distribution of popularity scores across different destination types.
            The higher the box, the more popular destinations of that type tend to be.
            """)

    elif choice == "About":
        st.header("About This Project")
        st.write("""
        This Travel Destination Recommendation System is designed to help users discover new places to visit based on:
        - Their personal preferences and travel history
        - Popular destinations among all users
        - Similarity to destinations they already like
        
        The system uses collaborative filtering, content-based filtering, and popularity-based approaches to provide diverse recommendations.
        """)
        
        st.subheader("Dataset Information")
        st.write(f"Total Users: {len(users_df)}")
        st.write(f"Total Destinations: {len(destinations_df)}")
        st.write(f"Total Reviews: {len(reviews_df)}")
        st.write(f"Total User History Records: {len(userhistory_df)}")
        
        st.subheader("Destination Types Available")
        st.write(", ".join(destinations_df['Type'].unique()))

if __name__ == "__main__":
    main()