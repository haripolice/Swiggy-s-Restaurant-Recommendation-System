import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


# Sidebar Navigation
st.sidebar.title("App Navigation")

#choose the Page
page = st.sidebar.radio("Select Page", ("Home", "Restaurant Recommendations"))

#Home page
if page == "Home":
  

    # Title with color
    st.markdown("<h1 style='color:#FFA500;'>Welcome to the Smart Restaurant Recommender!</h1>", unsafe_allow_html=True)

    # Subheader with color
    st.markdown("<h3 style='color:#00CED1;'>Your personalized restaurant guide based on city, cuisine, and preferences.</h3>", unsafe_allow_html=True)

    # Paragraph with color
    st.markdown("<p style='color:#FF1493;'>Get the best recommendations tailored to your taste.</p>", unsafe_allow_html=True)

    st.image("images.png")  # Adjust dimensions as needed

#Restaurant Recommendations page
elif page == "Restaurant Recommendations":

     # --- Page Title ---
    st.markdown("<h1 style='color:#FFA500;'>🍽️ Smart Swiggy Restaurant Recommender!</h1>", unsafe_allow_html=True)
    
    
        # --- Load Data ---
    cleaned_data = pd.read_csv("cleaned_data.csv")
    pca_data = pd.read_csv("pca_encoded_data.csv")

    # --- Load Models & Encoders ---

    #load cuisine encoder pickle file
    with open("cuisine_encoder.pkl", "rb") as f:
        cuisine_encoder = pickle.load(f)

    #load city encoder pickle file
    with open("city_encoder.pkl", "rb") as f:
        city_encoder = pickle.load(f)

    #load pca model pickle file
    with open("pca_model.pkl", "rb") as f:
        pca_model = pickle.load(f)
    
    #load scaling pickle file
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    
    #load kmeans model pickle file
    with open("kmeans_model.pkl", "rb") as f:
        kmeans_model = pickle.load(f)
        
    #load pca input pickle file
    with open("pca_input_columns.pkl", "rb") as f:
        pca_input_columns = pickle.load(f)

    

    # --- Sidebar Inputs ---
    with st.sidebar:
        #side bar header
        st.header("🔧 Filter Options")
       
       #To get the unquie values of city cloumn
        city_list = sorted(cleaned_data['city'].unique())

        #selectbox From city list 
        selected_city = st.selectbox("Select City", city_list)

        #To apply the filter based on city
        city_filtered = cleaned_data[cleaned_data['city'] == selected_city]

       # if the select city is empty
        if city_filtered.empty:
            st.warning("No data available for this city.")
            st.stop()

        #To get the unquie values of cuisine cloumn
        cuisine_list = sorted(city_filtered['cuisine'].unique())

         #selectbox From cuisine list
        selected_cuisine = st.selectbox("Select Cuisine", cuisine_list)
        
        #To apply the filter based on cuisine
        cuisine_filtered = city_filtered[city_filtered['cuisine'] == selected_cuisine]
        
        #To get the unquie values form rating cloumn
        rating_list = sorted(cuisine_filtered['rating'].unique())

        #To create the selectbox based on rating rating_list
        selected_rating = st.selectbox("Select Rating", rating_list)

        #To get the unquie values of rating count cloumn
        rating_count_list = sorted(cuisine_filtered[cuisine_filtered['rating'] == selected_rating]['rating_count'].unique() )

        #To create the selectboz based on rating_count_list
        selected_rating_count = st.selectbox("Select Rating Count", rating_count_list)

        #To get the cost values based on the condition for both rating and cuisine clooumn selection
        cost_list = sorted(cuisine_filtered[(cuisine_filtered['rating'] == selected_rating) &(cuisine_filtered['rating_count'] == selected_rating_count)]['cost'].unique())
        
        #to create the selectbox based on cost_list
        selected_cost = st.selectbox("Select Cost", cost_list)

       # To create the radio button for selection both distance method
        distance_method = st.radio("Select Distance Method", ["Euclidean", "Cosine"])

       # to create  button
        recommend_button = st.button("🔍 Get Recommendations")

    # --- Main Content: Recommendations ---
    if recommend_button:
        # --- Encode Inputs --- aaply the encoder for city cloumns
        city_encoded = pd.DataFrame(city_encoder.transform(pd.DataFrame([[selected_city]], columns=["city"])),columns=city_encoder.get_feature_names_out())

        #To Apply the encoder for cuisine cloumns
        cuisine_encoded = pd.DataFrame(cuisine_encoder.transform([[selected_cuisine]]),columns=cuisine_encoder.classes_)
       
       #To get user input from the user
        num_df = pd.DataFrame([{'rating': selected_rating,'rating_count': selected_rating_count,'cost': selected_cost}])
       
       # To concat the numerical user , city and cousine
        input_df = pd.concat([num_df, city_encoded, cuisine_encoded], axis=1)

        #from the contact datafreame to reindx for pca columns
        input_df = input_df.reindex(columns=pca_input_columns, fill_value=0)
       
       # To apply the scaling on input datafreame
        scaled_input = scaler.transform(input_df)

       # To apply the pca on the scaled input dataframe
        input_pca = pca_model.transform(scaled_input)
        
        #To apply the cluster model for input pca and using zero to get the accurate cluster
        input_cluster = kmeans_model.predict(input_pca)[0]

        #To check the model on which cluster
        cluster_indices = np.where(kmeans_model.labels_ == input_cluster)[0]

        # Get the original indices of all data points in the same cluster as the user input
        candidate_indices = pca_data.iloc[cluster_indices].index

        # To extract the results form the cleaned data scv based on lables 
        candidate_df = cleaned_data.loc[candidate_indices]
    
        # To Apply the conditon for both city and cuisine after clustering for better resluts
        candidate_df = candidate_df[(candidate_df['city'] == selected_city) &(candidate_df['cuisine'] == selected_cuisine)]

       #If my dataframe is empty it show no datafound
        if candidate_df.empty:
            st.warning("No similar restaurants found.")
        
        #Else it show the recomdate data
        else:
            pca_candidates = pca_data.loc[candidate_df.index]
          
            # from this recemandation we usint euclidan distance
            if distance_method == "Euclidean":
                distances = np.linalg.norm(pca_candidates.values - input_pca, axis=1)

            # Else cosine distance method    
            else:
                similarities = cosine_similarity(pca_candidates.values, input_pca)

               #use flateen to reshape the index 
                distances = 1 - similarities.flatten()
        
            # Get indices of the 10 most similar restaurants (lowest distances)
            top_indices = np.argsort(distances)[:10]
            
            # To take copy to avoid the prevoius data
            top_restaurants = candidate_df.iloc[top_indices].copy()

            #To identify the distance
            top_restaurants["Distance"] = distances[top_indices]

            #st.success(f"🎯 You belong to Cluster #{input_cluster}")
            st.markdown(f"<h3 style='color: #FF69B4;'>📍 Top 10 Recommended Restaurants (Using {distance_method}_recommended)</h3>",unsafe_allow_html=True)

           # from this result we can itearte all the idenex and row using forlop 
            for idx, row in top_restaurants.iterrows():

               #To create the markdown for name with clour 
                st.markdown(f"<h4 style='color:#FF4500'>🍴 {row['name']}</h4>", unsafe_allow_html=True)  # Orange Red
                #To create the markdown for city with clour 
                st.markdown(f"<span style='color:#FF1493'>📍 <b>City:</b> {row['city']}</span>", unsafe_allow_html=True)  # Deep Pink
                #To create the markdown for cuisine with clour 
                st.markdown(f"<span style='color:#4169E1'>🍽️ <b>Cuisine:</b> {row['cuisine']}</span>", unsafe_allow_html=True)  # Royal Blue
                #To create the markdown for rating with clour 
                st.markdown(f"<span style='color:#32CD32'>⭐ <b>Rating:</b> {row['rating']}</span>", unsafe_allow_html=True)  # Lime Green
                #To create the markdown for rating_count with clour 
                st.markdown(f"<span style='color:#008B8B'>💬 <b>Reviews:</b> {row['rating_count']}</span>", unsafe_allow_html=True)  # Gold
                #To create the markdown for cost with cuisine with clour 
                st.markdown(f"<span style='color:#9932CC'>💸 <b>Cost of {row['cuisine']}:</b> ₹{row['cost']}</span>", unsafe_allow_html=True)  # Dark Orchid


            # 📍 Address with Google Maps
            if pd.notna(row.get("address", None)):
                maps_url = f"https://www.google.com/maps/search/{row['address'].replace(' ', '+')}"
                st.markdown(f"📌 <a href='{maps_url}' style='color:#228B22' target='_blank'><b>View Location on Google Maps</b></a>", unsafe_allow_html=True)

            # 🔗 Website link
            if pd.notna(row.get("link", None)):
                st.markdown(f"🔗 <a href='{row['link']}' style='color:#20B2AA' target='_blank'><b>Order Online</b></a>", unsafe_allow_html=True)


            st.markdown("---")  # Divider between restaurants
