import os
import zipfile
import streamlit as st
import pandas as pd
import pickle

# --- Step 1: Unzip data if needed ---
ZIP_FILE = "encoded_data.zip"
EXTRACT_DIR = "encoded_data"

if not os.path.exists(EXTRACT_DIR):
    if os.path.exists(ZIP_FILE):
        with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_DIR)
        print(f"✅ Extracted files to: {EXTRACT_DIR}")
    else:
        st.error(f"❌ Missing required file: {ZIP_FILE}")
        st.stop()

# --- Step 2: Define file paths from extracted folder ---
cleaned_data_path = os.path.join(EXTRACT_DIR, "cleaned_data.csv")
pca_data_path = os.path.join(EXTRACT_DIR, "encoded_data.csv")

# Pickle files are in project root (not inside ZIP)
cuisine_encoder_path = "cuisine_encoder.pkl"
city_encoder_path = "city_encoder.pkl"
pca_model_path = "pca_model.pkl"
scaler_path = "scaler.pkl"
kmeans_model_path = "kmeans_model.pkl"
pca_input_columns_path = "pca_input_columns.pkl"

# --- Step 3: Load Data and Models safely ---
try:
    # Load CSVs
    cleaned_data = pd.read_csv(cleaned_data_path)
    pca_data = pd.read_csv(pca_data_path)

    # Load model and encoder files
    with open(cuisine_encoder_path, "rb") as f:
        cuisine_encoder = pickle.load(f)
    with open(city_encoder_path, "rb") as f:
        city_encoder = pickle.load(f)
    with open(pca_model_path, "rb") as f:
        pca_model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(kmeans_model_path, "rb") as f:
        kmeans_model = pickle.load(f)
    with open(pca_input_columns_path, "rb") as f:
        pca_input_columns = pickle.load(f)

    print("✅ Data and model files loaded successfully!")

except Exception as e:
    st.error(f"❌ Error loading data/models: {e}")
    st.stop()
# --- Sidebar Navigation ---
st.sidebar.title("App Navigation")
page = st.sidebar.radio("Select Page", ("Home", "Restaurant Recommendations"))

# --- Home Page ---
if page == "Home":
    st.markdown("<h1 style='color:#FFA500;'>Welcome to the Smart Restaurant Recommender!</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='color:#00CED1;'>Your personalized restaurant guide based on city, cuisine, and preferences.</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color:#FF1493;'>Get the best recommendations tailored to your taste.</p>", unsafe_allow_html=True)
    st.image("images.png")

# --- Restaurant Recommendations Page ---
elif page == "Restaurant Recommendations":
    st.markdown("<h1 style='color:#FFA500;'>🍽️ Smart Swiggy Restaurant Recommender!</h1>", unsafe_allow_html=True)

    with st.sidebar:
        st.header("🔧 Filter Options")

        city_list = sorted(cleaned_data['city'].unique())
        selected_city = st.selectbox("Select City", city_list)

        city_filtered = cleaned_data[cleaned_data['city'] == selected_city]
        if city_filtered.empty:
            st.warning("No data available for this city.")
            st.stop()

        cuisine_list = sorted(city_filtered['cuisine'].unique())
        selected_cuisine = st.selectbox("Select Cuisine", cuisine_list)

        cuisine_filtered = city_filtered[city_filtered['cuisine'] == selected_cuisine]

        rating_list = sorted(cuisine_filtered['rating'].unique())
        selected_rating = st.selectbox("Select Rating", rating_list)

        rating_count_list = sorted(cuisine_filtered[cuisine_filtered['rating'] == selected_rating]['rating_count'].unique())
        selected_rating_count = st.selectbox("Select Rating Count", rating_count_list)

        cost_list = sorted(cuisine_filtered[(cuisine_filtered['rating'] == selected_rating) & (cuisine_filtered['rating_count'] == selected_rating_count)]['cost'].unique())
        selected_cost = st.selectbox("Select Cost", cost_list)

        distance_method = st.radio("Select Distance Method", ["Euclidean", "Cosine"])
        recommend_button = st.button("🔍 Get Recommendations")

    if recommend_button:
        city_encoded = pd.DataFrame(city_encoder.transform(pd.DataFrame([[selected_city]], columns=["city"])),
                                    columns=city_encoder.get_feature_names_out())

        cuisine_encoded = pd.DataFrame(cuisine_encoder.transform([[selected_cuisine]]),
                                       columns=cuisine_encoder.classes_)

        num_df = pd.DataFrame([{'rating': selected_rating, 'rating_count': selected_rating_count, 'cost': selected_cost}])

        input_df = pd.concat([num_df, city_encoded, cuisine_encoded], axis=1)
        input_df = input_df.reindex(columns=pca_input_columns, fill_value=0)

        scaled_input = scaler.transform(input_df)
        input_pca = pca_model.transform(scaled_input)

        input_cluster = kmeans_model.predict(input_pca)[0]
        cluster_indices = np.where(kmeans_model.labels_ == input_cluster)[0]
        candidate_indices = pca_data.iloc[cluster_indices].index
        candidate_df = cleaned_data.loc[candidate_indices]
        candidate_df = candidate_df[(candidate_df['city'] == selected_city) & (candidate_df['cuisine'] == selected_cuisine)]

        if candidate_df.empty:
            st.warning("No similar restaurants found.")
        else:
            pca_candidates = pca_data.loc[candidate_df.index]
            if distance_method == "Euclidean":
                distances = np.linalg.norm(pca_candidates.values - input_pca, axis=1)
            else:
                similarities = cosine_similarity(pca_candidates.values, input_pca)
                distances = 1 - similarities.flatten()

            top_indices = np.argsort(distances)[:10]
            top_restaurants = candidate_df.iloc[top_indices].copy()
            top_restaurants["Distance"] = distances[top_indices]

            st.markdown(f"<h3 style='color: #FF69B4;'>📍 Top 10 Recommended Restaurants (Using {distance_method} Distance)</h3>", unsafe_allow_html=True)

            for idx, row in top_restaurants.iterrows():
                st.markdown(f"<h4 style='color:#FF4500'>🍴 {row['name']}</h4>", unsafe_allow_html=True)
                st.markdown(f"<span style='color:#FF1493'>📍 <b>City:</b> {row['city']}</span>", unsafe_allow_html=True)
                st.markdown(f"<span style='color:#4169E1'>🍽️ <b>Cuisine:</b> {row['cuisine']}</span>", unsafe_allow_html=True)
                st.markdown(f"<span style='color:#32CD32'>⭐ <b>Rating:</b> {row['rating']}</span>", unsafe_allow_html=True)
                st.markdown(f"<span style='color:#008B8B'>💬 <b>Reviews:</b> {row['rating_count']}</span>", unsafe_allow_html=True)
                st.markdown(f"<span style='color:#9932CC'>💸 <b>Cost of {row['cuisine']}:</b> ₹{row['cost']}</span>", unsafe_allow_html=True)

                if pd.notna(row.get("address", None)):
                    maps_url = f"https://www.google.com/maps/search/{row['address'].replace(' ', '+')}"
                    st.markdown(f"📌 <a href='{maps_url}' style='color:#228B22' target='_blank'><b>View Location on Google Maps</b></a>", unsafe_allow_html=True)

                if pd.notna(row.get("link", None)):
                    st.markdown(f"🔗 <a href='{row['link']}' style='color:#20B2AA' target='_blank'><b>Order Online</b></a>", unsafe_allow_html=True)

                st.markdown("---")




