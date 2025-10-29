Swiggy Restaurant Recommendation System

A Streamlit-based Restaurant Recommendation System that recommends restaurants to users based on preferences such as city, cuisine, cost, and rating. The system uses data preprocessing, One-Hot Encoding, and Cosine Similarity to find similar restaurants.

🧠 Features

Data cleaning and preprocessing (duplicates, missing values)

One-Hot Encoding of categorical features

Similarity-based restaurant recommendations (Cosine Similarity / K-Means)

Streamlit web interface for easy use

Dynamic user inputs for city, cuisine, and price filters

🗂️ Project Structure
Project-4-main/
│
├── cleaned_data.csv
├── encoded_data.csv
├── encoder.pkl
├── swiggy.py                  # Streamlit app
├── project.ipynb              # Jupyter Notebook for data processing
├── README.md
├── requirements.txt
└── Project_Report.md
⚙️ Installation
1️⃣ Clone the repository
git clone https://github.com/yourusername/swiggy-recommendation-system.git
cd swiggy-recommendation-system
2️⃣ Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
3️⃣ Install dependencies
pip install -r requirements.txt
🚀 Usage
Run the Streamlit App
streamlit run swiggy.py
In the browser:

You’ll see a web interface where you can:

Select city, cuisine, minimum rating, and cost range

Click Recommend to view top restaurant suggestions

📊 Methodology

Data Cleaning: Remove duplicates, handle missing values.

Encoding: Apply One-Hot Encoding on categorical columns.

Modeling: Compute similarity between restaurants using Cosine Similarity.

Interface: Build an interactive Streamlit app to display recommendations.

🧾 Deliverables

Cleaned dataset (cleaned_data.csv)

Encoded dataset (encoded_data.csv)

Pickle file of encoder (encoder.pkl)

Streamlit app (swiggy.py)

Report (Project_Report.md)

📈 Future Improvements

Add collaborative filtering

Integrate user login and history-based suggestions

Deploy on cloud (Streamlit Cloud / Heroku)
