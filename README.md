# Swiggy Restaurant Recommendation System

A Streamlit-based Restaurant Recommendation System that recommends restaurants to users based on preferences such as city, cuisine, cost, and rating. The system uses data preprocessing, One-Hot Encoding, and Cosine Similarity to find similar restaurants.

## Features
- Data cleaning and preprocessing
- One-Hot Encoding of categorical features
- Similarity-based restaurant recommendations (Cosine Similarity / K-Means)
- Streamlit web interface
- Dynamic user inputs for city, cuisine, and price filters

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
streamlit run swiggy.py
```

## Project Structure
- cleaned_data.csv
- encoded_data.csv
- encoder.pkl
- swiggy.py
- project.ipynb
- Project_Report.md
