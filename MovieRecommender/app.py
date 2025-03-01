from flask import Flask, render_template, request, jsonify
import requests
import json
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)

HF_TOKEN = "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"
API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

# Load dataset
df = pd.read_csv("E:/DS_GENRE.csv")
genre_columns = df.columns[1:-1]  # Exclude 'movie_name' and 'Cluster'

# Preprocessing for clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[genre_columns])
nbrs = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(X_scaled)

def classify_genres(description):
    half = len(genre_columns) // 2
    genre_list_1, genre_list_2 = list(genre_columns[:half]), list(genre_columns[half:])
    
    def call_api(genres):
        payload = {"inputs": description, "parameters": {"candidate_labels": genres, "multi_label": True}}
        response = requests.post(API_URL, headers=HEADERS, data=json.dumps(payload))
        return response.json() if response.status_code == 200 else {}
    
    result_1, result_2 = call_api(genre_list_1), call_api(genre_list_2)
    
    genre_scores = {}
    if "labels" in result_1 and "scores" in result_1:
        genre_scores.update({label: round(score * 100, 2) for label, score in zip(result_1["labels"], result_1["scores"])})
    if "labels" in result_2 and "scores" in result_2:
        genre_scores.update({label: round(score * 100, 2) for label, score in zip(result_2["labels"], result_2["scores"])})
    
    return genre_scores

def find_closest_movie(genre_percentages):
    input_vector = [genre_percentages.get(genre, 0) for genre in genre_columns]
    input_scaled = scaler.transform([input_vector])
    dist, idx = nbrs.kneighbors(input_scaled)
    return df.iloc[idx[0][0]]['movie_name'], dist[0][0]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    description = request.form.get('description', '').strip()
    if not description:
        return jsonify({"error": "No description provided"})
    
    genre_percentages = classify_genres(description)
    if not genre_percentages:
        return jsonify({"error": "Failed to classify genres"})
    
    closest_movie, distance = find_closest_movie(genre_percentages)
    
    return jsonify({
        "entered_description": description,
        "genre_scores": genre_percentages,
        "closest_movie": closest_movie,
        "distance": round(distance, 2)
    })

if __name__ == '__main__':
    app.run(debug=True)
