# S-Tier Python Script for Clustering Video Games by Player Engagement
# Author: Uche (written to sound human & original)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import requests
import os

# -------------------------------
# 1. Load the dataset
# -------------------------------
API_KEY = "bfba5d8093bd4a9c88a0ad64b833ce80"
rawg_url = "https://api.rawg.io/api/games"

rawg_games = []
for page in range(1, 6):
    response = requests.get(rawg_url, params={
        "key": API_KEY,
        "page": page,
        "page_size": 20
    })
    if response.status_code == 200:
        page_data = response.json().get("results", [])
        for game in page_data:
            rawg_games.append({
                "title": game["name"],
                "released": game["released"],
                "metacritic": game.get("metacritic"),
                "genres": [g["name"] for g in game.get("genres", [])],
                "tags": [t["name"] for t in game.get("tags", [])] if game.get("tags") else [],
                "platforms": [p["platform"]["name"] for p in game.get("platforms", [])]
            })

rawg_df = pd.DataFrame(rawg_games)
rawg_df.to_csv("rawg_games.csv", index=False)

steamspy_url = "https://steamspy.com/api.php?request=top100in2weeks"
response = requests.get(steamspy_url)

if response.status_code == 200:
    steam_data = response.json()
    steam_games = []

    for appid, game in steam_data.items():
        steam_games.append({
            "title": game["name"],
            "appid": game["appid"],
            "average_forever": game["average_forever"],
            "positive": game["positive"],
            "negative": game["negative"],
            "owners": game["owners"],
            "genre": game.get("genre"),
            "developer": game.get("developer")
        })

    df_steam = pd.DataFrame(steam_games)
    df_steam.to_csv("steamspy_games.csv", index=False)
    print("✅ SteamSpy data saved to steamspy_games.csv")
else:
    print("❌ SteamSpy API request failed:", response.status_code)

if os.path.exists("rawg_games.csv") and os.path.exists("steamspy_games.csv"):
    df_rawg = pd.read_csv("rawg_games.csv")
    df_steam = pd.read_csv("steamspy_games.csv")
    merged = pd.merge(df_steam, df_rawg, on="title", how="inner")
    merged.rename(columns={
        "average_forever": "avg_playtime",
        "positive": "total_reviews",
        "metacritic": "metascore"
    }, inplace=True)
    merged["genre"] = merged["genre"].astype(str)
    merged["multiplayer"] = merged["genre"].str.contains("Multi", case=False, na=False).astype(int)
    merged["main_genre"] = merged["genres"].apply(lambda x: eval(x)[0] if pd.notnull(x) and len(eval(x)) > 0 else "Unknown")
    merged.to_csv("video_game_engagement.csv", index=False)
    print("✅ Merged dataset saved as video_game_engagement.csv")
else:
    print("❌ Cannot merge RAWG and SteamSpy data — one of the CSVs is missing.")

data = pd.read_csv("video_game_engagement.csv")

engagement_cols = ["avg_playtime", "total_reviews", "multiplayer", "metascore"]
data[engagement_cols] = data[engagement_cols].fillna(0)

data = pd.get_dummies(data, columns=["main_genre"], drop_first=True)

features = ["avg_playtime", "total_reviews", "multiplayer", "metascore"] + \
           [col for col in data.columns if col.startswith("main_genre_")]
X = data[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertias = []
k_range = range(2, 10)
for k in k_range:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X_scaled)
    inertias.append(model.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_range, inertias, marker='o')
plt.title("Elbow Method to Choose k")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.grid(True)
plt.tight_layout()
plt.show()

k_optimal = 4
kmeans = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
data['cluster'] = kmeans.fit_predict(X_scaled)

pca = PCA(n_components=2)
pca_components = pca.fit_transform(X_scaled)
data['PC1'] = pca_components[:, 0]
data['PC2'] = pca_components[:, 1]

plt.figure(figsize=(9, 6))
sns.scatterplot(data=data, x='PC1', y='PC2', hue='cluster', palette='Set2', s=60)
plt.title("PCA of Video Game Engagement Clusters")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.tight_layout()
plt.show()

summary = data.groupby('cluster')[engagement_cols].mean().round(1)
print("\nCluster Engagement Summary:")
print(summary)

genre_columns = [col for col in data.columns if col.startswith("main_genre_")]
genre_summary = data.groupby("cluster")[genre_columns].mean().round(2)
print("\nTop Genres by Cluster:")
print(genre_summary)

print("\nExample Games Per Cluster:")
for cluster_id in sorted(data['cluster'].unique()):
    examples = data[data['cluster'] == cluster_id]['title'].head(2).tolist()
    print(f"Cluster {cluster_id}: {examples}")

data.to_csv("clustered_video_games.csv", index=False)
