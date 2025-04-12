# Video Game Engagement Clustering 🎮✨

Welcome to the *Video Game Engagement Clustering* project. This repo breaks down trends in video game engagement by clustering games based on metrics like average playtime, review count, metascore, and genre using KMeans. Whether you're a gamer, data science enthusiast, or game dev trying to understand audience behavior—this project gives you insight into what types of games get played, loved, and talked about the most.

---

## 🔎 What This Project Does
This project answers a simple but powerful question:
> **"What types of video games tend to have the highest engagement and popularity?"**

Using real-world data from the **RAWG API** and **SteamSpy API**, we:
- Pull game data like genres, platforms, metascores, average playtime, and review stats
- Merge the data into a single dataset
- Standardize and clean the data
- Use **KMeans Clustering** to group games with similar player behavior patterns
- Visualize clusters using PCA
- Interpret clusters by genre, popularity, and examples

---

## 📚 Key Concepts Used
- KMeans Clustering (scikit-learn)
- PCA for visualization
- Feature scaling (StandardScaler)
- API data collection (requests)
- Data cleaning & one-hot encoding
- Elbow Method for finding optimal cluster count

---

## ⚡ How to Run This Code (Step-by-Step)

> ✅ Prerequisite: Python 3.8+ and pip installed

### 1. **Clone the repo**
```bash
git clone https://github.com/uche-onyejiaka/Video-Game-Clustering.git
cd Video-Game-Clustering
```

### 2. **(Optional but Recommended) Set up a Virtual Environment**
```bash
python3 -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
```

### 3. **Install required packages**
```bash
pip install -r requirements.txt
```
> If you don’t have a `requirements.txt`, manually install:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn requests
```

### 4. **Get a RAWG API Key**
- Visit: https://rawg.io/apidocs
- Create an account and grab your API key
- Replace the `API_KEY` variable in the script with your key

### 5. **Run the script**
```bash
python Video_Game_Clustering.py
```
This will:
- Pull data from RAWG and SteamSpy
- Merge and clean the dataset
- Perform clustering
- Show you:
  - An Elbow Plot (to confirm cluster count)
  - A PCA Scatterplot of clusters
  - Print summaries of cluster characteristics
  - Save `clustered_video_games.csv` locally

---

## 📊 What the Clusters Tell You
Each cluster groups games with similar engagement patterns.
You’ll learn things like:
- Which genres dominate high-playtime games?
- Are strategy games reviewed better than shooters?
- What makes a game both critically acclaimed *and* widely played?

You’ll also get actual sample games per cluster so it’s not just numbers—it’s recognizable titles.

---

## 🚀 Sample Output
You’ll see output like this:
```text
Cluster 2: ['Counter-Strike: Global Offensive']
Top Genre: Shooter
Avg Playtime: 30,245 mins
Total Reviews: 7.5M
Metascore: 81
```

---

## 💡 Why This Matters
This clustering approach gives devs, publishers, and gamers an edge:
- **Game developers** can target features that drive long-term play
- **Marketers** can tailor promotions by genre engagement trends
- **Players** can discover similar games based on their playstyle

---

## 📍 Author
Built by Uche Onyejiaka as part of a data science course at the University of Maryland 🎓

---

## 🔗 License
This project is open-source. Feel free to remix and build on it. Just give credit where it’s due ✨

---

Let me know if you end up using this! I’d love to see how you apply it or what datasets you plug in next.

Stay analytical,
Uche 🤝

