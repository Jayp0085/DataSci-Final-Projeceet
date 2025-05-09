import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# --- 1. Load Data ---
prices_df = pd.read_csv("2024_panini_nba_hoops_prices.csv")
stats_df  = pd.read_csv("espn_fantasy_all_projections.csv")

# --- 2. Preprocess Price Data ---
def clean_price(p):
    if pd.isna(p) or p == '': return np.nan
    s = str(p).replace('$','').replace(',','')
    try: return float(s)
    except: return np.nan

for col in ['Ungraded', 'Grade 9', 'PSA 10']:
    prices_df[col] = prices_df[col].apply(clean_price)

def extract_player_name(desc):
    if pd.isna(desc): return None
    name = re.sub(r"\[.*?\]", "", desc).strip()
    name = re.split(r"\s*#\d+|\s*/\d+", name)[0].strip()
    if name.lower() in ["blaster box", "hobby box"]:
        return None
    for suf in ["Premium","Silver","Gold","Red Explosion","Lazer Blue Prizm",
                "Silver Prizm","Gold Vinyl Prizm","Artist Proof",
                "Nebula Prizm","Black Artist Proof"]:
        if name.endswith(suf):
            name = name[:-len(suf)].strip()
            break
    return name

prices_df['Player'] = prices_df['Card'].apply(extract_player_name)
prices_df['Is_RC'] = prices_df['Card'].str.contains(r'\[RC\]', na=False).astype(int)
prices_df['Print_Run'] = prices_df['Card'].str.extract(r'/(\d+)', expand=False).astype(float)
prices_df['Is_Serial'] = prices_df['Print_Run'].notna().astype(int)

def extract_variant(desc):
    if pd.isna(desc): return "Base"
    if "Gold Artist Proof" in desc: return "Gold Artist Proof"
    if "Premium Gold Prizm" in desc: return "Premium Gold Prizm"
    if "Gold Vinyl Prizm Premium" in desc: return "Gold Vinyl Prizm Premium"
    if "Red Explosion" in desc: return "Red Explosion"
    if "Lazer Blue Prizm Premium" in desc: return "Lazer Blue Prizm Premium"
    if "Nebula Prizm Premium" in desc: return "Nebula Prizm Premium"
    if "Artist Proof" in desc: return "Artist Proof"
    if "Silver Prizm Premium" in desc: return "Silver Prizm Premium"
    if "Silver" in desc and "/199" in desc: return "Silver /199"
    if "Premium" in desc: return "Premium"
    return "Base"

prices_df['Variant'] = prices_df['Card'].apply(extract_variant)
prices_df.dropna(subset=['Player'], inplace=True)

# --- 3. Preprocess Stats Data ---
def clean_stats_name(s):
    if pd.isna(s): return None
    name = re.sub(r"([A-Z]{2,5}|O|DTD|FA)+([A-Z]{1,2}(?:,\s*[A-Z]{1,2})*)?$", "", str(s)).strip()
    return re.sub(r"([A-Z]{3,})$", "", name).strip()

stats_df['Player_Clean'] = stats_df['Player'].apply(clean_stats_name)
stats_df.drop_duplicates(subset=['Player_Clean'], keep='first', inplace=True)

stat_cols = ['MIN','FG%','FT%','3PM','REB','AST','STL','BLK','PTS']
for c in stat_cols:
    stats_df[c] = pd.to_numeric(stats_df[c], errors='coerce')

# --- 4. Merge ---
merged_df = pd.merge(
    prices_df,
    stats_df[['Player_Clean'] + stat_cols],
    left_on='Player', right_on='Player_Clean',
    how='left'
).drop(columns=['Player_Clean'])

# --- 5. Feature Engineering ---
merged_df['Print_Run'].fillna(5000, inplace=True)
merged_df = pd.get_dummies(merged_df, columns=['Variant'], prefix='Var')
for c in stat_cols:
    merged_df[c].fillna(0, inplace=True)

feature_cols = stat_cols + ['Is_RC','Print_Run','Is_Serial'] \
               + [c for c in merged_df.columns if c.startswith('Var_')]
X = merged_df[feature_cols]

# --- 6. Standardize & Elbow Method ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

wcss = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    wcss.append(km.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(range(1, 11), wcss, 'o-', linewidth=2)
plt.xlabel('Number of clusters k')
plt.ylabel('Within-cluster sum of squares (WCSS)')
plt.title('Elbow Method for Optimal k')
plt.show()

# --- 7. Fit K-Means with k = 3 ---
k = 3
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
merged_df['Cluster'] = kmeans.fit_predict(X_scaled)

# --- 8. Inspect Cluster Centers ---
centers = scaler.inverse_transform(kmeans.cluster_centers_)
centers_df = pd.DataFrame(centers, columns=feature_cols)

# --- 9. Cluster Summary ---
cluster_summary = (
    merged_df
    .groupby('Cluster')
    .agg(
        count=('Cluster', 'size'),
        avg_price=('PSA 10', 'mean'),
        avg_pts=('PTS', 'mean'),
        avg_reb=('REB', 'mean'),
        avg_printrun=('Print_Run', 'mean'),
        pct_rookie=('Is_RC', 'mean')
    )
    .reset_index()
)

# --- 10. Validate Clustering ---
sil = silhouette_score(X_scaled, merged_df['Cluster'])

# --- 11. Prettify for Report (Markdown tables) ---
print("\n## Cluster Centers\n")
print(centers_df.to_markdown(index=False))

print("\n## Cluster Summary\n")
print(cluster_summary.to_markdown(index=False))

print(f"\nSilhouette score for k={k}: {sil:.3f}")
