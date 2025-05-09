import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
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

for col in ['Ungraded','Grade 9','PSA 10']:
    prices_df[col] = prices_df[col].apply(clean_price)

def extract_player_name_from_card(desc):
    if pd.isna(desc): return None
    name = re.sub(r"\[.*?\]","",desc).strip()
    name = re.split(r"\s*#\d+|\s*/\d+", name)[0].strip()
    # drop box lines
    if name.lower() in ["blaster box","hobby box"]: return None
    # strip known suffixes
    for var in ["Premium","Silver","Gold","Red Explosion","Lazer Blue Prizm","Silver Prizm","Gold Vinyl Prizm","Artist Proof","Nebula Prizm","Black Artist Proof"]:
        if name.endswith(var):
            name = name[:-len(var)].strip()
            break
    return name

prices_df['Player_Extracted'] = prices_df['Card'].apply(extract_player_name_from_card)
prices_df['Is_RC'] = prices_df['Card'].str.contains(r'\[RC\]', na=False).astype(int)

def extract_print_run(desc):
    if pd.isna(desc): return np.nan
    m = re.search(r'/(\d+)', desc)
    return int(m.group(1)) if m else np.nan

prices_df['Print_Run'] = prices_df['Card'].apply(extract_print_run)
prices_df['Is_Serial_Numbered'] = prices_df['Print_Run'].notna().astype(int)

def extract_variant(desc):
    if pd.isna(desc): return "Base"
    s = re.sub(r"\[.*?\]","",desc)
    s = re.sub(r"^[^#]+#\d+\s*","",s)
    s = re.sub(r"\s*/\d+","",s).strip()
    if not s: return "Base"
    if "Gold Vinyl Prizm Premium" in desc: return "Gold Vinyl Prizm Premium"
    if "Gold Artist Proof" in desc: return "Gold Artist Proof"
    if "Red Explosion" in desc: return "Red Explosion"
    if "Lazer Blue Prizm Premium" in desc: return "Lazer Blue Prizm Premium"
    if "Nebula Prizm Premium" in desc: return "Nebula Prizm Premium"
    if "Black Artist Proof" in desc: return "Black Artist Proof"
    if "Silver Prizm Premium" in desc: return "Silver Prizm Premium"
    if "Silver" in desc and "/199" in desc: return "Silver /199"
    if "Premium" in desc: return "Premium Base"
    keywords = ["Gold","Silver","Prizm","Artist Proof","Explosion","Vinyl","Nebula"]
    found = [kw for kw in keywords if kw.lower() in s.lower()]
    return " ".join(found) if found else ("Base RC" if "[RC]" in desc else "Base")

prices_df['Variant'] = prices_df['Card'].apply(extract_variant)
prices_df.dropna(subset=['Player_Extracted'], inplace=True)

# --- 3. Preprocess Stats Data ---
def clean_stats_player_name(name):
    if pd.isna(name): return None
    s = re.sub(r"([A-Z]{2,5}|O|DTD|FA)+([A-Z]{1,2}(?:,\s*[A-Z]{1,2})*)?$","", str(name)).strip()
    return re.sub(r"([A-Z]{3,})$","", s).strip()

stats_df['Player_Cleaned'] = stats_df['Player'].apply(clean_stats_player_name)
stats_df.drop_duplicates(subset=['Player_Cleaned'], keep='first', inplace=True)

stat_cols = ['MIN','FG%','FT%','3PM','REB','AST','STL','BLK','PTS']
for c in stat_cols:
    stats_df[c] = pd.to_numeric(stats_df[c], errors='coerce')

# --- 4. Merge ---
merged_df = pd.merge(
    prices_df,
    stats_df[['Player_Cleaned'] + stat_cols],
    left_on='Player_Extracted', right_on='Player_Cleaned',
    how='left'
).drop(columns=['Player_Cleaned'])

# --- 5. Feature Engineering ---
merged_df['Print_Run'].fillna(5000, inplace=True)
merged_df = pd.get_dummies(merged_df, columns=['Variant'], prefix='Var')
for c in stat_cols:
    merged_df[c].fillna(0, inplace=True)

# Now define the features for clustering
feature_cols = stat_cols + ['Is_RC','Print_Run','Is_Serial_Numbered'] \
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

plt.figure(figsize=(8,4))
plt.plot(range(1,11), wcss, 'o-')
plt.xlabel('k')
plt.ylabel('WCSS')
plt.title('Elbow Method')
plt.show()

# --- 7. Fit & Inspect K-Means ---
k = 4  # choose based on your elbow plot
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
merged_df['Cluster'] = kmeans.fit_predict(X_scaled)

centers = scaler.inverse_transform(kmeans.cluster_centers_)
centers_df = pd.DataFrame(centers, columns=feature_cols)
print("Cluster centers (original units):")
print(centers_df)

# Optional: quick scatter of two stats
plt.figure(figsize=(8,6))
ix_pts = feature_cols.index('PTS')
ix_reb = feature_cols.index('REB')
plt.scatter(X_scaled[:,ix_pts], X_scaled[:,ix_reb],
            c=merged_df['Cluster'], cmap='tab10', alpha=0.6)
plt.xlabel('PTS (scaled)')
plt.ylabel('REB (scaled)')
plt.show()
