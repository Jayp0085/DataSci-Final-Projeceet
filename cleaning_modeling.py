import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Load Data ---
try:
    prices_df = pd.read_csv("2024_panini_nba_hoops_prices.csv")
    stats_df = pd.read_csv("espn_fantasy_all_projections.csv")
except FileNotFoundError:
    print("Make sure the CSV files are in the same directory as the script.")
    exit()

# --- 2. Preprocess Price Data ---
print("Preprocessing Price Data...")

# Make a copy to avoid erronious manipulation
prices_df = prices_df.copy()

# Helper to clean price strings
def clean_price(price):
    if pd.isna(price) or price == '':
        return np.nan
    price_str = str(price).replace('$', '').replace(',', '')
    try:
        return float(price_str)
    except ValueError:
        return np.nan

for col in ['Ungraded', 'Grade 9', 'PSA 10']:
    prices_df[col] = prices_df[col].apply(clean_price)

# Extract Player Name (might need improvement)
# Player name is usually before a '#', '[', or a common variant keyword
def extract_player_name_from_card(card_description):
    if pd.isna(card_description):
        return None
    # Remove content in brackets first to avoid premature splitting
    name = re.sub(r"\[.*?\]", "", card_description).strip()
    # Split by common delimiters for card numbers or variations
    name_parts = re.split(r"\s*#\d+\s*|\s*/\d+\s*", name)
    name = name_parts[0].strip()
    
    # common variations that might be appended without brackets
    variations_to_remove = [
        "Premium", "Silver", "Gold", "Red Explosion", "Lazer Blue Prizm",
        "Silver Prizm", "Gold Vinyl Prizm", "Artist Proof", "Nebula Prizm",
        "Black Artist Proof" # Add more if I didn't cover all of them already
    ]
    for var in variations_to_remove:
        if name.endswith(var): # Check if it ends with a variation
            # Attempt to remove variation if it's clearly separated
            name = name[:-len(var)].strip() # Simple removal from end
            break # Only remove one variation
    
    # Filter out non-player items
    if name.lower() in ["blaster box", "hobby box"]:
        return None
    return name

prices_df['Player_Extracted'] = prices_df['Card'].apply(extract_player_name_from_card)

# Extract RC (Rookie Card)
prices_df['Is_RC'] = prices_df['Card'].str.contains(r'\[RC\]', na=False).astype(int)

# Extract Serial Numbering and Print Run
def extract_print_run(card_description):
    if pd.isna(card_description):
        return np.nan
    match = re.search(r'/(\d+)', card_description)
    return int(match.group(1)) if match else np.nan

prices_df['Print_Run'] = prices_df['Card'].apply(extract_print_run)
prices_df['Is_Serial_Numbered'] = prices_df['Print_Run'].notna().astype(int)

# Extract Card Variant (might need improvement)
def extract_variant(card_description):
    if pd.isna(card_description):
        return "Base"
    # Remove player name and card number part to isolate variant info
    cleaned_desc = re.sub(r"\[.*?\]", "", card_description).strip() # Remove RCs, etc.
    cleaned_desc = re.sub(r"^[^#]+#\d+\s*", "", cleaned_desc).strip() # Remove "Player Name #Number"
    cleaned_desc = re.sub(r"\s*/\d+", "", cleaned_desc).strip() # Remove /print run
    
    if not cleaned_desc: # If nothing left, it's either a base card or name extraction removed more than needed
        return "Base"
    
    # Feature Categorizing
    if "Gold Vinyl Prizm Premium" in card_description: return "Gold Vinyl Prizm Premium"
    if "Gold Artist Proof" in card_description: return "Gold Artist Proof"
    if "Premium Gold Prizm" in card_description: return "Premium Gold Prizm"
    if "Red Explosion" in card_description: return "Red Explosion"
    if "Lazer Blue Prizm Premium" in card_description: return "Lazer Blue Prizm Premium"
    if "Nebula Prizm Premium" in card_description: return "Nebula Prizm Premium"
    if "Black Artist Proof" in card_description: return "Black Artist Proof"
    if "Artist Proof" in card_description: return "Artist Proof" # General Artist Proof
    if "Silver Prizm Premium" in card_description: return "Silver Prizm Premium"
    if "Silver" in card_description and "/199" in card_description : return "Silver /199" # Stephon Castle has this
    if "Premium" in card_description : return "Premium Base" # e.g. Bronny James Jr. [Premium] #280
    
    # Simple check for other keywords
    # Probably needs to be added to
    keywords = ["Gold", "Silver", "Prizm", "Artist Proof", "Explosion", "Vinyl", "Nebula"]
    found_keywords = [kw for kw in keywords if kw.lower() in cleaned_desc.lower()]
    
    if found_keywords:
        return " ".join(found_keywords) # Combine found keywords
    elif "[RC]" in card_description and not cleaned_desc: # If it's an RC and no other variant info
        return "Base RC"
    return "Base" # Default if no specific variant identified

prices_df['Variant'] = prices_df['Card'].apply(extract_variant)

# Drop rows where player name couldn't be extracted or are non-player items (might need to configure previous functions to make sure all names can be extracted)
prices_df.dropna(subset=['Player_Extracted'], inplace=True)
print(f"Price data shape after initial processing: {prices_df.shape}")
# print("\nSample of processed price data:")
# print(prices_df[['Card', 'Player_Extracted', 'Is_RC', 'Print_Run', 'Variant', 'Ungraded', 'PSA 10']].head())

# --- 3. Preprocess Stats Data ---
print("\nPreprocessing Stats Data...")
stats_df = stats_df.copy()

# Clean Player Name in stats_df (remove team/position, e.g., "OSAC", "DenC" -- think it is not perfect, needs fixing)
# Also handles names like "Anthony DavisDalC, PF"
def clean_stats_player_name(name_field):
    if pd.isna(name_field):
        return None
    # Remove potential O, DTD, FA suffixes and team codes like "LALPG"
    name = re.sub(r"([A-Z]{2,5}|[O|DTD|FA]+)([A-Z]{1,2}(?:,\s*[A-Z]{1,2})*)?$", "", str(name_field)).strip()
    # For names like "Victor WembanyamaOSAC"
    name = re.sub(r"([A-Z]{3,})$", "", name).strip() # Remove trailing all-caps team/status if not caught
    return name

stats_df['Player_Cleaned'] = stats_df['Player'].apply(clean_stats_player_name)

# The stats CSV has many duplicate player entries (e.g. Rank 1, 51, 101 are all Wembanyama)
# Only keep first occurance (most recent)
stats_df.drop_duplicates(subset=['Player_Cleaned'], keep='first', inplace=True)

# Select relevant stat columns and ensure they are numeric
stat_cols_to_use = ['MIN', 'FG%', 'FT%', '3PM', 'REB', 'AST', 'STL', 'BLK', 'PTS']
for col in stat_cols_to_use:
    stats_df[col] = pd.to_numeric(stats_df[col], errors='coerce')

print(f"Stats data shape after processing: {stats_df.shape}")
# print("\nSample of processed stats data:")
# print(stats_df[['Player', 'Player_Cleaned'] + stat_cols_to_use].head())

# --- 4. Merge Data ---
print("\nMerging Data...")
# Merge on the cleaned/extracted player names
merged_df = pd.merge(prices_df, stats_df[['Player_Cleaned'] + stat_cols_to_use],
                     left_on='Player_Extracted', right_on='Player_Cleaned', how='left')

# Drop the redundant player name column from stats_df
merged_df.drop(columns=['Player_Cleaned'], inplace=True)

print(f"Merged data shape: {merged_df.shape}")
# print("\nSample of merged data (showing some NaNs for non-matches):")
# print(merged_df[merged_df['Player_Extracted'] == 'Victor Wembanyama'][['Card', 'Player_Extracted', 'PTS', 'Ungraded']].head())
# print(merged_df[merged_df['Player_Extracted'] == 'Bronny James Jr.'][['Card', 'Player_Extracted', 'PTS', 'Ungraded']].head()) # Likely NaN for PTS

# --- 5. Feature Engineering (Post-Merge) ---
print("\nFeature Engineering...")

# Target variable: Predicting 'PSA 10' price
# drop rows where this target is missing
TARGET_PRICE = 'PSA 10' 
# TARGET_PRICE = 'Ungraded' # Target column easily interchangeable for Ungraded/PSA 9

# Fill NaNs in Print_Run with a high value if Is_Serial_Numbered is False
# Placeholder if it's True but missing (tries to cover for where extraction messes up)
merged_df['Print_Run'].fillna(value=5000, inplace=True) # Arbitrary high for non-serial, or median of serials.

# One-hot encode 'Variant'
merged_df = pd.get_dummies(merged_df, columns=['Variant'], prefix='Var', dummy_na=False)

# For player stats, if a player wasn't in stats_df (e.g., some rookies), their stats will be NaN.
# Fill some left out rookie stats with 0 (stats not recorded yet).
for col in stat_cols_to_use:
    merged_df[col].fillna(0, inplace=True)

# Select features for the model
# Numerical features from stats + our engineered card features
features = stat_cols_to_use + ['Is_RC', 'Print_Run', 'Is_Serial_Numbered']
# Add one-hot encoded variant columns
features += [col for col in merged_df.columns if col.startswith('Var_')]

# Ensure all feature columns are present (some variants might only be in train or test)
# Drop rows where the target price is NaN
merged_df.dropna(subset=[TARGET_PRICE], inplace=True)
print(f"Shape after dropping NaNs in target '{TARGET_PRICE}': {merged_df.shape}")

if merged_df.empty:
    print(f"No data left after processing and filtering for target {TARGET_PRICE}. Check preprocessing steps.")
    exit()

X = merged_df[features]
y = merged_df[TARGET_PRICE]

# Log transform target variable if it's highly skewed (common for prices)
y_log = np.log1p(y) # log1p handles zeros

# --- 6. Model Selection & Training ---
print("\nModel Training...")
X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.4, random_state=42)

# Commented out data scaling, can uncomment to see how scaled data will perform
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
# Using non-scaled for Random Forest simplicity here
X_train_scaled = X_train
X_test_scaled = X_test


# Using RandomForestRegressor (If anyone can try to implement linear regression or other modeling methods we should do that)
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10, min_samples_leaf=2)
model.fit(X_train_scaled, y_train)

# --- 7. Evaluation ---
print("\nModel Evaluation...")
y_pred_log = model.predict(X_test_scaled)

# Inverse transform predictions to get actual prices
y_pred = np.expm1(y_pred_log)
y_test_actual = np.expm1(y_test) # y_test was already log-transformed

# Handle potential negative predictions if expm1 results in tiny negatives due to log(0) issues
y_pred[y_pred < 0] = 0

mae = mean_absolute_error(y_test_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
r2 = r2_score(y_test_actual, y_pred)

print(f"Target Variable: {TARGET_PRICE}")
print(f"Mean Absolute Error (MAE): ${mae:.2f}")
print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")
print(f"R-squared (R²): {r2:.4f}")

# Feature Importances
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({'feature': X.columns, 'importance': importances})
feature_importance_df = feature_importance_df.sort_values('importance', ascending=False).reset_index(drop=True)

print("\nTop Feature Importances:")
print(feature_importance_df.head(15))

# --- 8. Visualization of Predictions vs Actual ---
plt.figure(figsize=(10, 6))
plt.scatter(y_test_actual, y_pred, alpha=0.5)
plt.plot([y_test_actual.min(), y_test_actual.max()], [y_test_actual.min(), y_test_actual.max()], 'k--', lw=2) # Diagonal line
plt.xlabel(f"Actual {TARGET_PRICE} Price ($)")
plt.ylabel(f"Predicted {TARGET_PRICE} Price ($)")
plt.title(f"Actual vs. Predicted {TARGET_PRICE} Prices (Log-Transformed Target)")
plt.xscale('log') # log scale for better visualization since prices vary widely
plt.yscale('log')
plt.grid(True)
plt.show()

# --- Example: Show some predictions ---
predictions_df = X_test.copy()
predictions_df[f'Actual_{TARGET_PRICE}'] = y_test_actual
predictions_df[f'Predicted_{TARGET_PRICE}'] = y_pred
# Add back original card name for context
predictions_df = predictions_df.merge(merged_df[['Card'] + stat_cols_to_use], 
                                      left_index=True, right_index=True, suffixes=('', '_orig_stats'))

print("\nSample Predictions:")
print(predictions_df[['Card', f'Actual_{TARGET_PRICE}', f'Predicted_{TARGET_PRICE}'] + stat_cols_to_use])

# Below we can try to implement other modeling methods
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

# --- Linear Regression ---
print("\nTraining Linear Regression Model...")
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

y_pred_lr_log = lr_model.predict(X_test_scaled)
y_pred_lr = np.expm1(y_pred_lr_log)
y_pred_lr[y_pred_lr < 0] = 0  # no negatives 

mae_lr = mean_absolute_error(y_test_actual, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test_actual, y_pred_lr))
r2_lr = r2_score(y_test_actual, y_pred_lr)

print(f"\nLinear Regression Results:")
print(f"MAE: ${mae_lr:.2f}")
print(f"RMSE: ${rmse_lr:.2f}")
print(f"R²: {r2_lr:.4f}")

# plt.figure(figsize=(8, 6))
# plt.scatter(y_test_actual, y_pred_lr, alpha=0.6, color='royalblue', edgecolor='k')
# plt.plot([y_test_actual.min(), y_test_actual.max()],
#          [y_test_actual.min(), y_test_actual.max()], 'r--', linewidth=2)
# plt.xlabel("Actual PSA 10 Price ($)")
# plt.ylabel("Predicted PSA 10 Price ($)")
# plt.title("Linear Regression: Actual vs. Predicted Prices")
# plt.grid(True)
# plt.show()
# residuals = y_test_actual - y_pred_lr




