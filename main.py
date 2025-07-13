import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# --- Configuration ---
DATA_PATH = "data/project3_bits.csv"
OUTPUT_DIR = "output"
N_CLUSTERS = 4
RANDOM_STATE = 42

# --- Create Directories ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("data", exist_ok=True)

# --- Load Dataset ---
try:
    df = pd.read_csv(DATA_PATH)
    print(f"[INFO] Loaded dataset with shape: {df.shape}")
except FileNotFoundError:
    print(f"[ERROR] File not found at {DATA_PATH}")
    exit()

if 'User ID' in df.columns:
    df.drop('User ID', axis=1, inplace=True)

# --- Process 'Top Interests' ---
df['Top Interests'] = df['Top Interests'].fillna('')
df['Top Interests'] = df['Top Interests'].apply(lambda x: [i.strip() for i in x.split(',') if i.strip()])
mlb = MultiLabelBinarizer()
interests_encoded = pd.DataFrame(mlb.fit_transform(df['Top Interests']), columns=mlb.classes_, index=df.index)
df = df.drop('Top Interests', axis=1)
df = pd.concat([df, interests_encoded], axis=1)

# --- One-hot encode other categorical variables ---
categorical_cols = df.select_dtypes(include='object').columns
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# --- Feature Scaling ---
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_data, columns=df.columns)

# --- KMeans Clustering ---
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
df['Segment'] = kmeans.fit_predict(scaled_data)

# --- PCA Visualization ---
pca = PCA(n_components=2, random_state=RANDOM_STATE)
pca_data = pca.fit_transform(scaled_data)

plt.figure(figsize=(14, 10))
palette = sns.color_palette("tab20", N_CLUSTERS)
sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1], hue=df['Segment'], palette=palette, s=120, edgecolor='white', linewidth=1.5)
plt.title("User Segments via PCA - Enhanced Visualization", fontsize=22, weight='bold', color='darkblue')
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)", fontsize=14, color='navy')
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)", fontsize=14, color='navy')
plt.grid(True, linestyle='--', alpha=0.4)
plt.gca().set_facecolor('#f9f9f9')
plt.legend(title="Segment ID", title_fontsize=13, fontsize=11, loc='best', frameon=True, fancybox=True, framealpha=0.8)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "PCA_Enhanced_Segments.png"), dpi=300)
plt.close()

# --- Segment Count Barplot ---
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Segment', palette='cubehelix', edgecolor='black', linewidth=2)
plt.title("Segment Counts", fontsize=20, weight='bold', color='darkgreen')
plt.xlabel("Segment", fontsize=14)
plt.ylabel("Number of Users", fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Segment_Counts_Enhanced.png"), dpi=300)
plt.close()

# --- Save Data and Models ---
df.to_csv(os.path.join(OUTPUT_DIR, "segmented_profiles.csv"), index=False)
joblib.dump(kmeans, os.path.join(OUTPUT_DIR, "kmeans_model.pkl"))
joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))
joblib.dump(mlb, os.path.join(OUTPUT_DIR, "mlb_encoder.pkl"))

# --- Classification Models ---
X = scaled_df
y = df['Segment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

# Logistic Regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)
print("[LOGISTIC] Accuracy:", accuracy_score(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))
joblib.dump(log_model, os.path.join(OUTPUT_DIR, "logistic_model.pkl"))

# SVM
svm_model = SVC()
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
print("[SVM] Accuracy:", accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))
joblib.dump(svm_model, os.path.join(OUTPUT_DIR, "svm_model.pkl"))

# --- Boxplots for Top Varying Features ---
segment_summary = df.groupby('Segment').mean()
std_devs = segment_summary.std().sort_values(ascending=False)
top_features = std_devs.head(5).index.tolist()

for feature in top_features:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Segment', y=feature, palette='Set2', linewidth=2, fliersize=4, boxprops=dict(alpha=0.7))
    sns.stripplot(data=df, x='Segment', y=feature, color='black', size=4, alpha=0.3, jitter=0.25)
    plt.title(f"Distribution of '{feature}' by Segment", fontsize=16, weight='bold', color='purple')
    plt.grid(True, axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"boxplot_{feature}.png"), dpi=300)
    plt.close()

print("\nAll steps completed. Outputs saved in the 'output/' folder with enhanced visuals.")
