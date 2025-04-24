import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import Birch
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit Title
st.title("Customer Identity and Brand Loyalty Clustering")

# Load data from GitHub (Raw URL)
file_url = "https://raw.githubusercontent.com/roshanmathuray/Machine-Learning/main/marketing_campaign.csv"
df = pd.read_csv(file_url)

# Show dataset preview
st.write("### Dataset Preview:")
st.write(df.head())

# Select relevant features for clustering
identity_loyalty_features = [
    "Income", "Recency", "MntWines", "MntFruits", "MntMeatProducts", 
    "MntFishProducts", "MntSweetProducts", "MntGoldProds", "NumWebPurchases", 
    "NumStorePurchases", "NumCatalogPurchases", "NumWebVisitsMonth", "NumDealsPurchases"
]

# Prepare the data
df_identity_loyalty = df[identity_loyalty_features].copy()

# Fill missing values with the median of each column
df_identity_loyalty.fillna(df_identity_loyalty.median(), inplace=True)

# Standardize the data for clustering
scaler = StandardScaler()
df_identity_loyalty_scaled = scaler.fit_transform(df_identity_loyalty)

# Apply BIRCH clustering (3 clusters)
birch = Birch(n_clusters=3)
df_identity_loyalty_scaled = pd.DataFrame(df_identity_loyalty_scaled, columns=identity_loyalty_features)
df_identity_loyalty_scaled["BIRCH_Cluster"] = birch.fit_predict(df_identity_loyalty_scaled)

# Apply GMM clustering (3 clusters)
gmm = GaussianMixture(n_components=3, random_state=42)
df_identity_loyalty_scaled["GMM_Cluster"] = gmm.fit_predict(df_identity_loyalty_scaled)

# Apply PCA for 2D visualization
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_identity_loyalty_scaled.drop(columns=["BIRCH_Cluster", "GMM_Cluster"]))

# Add PCA features to the dataframe
df_identity_loyalty_scaled["PCA_1"] = df_pca[:, 0]
df_identity_loyalty_scaled["PCA_2"] = df_pca[:, 1]

# Display Clusters using Seaborn/Matplotlib
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot Birch clusters
sns.scatterplot(x="PCA_1", y="PCA_2", hue="BIRCH_Cluster", data=df_identity_loyalty_scaled, ax=ax1, palette="viridis")
ax1.set_title("BIRCH Clustering - Customer Identity & Brand Loyalty")

# Plot GMM clusters
sns.scatterplot(x="PCA_1", y="PCA_2", hue="GMM_Cluster", data=df_identity_loyalty_scaled, ax=ax2, palette="coolwarm")
ax2.set_title("GMM Clustering - Customer Identity & Brand Loyalty")

# Show the plot in Streamlit
st.pyplot(fig)

# Show cluster statistics (average values per cluster)
st.write("### Cluster Statistics (Birch Clustering):")
st.write(df_identity_loyalty_scaled.groupby("BIRCH_Cluster").mean())

# Show GMM cluster statistics
st.write("### Cluster Statistics (GMM Clustering):")
st.write(df_identity_loyalty_scaled.groupby("GMM_Cluster").mean())








