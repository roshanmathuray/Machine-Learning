#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# Load dataset from local path (use raw string)
file_path = r"C:\Users\ACER\marketing_campaign.csv"
df = pd.read_csv(file_path, sep="\t")  # Adjust delimiter if needed

# Display first few rows
df.head()


# In[2]:


identity_loyalty_features = [
    "Income", "Recency", "MntWines", "MntFruits", "MntMeatProducts", 
    "MntFishProducts", "MntSweetProducts", "MntGoldProds", "NumWebPurchases", 
    "NumStorePurchases", "NumCatalogPurchases", "NumWebVisitsMonth", "NumDealsPurchases"
]

df_identity_loyalty = df[identity_loyalty_features].copy()


# In[3]:


from sklearn.preprocessing import StandardScaler

# Fill missing values with the median of each column
df_identity_loyalty.fillna(df_identity_loyalty.median(), inplace=True)

# Standardize the data for clustering
scaler = StandardScaler()
df_identity_loyalty_scaled = scaler.fit_transform(df_identity_loyalty)

# Convert back to DataFrame
df_identity_loyalty_scaled = pd.DataFrame(df_identity_loyalty_scaled, columns=df_identity_loyalty.columns)

# Check the first few rows
df_identity_loyalty_scaled.head()


# In[4]:


from sklearn.cluster import Birch

# Apply BIRCH clustering (3 clusters)
birch = Birch(n_clusters=3)
df_identity_loyalty_scaled["BIRCH_Cluster"] = birch.fit_predict(df_identity_loyalty_scaled)

# Check the distribution of clusters
df_identity_loyalty_scaled["BIRCH_Cluster"].value_counts()


# In[5]:


from sklearn.mixture import GaussianMixture

# Apply GMM clustering (3 clusters)
gmm = GaussianMixture(n_components=3, random_state=42)
df_identity_loyalty_scaled["GMM_Cluster"] = gmm.fit_predict(df_identity_loyalty_scaled)

# Check the distribution of clusters
df_identity_loyalty_scaled["GMM_Cluster"].value_counts()


# In[6]:


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Reduce to 2 dimensions for visualization
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_identity_loyalty_scaled.drop(columns=["BIRCH_Cluster", "GMM_Cluster"]))

# Convert to DataFrame
df_identity_loyalty_scaled["PCA_1"] = df_pca[:, 0]
df_identity_loyalty_scaled["PCA_2"] = df_pca[:, 1]

# Plot BIRCH clusters
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.scatterplot(x=df_identity_loyalty_scaled["PCA_1"], y=df_identity_loyalty_scaled["PCA_2"], 
                hue=df_identity_loyalty_scaled["BIRCH_Cluster"], palette="viridis")
plt.title("BIRCH Clustering - Customer Identity & Brand Loyalty")
plt.xlabel("PCA Feature 1")
plt.ylabel("PCA Feature 2")

# Plot GMM clusters
plt.subplot(1, 2, 2)
sns.scatterplot(x=df_identity_loyalty_scaled["PCA_1"], y=df_identity_loyalty_scaled["PCA_2"], 
                hue=df_identity_loyalty_scaled["GMM_Cluster"], palette="coolwarm")
plt.title("GMM Clustering - Customer Identity & Brand Loyalty")
plt.xlabel("PCA Feature 1")
plt.ylabel("PCA Feature 2")

plt.tight_layout()
plt.show()


# In[7]:


# Group data by BIRCH clusters
df_identity_loyalty_scaled.groupby("BIRCH_Cluster").mean()


# In[ ]:




