#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# Load dataset
file_path = r"C:\Users\ACER\marketing_campaign.csv"
df = pd.read_csv(file_path, sep="\t")  # Adjust delimiter if needed

# Display dataset info
df.info()
df.head()


# In[2]:


# Check missing values
print(df.isnull().sum())

# Fill missing values with median (recommended for numerical data)
df.fillna(df.median(), inplace=True)


# In[3]:


features = ['Income', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts', 
            'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases', 
            'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth']

df_selected = df[features].copy()


# In[4]:


import numpy as np

# Define function to remove outliers using the IQR method
def remove_outliers(df, cols):
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

# Remove outliers
df_selected = remove_outliers(df_selected, features)


# In[5]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_selected)

# Convert back to DataFrame
df_scaled = pd.DataFrame(df_scaled, columns=df_selected.columns)


# In[6]:


df_scaled.describe()


# In[7]:


from sklearn.cluster import Birch

# Apply BIRCH clustering
birch = Birch(n_clusters=3)  # 3 clusters (can be tuned)
df_scaled["BIRCH_Cluster"] = birch.fit_predict(df_scaled)

# Check cluster distribution
df_scaled["BIRCH_Cluster"].value_counts()


# In[8]:


from sklearn.mixture import GaussianMixture

# Apply GMM clustering
gmm = GaussianMixture(n_components=3, random_state=42)
df_scaled["GMM_Cluster"] = gmm.fit_predict(df_scaled)

# Check cluster distribution
df_scaled["GMM_Cluster"].value_counts()


# In[9]:


from sklearn.decomposition import PCA

# Reduce to 2 principal components
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled.drop(columns=["BIRCH_Cluster", "GMM_Cluster"]))

# Add PCA results to DataFrame
df_scaled["PCA_1"] = df_pca[:, 0]
df_scaled["PCA_2"] = df_pca[:, 1]


# In[10]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 5))

# BIRCH Clustering Plot
plt.subplot(1, 2, 1)
sns.scatterplot(x=df_scaled["PCA_1"], y=df_scaled["PCA_2"], hue=df_scaled["BIRCH_Cluster"], palette="viridis")
plt.title("BIRCH Clustering - Customer Retention")
plt.xlabel("PCA Feature 1")
plt.ylabel("PCA Feature 2")

# GMM Clustering Plot
plt.subplot(1, 2, 2)
sns.scatterplot(x=df_scaled["PCA_1"], y=df_scaled["PCA_2"], hue=df_scaled["GMM_Cluster"], palette="coolwarm")
plt.title("GMM Clustering - Customer Retention")
plt.xlabel("PCA Feature 1")
plt.ylabel("PCA Feature 2")

plt.tight_layout()
plt.show()


# In[11]:


df_scaled.groupby("BIRCH_Cluster").mean()


# In[ ]:




