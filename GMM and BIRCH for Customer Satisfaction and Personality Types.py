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


features = ['Income', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts', 
            'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases', 
            'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth']

df_selected = df[features].copy()


# In[3]:


df_selected.fillna(df_selected.median(), inplace=True)


# In[4]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_selected)


# In[5]:


from sklearn.cluster import Birch

birch = Birch(n_clusters=3)
df_selected['BIRCH_Cluster'] = birch.fit_predict(df_scaled)


# In[6]:


from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=3, random_state=42)
df_selected['GMM_Cluster'] = gmm.fit_predict(df_scaled)


# In[7]:


from sklearn.decomposition import PCA

pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)

df_selected['PCA_1'] = df_pca[:, 0]
df_selected['PCA_2'] = df_pca[:, 1]


# In[8]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 5))

# BIRCH Clustering Plot
plt.subplot(1, 2, 1)
sns.scatterplot(x=df_selected["PCA_1"], y=df_selected["PCA_2"], hue=df_selected["BIRCH_Cluster"], palette="viridis")
plt.title("BIRCH Clustering")
plt.xlabel("PCA Feature 1")
plt.ylabel("PCA Feature 2")

# GMM Clustering Plot
plt.subplot(1, 2, 2)
sns.scatterplot(x=df_selected["PCA_1"], y=df_selected["PCA_2"], hue=df_selected["GMM_Cluster"], palette="coolwarm")
plt.title("GMM Clustering")
plt.xlabel("PCA Feature 1")
plt.ylabel("PCA Feature 2")

plt.tight_layout()
plt.show()


# In[9]:


df_selected.groupby("BIRCH_Cluster").mean()


# In[ ]:




