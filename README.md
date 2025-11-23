# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

# Step 1: Load the data
df = pd.read_csv('Mall_Customers.csv')

# Step 2: Encode categorical data if necessary (e.g., Gender)
if df['Gender'].dtype == 'object':
    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])

# Step 3: Select features for clustering (often 'Annual Income (k$)' and 'Spending Score (1-100)')
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Step 4: Determine the optimal clusters using the elbow method (optional, but shown here)
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss, marker='o')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Step 5: Fit the KMeans with the optimal number of clusters (e.g., 5)
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X)

# Step 6: Visualize the clusters
plt.figure(figsize=(8, 6))
colors = ['red', 'blue', 'green', 'cyan', 'magenta']
for i in range(5):
    plt.scatter(X.values[y_kmeans == i, 0], X.values[y_kmeans == i, 1], s=100,
                color=colors[i], label=f'Cluster {i+1}')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=300, c='yellow', label='Centroids', edgecolor='black')
plt.title('Customer Segmentation with KMeans Clustering')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

```

## Output:


<img width="906" height="570" alt="image" src="https://github.com/user-attachments/assets/394ec6ac-3842-4587-a18d-59ea1c050aa2" />

<img width="873" height="704" alt="image" src="https://github.com/user-attachments/assets/b4030fe6-f337-4e37-b60f-c94cf286908f" />


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
