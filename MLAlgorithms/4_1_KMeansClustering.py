from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

X, _ = make_blobs(n_samples=300, centers=4, cluster_std=1.4, random_state=42)
# sapmles=örnek sayısı,centers:küme sayısı,cluster_Std  verilerin standert sapması

plt.figure()
plt.scatter(X[:, 0], X[:, 1])
plt.title("Ornek Veri")

kmeans = KMeans(n_clusters=4)# 4 tane küme bul
kmeans.fit(X)

labels = kmeans.labels_ # küme etiketleme

plt.figure()
plt.scatter(X[:, 0], X[:, 1], c = labels, cmap = "viridis")

centers = kmeans.cluster_centers_ # kümenlerin merkezi
plt.scatter(centers[:, 0], centers[:, 1], c = "red", marker = "X")
plt.title("K-Means")