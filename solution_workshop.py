import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score
from sklearn.datasets import make_blobs


X, _ = make_blobs(n_samples=500, n_features=5, centers=4, random_state=42)


print("Taille des données :", X.shape)

kmeans_random = KMeans(n_clusters=4, init='random', random_state=42).fit(X)
kmeans_plus = KMeans(n_clusters=4, init='k-means++', random_state=42).fit(X)

score_random = calinski_harabasz_score(X, kmeans_random.labels_)
score_plus = calinski_harabasz_score(X, kmeans_plus.labels_)
print("Score Calinski-Harabasz (Random):", score_random)
print("Score Calinski-Harabasz (K-means++):", score_plus)

if score_plus > score_random:
    print("K-means++ donne de meilleurs résultats.")
else:
    print("L'initialisation aléatoire donne de meilleurs résultats.")

meilleur_modele = kmeans_plus if score_plus > score_random else kmeans_random
print("Meilleur modèle de clustering :", "K-means++" if score_plus > score_random else "Aléatoire")

print("Centres des clusters :", meilleur_modele.cluster_centers_)


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print("Nouvelle matrice des observations :", X_pca)

print("Valeurs propres :", pca.explained_variance_)
print("Vecteurs propres :", pca.components_)


print("Inertie de chaque axe :", pca.explained_variance_ratio_)


print("Somme des inerties :", np.sum(pca.explained_variance_ratio_))

centers_pca = pca.transform(meilleur_modele.cluster_centers_)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=meilleur_modele.labels_, cmap='viridis')
plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', marker='x')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Représentation des données et des centres')
plt.show()


print("Les clusters sont bien séparés sur les axes principaux.")
