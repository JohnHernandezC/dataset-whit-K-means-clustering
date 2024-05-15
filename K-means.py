# Paso 1: Importar bibliotecas necesarias y cargar los datos
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# Cargar el dataset
data = pd.read_csv('Mall_Customers.csv') # Asegúrate de poner la ruta correcta del archivo

# Mostrar las primeras filas del dataset
print(data.head())

# Obtener información sobre el dataset
print(data.info())

# Obtener estadísticas descriptivas
print(data.describe())

# Paso 2: Preprocesar los datos
# Verificar si hay valores faltantes
print(data.isnull().sum())


# Paso 3: Entrenar el modelo K-means
# Seleccionar las características relevantes para entrenar el modelo
X = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Estandarizar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Definir el número de clusters
k = 5

# Entrenar el modelo K-means
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X_scaled)

# Paso 4: Evaluar el desempeño del modelo
# Calcular el coeficiente de Silhouette
silhouette_avg = silhouette_score(X_scaled, kmeans.labels_)
print("Coeficiente de Silhouette:", silhouette_avg)

# Calcular el índice de Calinski-Harabasz
calinski_score = calinski_harabasz_score(X_scaled, kmeans.labels_)
print("Índice de Calinski-Harabasz:", calinski_score)

# Paso 5: Visualizar los resultados
# Visualizar los clusters
data['Cluster'] = kmeans.labels_
sns.pairplot(data, hue='Cluster', palette='Dark2')
plt.show()



