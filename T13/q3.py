import numpy as np
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Função para criar dados de treinamento para a porta NOR
def create_nor_data():
    X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=42)
    y[y == 1] = 0  # Convertendo 1 para 0 para representar a porta NOR
    return X, y

# Função para treinar uma RNA RBF
def train_rbf_network(X_train, y_train, num_rbf_neurons=10):
    # Etapa 1: Encontrar os centros das funções de base radial usando K-Means
    kmeans = KMeans(n_clusters=num_rbf_neurons, random_state=42)
    kmeans.fit(X_train)
    rbf_centers = kmeans.cluster_centers_

    # Etapa 2: Calcular a largura do kernel (sigma) como a média das distâncias dos pontos aos centros
    sigma = np.mean([np.linalg.norm(X - center) for X, center in zip(X_train, rbf_centers)])

    # Etapa 3: Calcular as saídas das funções de base radial
    rbf_outputs = np.exp(-np.square(np.linalg.norm(X_train[:, np.newaxis] - rbf_centers, axis=2)) / (2 * sigma**2))

    # Etapa 4: Treinar uma camada de saída linear usando as saídas das funções de base radial
    clf = MLPClassifier(hidden_layer_sizes=(num_rbf_neurons,), activation='identity', solver='lbfgs', random_state=42)
    clf.fit(rbf_outputs, y_train)

    return kmeans, sigma, clf

# Função para fazer previsões usando a RNA RBF treinada
def predict_rbf_network(X_test, kmeans, sigma, clf):
    # Etapa 1: Calcular as saídas das funções de base radial para os dados de teste
    rbf_outputs_test = np.exp(-np.square(np.linalg.norm(X_test[:, np.newaxis] - kmeans.cluster_centers_, axis=2)) / (2 * sigma**2))

    # Etapa 2: Fazer previsões usando a camada de saída linear
    predictions = clf.predict(rbf_outputs_test)

    return predictions

# Criar dados de treinamento
X, y = create_nor_data()

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar a RNA RBF
kmeans, sigma, clf = train_rbf_network(X_train, y_train)

# Fazer previsões nos dados de teste
predictions = predict_rbf_network(X_test, kmeans, sigma, clf)

# Calcular a acurácia do modelo
accuracy = accuracy_score(y_test, predictions)
print(f'Acurácia da RNA RBF para a porta NOR: {accuracy * 100:.2f}%')
