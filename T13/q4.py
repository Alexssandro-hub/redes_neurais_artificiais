import numpy as np
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Função para criar dados de treinamento para a porta XOR
def create_xor_data():
    X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=2, random_state=42)
    return X, y

# Função para treinar uma RNA RBF com duas camadas
def train_rbf_network(X_train, y_train, num_rbf_neurons=10):
    # Etapa 1: Encontrar os centros das funções de base radial usando K-Means
    kmeans = KMeans(n_clusters=num_rbf_neurons, random_state=42)
    kmeans.fit(X_train)
    rbf_centers = kmeans.cluster_centers_

    # Etapa 2: Calcular a largura do kernel (sigma) como a média das distâncias dos pontos aos centros
    sigma = np.mean([np.linalg.norm(X - center) for X, center in zip(X_train, rbf_centers)])

    # Etapa 3: Calcular as saídas das funções de base radial
    rbf_outputs = np.exp(-np.square(np.linalg.norm(X_train[:, np.newaxis] - rbf_centers, axis=2)) / (2 * sigma**2))

    # Etapa 4: Treinar uma camada oculta linear usando as saídas das funções de base radial
    hidden_layer = MLPClassifier(hidden_layer_sizes=(num_rbf_neurons,), activation='identity', solver='lbfgs', random_state=42)
    hidden_layer.fit(rbf_outputs, y_train)

    # Etapa 5: Treinar a camada de saída usando a camada oculta
    clf = MLPClassifier(hidden_layer_sizes=(num_rbf_neurons,), activation='logistic', solver='lbfgs', random_state=42)
    clf.fit(X_train, y_train)

    return kmeans, sigma, hidden_layer, clf

# Função para fazer previsões usando a RNA RBF treinada
def predict_rbf_network(X_test, kmeans, sigma, hidden_layer, clf):
    # Etapa 1: Calcular as saídas das funções de base radial para os dados de teste
    rbf_outputs_test = np.exp(-np.square(np.linalg.norm(X_test[:, np.newaxis] - kmeans.cluster_centers_, axis=2)) / (2 * sigma**2))

    # Etapa 2: Fazer previsões usando a camada oculta linear
    hidden_layer_predictions = hidden_layer.predict(rbf_outputs_test)

    # Etapa 3: Fazer previsões usando a camada de saída
    predictions = clf.predict(X_test)

    return predictions

# Criar dados de treinamento
X, y = create_xor_data()

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar a RNA RBF com duas camadas
kmeans, sigma, hidden_layer, clf = train_rbf_network(X_train, y_train)

# Fazer previsões nos dados de teste
predictions = predict_rbf_network(X_test, kmeans, sigma, hidden_layer, clf)

# Calcular a acurácia do modelo
accuracy = accuracy_score(y_test, predictions)
print(f'Acurácia da RNA RBF para a porta XOR: {accuracy * 100:.2f}%')
