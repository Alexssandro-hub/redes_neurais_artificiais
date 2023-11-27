import numpy as np
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

# Dados de entrada e saída
X_train = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(-1, 1)
y_train = np.array([9, 10, 19, 21, 22, 25, 39, 40, 41, 47])

# Função para treinar uma RNA RBF
def train_rbf_network(X_train, y_train, num_rbf_neurons=10):
    # Etapa 1: Encontrar os centros das funções de base radial usando K-Means
    kmeans = KMeans(n_clusters=num_rbf_neurons, random_state=42)
    kmeans.fit(X_train)
    rbf_centers = kmeans.cluster_centers_

    # Etapa 2: Calcular a largura do kernel (sigma) como a média das distâncias dos pontos aos centros
    sigma = np.mean([np.linalg.norm(X - center) for X, center in zip(X_train, rbf_centers)])

    # Etapa 3: Calcular as saídas das funções de base radial
    rbf_outputs = np.exp(-np.square(np.linalg.norm(X_train - rbf_centers, axis=1)) / (2 * sigma**2))

    # Etapa 4: Treinar uma camada de saída linear usando as saídas das funções de base radial
    reg = MLPRegressor(hidden_layer_sizes=(num_rbf_neurons,), activation='identity', solver='lbfgs', random_state=42)
    reg.fit(rbf_outputs.reshape(-1, 1), y_train)

    return kmeans, sigma, reg

# Função para fazer previsões usando a RNA RBF treinada
def predict_rbf_network(X_test, kmeans, sigma, reg):
    # Etapa 1: Calcular as saídas das funções de base radial para os dados de teste
    rbf_outputs_test = np.exp(-np.square(np.linalg.norm(X_test - kmeans.cluster_centers_, axis=1)) / (2 * sigma**2))

    # Etapa 2: Fazer previsões usando a camada de saída linear
    predictions = reg.predict(rbf_outputs_test.reshape(-1, 1))

    return predictions

# Criar dados de teste para visualização
X_test = np.linspace(0, 9, 100).reshape(-1, 1)

# Treinar a RNA RBF
kmeans, sigma, reg = train_rbf_network(X_train, y_train)

# Fazer previsões nos dados de teste
predictions = predict_rbf_network(X_test, kmeans, sigma, reg)

# Visualizar os resultados
plt.scatter(X_train, y_train, label='Dados de Treinamento')
plt.plot(X_test, predictions, label='Previsões da RNA RBF', color='red')
plt.legend()
plt.xlabel('Entradas')
plt.ylabel('Saídas')
plt.title('RNA RBF para Problema de Regressão')
plt.show()
