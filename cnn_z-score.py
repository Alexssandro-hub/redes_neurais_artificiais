import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Importa o conjunto de dados
from ucimlrepo import fetch_ucirepo
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)
X = breast_cancer_wisconsin_diagnostic.data.features
y = breast_cancer_wisconsin_diagnostic.data.targets

# Normalização Z-score
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Divisão em conjuntos de treino e teste (70% treino, 30% teste)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Adição de bias aos dados
X_train_bias = np.hstack((X_train, np.ones((X_train.shape[0], 1))))
X_test_bias = np.hstack((X_test, np.ones((X_test.shape[0], 1))))

# Geração de pesos aleatórios para a camada de entrada oculta
input_size = X_train_bias.shape[1]
hidden_size = 100  # ajuste conforme necessário
input_weights = np.random.randn(input_size, hidden_size)

# Ativação da camada de entrada oculta usando a função sigmoid
hidden_activations = 1 / (1 + np.exp(-np.dot(X_train_bias, input_weights)))

# Cálculo dos pesos da camada de saída usando a pseudoinversa
output_weights = np.dot(np.linalg.pinv(hidden_activations), y_train)

# Ativação da camada de entrada oculta nos dados de teste
hidden_activations_test = 1 / (1 + np.exp(-np.dot(X_test_bias, input_weights)))

# Predição nos dados de teste
y_pred = np.dot(hidden_activations_test, output_weights)

# Conversão para rótulos binários (0 ou 1)
y_pred_binary = (y_pred > 0.5).astype(int)

# Avaliação do desempenho
accuracy = accuracy_score(y_test, y_pred_binary)
print(f'Acurácia no conjunto de teste: {accuracy}')
