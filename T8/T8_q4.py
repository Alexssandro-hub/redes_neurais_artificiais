import numpy as np

# Função de ativação tangente hiperbólica (tanh)
def tanh(x):
    return np.tanh(x)

# Derivada da função de ativação tangente hiperbólica (tanh)
def tanh_derivative(x):
    return 1 - np.tanh(x)**2

# Dados de entrada e saída
X = np.array([[0],
              [1],
              [2],
              [3],
              [4],
              [5],
              [6],
              [7],
              [8],
              [9]])

y = np.array([[9],
              [10],
              [19],
              [21],
              [22],
              [25],
              [39],
              [40],
              [41],
              [47]])

# Normalização dos dados
X = X / np.max(X)
y = y / np.max(y)

# Inicialização dos pesos da camada oculta e da camada de saída
input_size = 1
hidden_size = 4
output_size = 1

np.random.seed(42)
weights_input_hidden = np.random.uniform(size=(input_size, hidden_size))
weights_hidden_output = np.random.uniform(size=(hidden_size, output_size))

# Taxa de aprendizado
learning_rate = 0.1

# Número de épocas de treinamento
epochs = 10000

# Treinamento da rede neural
for epoch in range(epochs):
    # Forward propagation
    hidden_input = np.dot(X, weights_input_hidden)
    hidden_output = tanh(hidden_input)
    output_input = np.dot(hidden_output, weights_hidden_output)
    output = tanh(output_input)

    # Cálculo do erro
    error = y - output

    # Backpropagation
    d_output = error * tanh_derivative(output)
    error_hidden = d_output.dot(weights_hidden_output.T)
    d_hidden = error_hidden * tanh_derivative(hidden_output)

    # Atualização dos pesos
    weights_hidden_output += hidden_output.T.dot(d_output) * learning_rate
    weights_input_hidden += X.T.dot(d_hidden) * learning_rate

# Desnormalização dos resultados
predicted_output = output * np.max(y)

print("Saídas previstas:")
print(predicted_output)
