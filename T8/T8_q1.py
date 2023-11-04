import numpy as np

# Função de ativação sigmoide
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivada da função de ativação sigmoide
def sigmoid_derivative(x):
    return x * (1 - x)

# Dados de treinamento
X = np.array([[0, 0, 0],
              [0, 0, 1],
              [0, 1, 0],
              [0, 1, 1],
              [1, 0, 0],
              [1, 0, 1],
              [1, 1, 0],
              [1, 1, 1]])

# Saídas correspondentes da porta NAND
y = np.array([[1],
              [1],
              [1],
              [1],
              [1],
              [1],
              [1],
              [0]])

# Inicialização dos pesos da camada oculta e da camada de saída
input_size = 3
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
    hidden_output = sigmoid(hidden_input)
    output_input = np.dot(hidden_output, weights_hidden_output)
    output = sigmoid(output_input)

    # Cálculo do erro
    error = y - output

    # Backpropagation
    d_output = error * sigmoid_derivative(output)
    error_hidden = d_output.dot(weights_hidden_output.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_output)

    # Atualização dos pesos
    weights_hidden_output += hidden_output.T.dot(d_output) * learning_rate
    weights_input_hidden += X.T.dot(d_hidden) * learning_rate

# Teste da rede neural treinada
test_input = np.array([[0, 0, 1],
                       [1, 1, 0],
                       [1, 0, 1]])

hidden_input = np.dot(test_input, weights_input_hidden)
hidden_output = sigmoid(hidden_input)
output_input = np.dot(hidden_output, weights_hidden_output)
output = sigmoid(output_input)

print("Saídas previstas:")
print(output)
