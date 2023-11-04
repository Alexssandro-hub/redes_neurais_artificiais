import numpy as np

# Função de ativação sigmoide
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Função que implementa a MLP para a porta NAND
def mlp(input_data, weights):
    # Calcula a saída da camada oculta
    hidden_layer_input = np.dot(input_data, weights[0])
    hidden_layer_output = sigmoid(hidden_layer_input)
    
    # Calcula a saída da camada de saída
    output_layer_input = np.dot(hidden_layer_output, weights[1])
    output_layer_output = sigmoid(output_layer_input)
    
    return output_layer_output

# Pesos iniciais gerados aleatoriamente
np.random.seed(0)  # Define a semente para reprodução
input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Entradas da porta NAND
weights = [np.random.rand(2, 2), np.random.rand(2, 1)]  # Pesos aleatórios para camada oculta e camada de saída

# Testa a MLP para cada entrada
for i in range(len(input_data)):
    input_vector = input_data[i]
    output = mlp(input_vector, weights)
    print(f"Entrada: {input_vector}, Saída: {output}")

