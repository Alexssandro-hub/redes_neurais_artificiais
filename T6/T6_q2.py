import numpy as np

# Função de ativação tangente hiperbólica (tanh)
def tanh(x):
    return np.tanh(x)

# Definir a arquitetura da MLP
input_size = 3
hidden_size = 4
output_size = 1

# Inicializar os pesos aleatoriamente
np.random.seed(0)
weights_input_hidden = np.random.randn(input_size, hidden_size)
weights_hidden_output = np.random.randn(hidden_size, output_size)

# Definir as entradas
inputs = np.array([
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1]
])

# Calcular as saídas da camada oculta
hidden_inputs = np.dot(inputs, weights_input_hidden)
hidden_outputs = tanh(hidden_inputs)

# Calcular as saídas da camada de saída
output_inputs = np.dot(hidden_outputs, weights_hidden_output)
output_outputs = tanh(output_inputs)

# Exibir as saídas da porta NOR
print("Saídas da porta NOR")
for i in range(len(inputs)):
    input_data = inputs[i]
    output_data = output_outputs[i][0]
    print(f"Entradas {input_data} - Saída {round(output_data)}")