# Defina os pesos e bias
pesos_camada_input = [[0.2, -0.1, 0.4], [0.7, -1.2, 1.2]]
bias_camada_input = [-1, -1]

pesos_camada_oculta = [[1.1, 0.1], [3.1, 1.17]]
bias_camada_oculta = [-1, -1]

# Função para calcular a saída de uma camada
def calcular_saida(camada_entrada, pesos, bias):
    soma_ponderada = sum([x * w for x, w in zip(camada_entrada, pesos)]) + bias
    return soma_ponderada

# Exemplo de entrada [10, 12, -9]
entrada = [10, 12, -9]

# Camada de entrada para camada oculta
saida_camada_oculta = [calcular_saida(entrada, pesos, bias) for pesos, bias in zip(pesos_camada_input, bias_camada_input)]

# Camada oculta para camada de saída
saida_camada_saida = [calcular_saida(saida_camada_oculta, pesos, bias) for pesos, bias in zip(pesos_camada_oculta, bias_camada_oculta)]

# Classificação com base nas saídas da camada de saída
if saida_camada_saida[0] >= saida_camada_saida[1]:
    classe_estimada = 1
else:
    classe_estimada = 2

print("Classe estimada para o primeiro exemplo de entrada [10, 12, -9]:", classe_estimada)

# Exemplo de entrada [-2, 3, 30]
entrada = [-2, 3, 30]

# Camada de entrada para camada oculta
saida_camada_oculta = [calcular_saida(entrada, pesos, bias) for pesos, bias in zip(pesos_camada_input, bias_camada_input)]

# Camada oculta para camada de saída
saida_camada_saida = [calcular_saida(saida_camada_oculta, pesos, bias) for pesos, bias in zip(pesos_camada_oculta, bias_camada_oculta)]

# Classificação com base nas saídas da camada de saída
if saida_camada_saida[0] >= saida_camada_saida[1]:
    classe_estimada = 1
else:
    classe_estimada = 2

print("Classe estimada para o segundo exemplo de entrada [-2, 3, 30]:", classe_estimada)
