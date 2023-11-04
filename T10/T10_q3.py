import numpy as np

entradas = np.array([[0, 0],
                    [0, 1],
                    [1, 0],
                    [1, 1]])

saidas = np.array([[0],
                   [1],
                   [1],
                   [0]])


numero_neuronios_oculta = 5


pesos_entrada_oculta = np.random.randn(entradas.shape[1], numero_neuronios_oculta)


saida_camada_oculta = np.dot(entradas, pesos_entrada_oculta)


def funcao_ativacao(x):
    return np.tanh(x)

saida_camada_oculta_ativada = funcao_ativacao(saida_camada_oculta)


pesos_oculta_saida = np.dot(np.linalg.pinv(saida_camada_oculta_ativada), saidas)


saida = np.dot(saida_camada_oculta_ativada, pesos_oculta_saida)


print("Previsões da ELM para a porta XOR:")
print(np.round(saida))  # Arredondando para obter saídas binárias (0 ou 1)
