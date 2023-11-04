import numpy as np


entradas = np.array([[1, 1, 1],
                    [0, 1, 1],
                    [1, 0, 1],
                    [1, 1, 0],
                    [0, 0, 1],
                    [0, 1, 0],
                    [1, 0, 0],
                    [0, 0, 0]])

saidas = np.array([[1],
                   [1],
                   [1],
                   [1],
                   [1],
                   [1],
                   [1],
                   [0]])


numero_neuronios_oculta = 10


pesos_entrada_oculta = np.random.randn(entradas.shape[1], numero_neuronios_oculta)


saida_camada_oculta = np.dot(entradas, pesos_entrada_oculta)


def funcao_ativacao(x):
    return 1 / (1 + np.exp(-x))

saida_camada_oculta_ativada = funcao_ativacao(saida_camada_oculta)


pesos_oculta_saida = np.dot(np.linalg.pinv(saida_camada_oculta_ativada), saidas)


saida = np.dot(saida_camada_oculta_ativada, pesos_oculta_saida)


print("Previsões da ELM:")
print(saida)
