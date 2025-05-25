# Inteligência Artificial - Lista 10: RNA - Perceptron e Backpropagation

Este repositório contém a resolução da Lista de Exercícios #10 da disciplina de Inteligência Artificial, focada na implementação e análise de Redes Neurais Artificiais, especificamente os algoritmos Perceptron e Backpropagation.

## Conteúdo do Repositório

* **`lista10.ipynb`**: Notebook Jupyter contendo as implementações e experimentos.
* **`Relatorio_Lista_10_Simples.pdf`**: Versão compilada em PDF do relatório simplificado (para ser gerada a partir do `.tex`).

### Exercício 1: Perceptron
* Implementar o Perceptron para classificar as funções lógicas AND e OR com um número variável de entradas (`n`).
* Visualizar a separação de classes através da plotagem do hiperplano (para `n=2`) no notebook.
* Demonstrar e explicar por que o Perceptron não consegue resolver a função XOR.

### Exercício 2: Backpropagation
* Implementar uma rede neural MLP treinada com Backpropagation para classificar as funções lógicas AND, OR e XOR com `n` entradas.
* Investigar o impacto de diferentes taxas de aprendizado no treinamento da rede.
* Analisar a importância do termo de bias nos neurônios.
* Comparar o desempenho da rede utilizando diferentes funções de ativação (ex: Sigmoid, Tanh, ReLU).

## Estrutura do Código no Notebook (`lista10.ipynb`)

O notebook Jupyter está organizado em seções claras, separando os dois exercícios principais da lista. As principais bibliotecas utilizadas são `numpy` para operações numéricas e manipulação de arrays, `matplotlib` para a geração de gráficos, e `tensorflow.keras` para a implementação das redes neurais no Exercício 2.

### Parte Introdutória
* Células de markdown com o título, nome e uma introdução.
* Célula de código para importação das bibliotecas necessárias para ambos os exercícios.

### Exercício 1: Perceptron
Esta seção foca na implementação do algoritmo Perceptron "do zero" (usando apenas `numpy` para cálculos).
1.  **`gerar_dados_logicos(n_entradas, operacao)`**: Uma função utilitária para gerar todas as $2^n$ combinações de entradas booleanas para um dado número `n` de entradas e a operação lógica especificada ('AND', 'OR', 'XOR'). Retorna os arrays `X` (entradas) e `y` (saídas esperadas).
2.  **`class Perceptron`**:
    * `__init__(self, num_inputs, learning_rate=0.1)`: Construtor que inicializa os pesos sinápticos (incluindo o bias) aleatoriamente e armazena a taxa de aprendizado.
    * `predict(self, inputs)`: Recebe um conjunto de entradas, calcula a soma ponderada e aplica a função de ativação degrau para produzir uma saída (0 ou 1).
    * `train(self, training_inputs, labels, epochs=100)`: Implementa o algoritmo de treinamento do Perceptron. Itera sobre o conjunto de dados por um número especificado de épocas, ajustando os pesos com base no erro de predição. Armazena o histórico de erros e pesos para análise e plotagem.
3.  **Funções de Plotagem**:
    * `plotar_hiperplano(X, y, perceptron_model, title="", epoch_to_plot=-1)`: Para problemas com $n=2$ entradas, esta função plota os pontos de dados e o hiperplano de separação (reta) aprendido pelo Perceptron. Permite visualizar o estado do hiperplano em diferentes momentos do treinamento.
    * `plotar_erro(errors, title)`: Plota o número de erros de classificação por época, permitindo visualizar a curva de aprendizado.
4.  **`run_perceptron_experiment(...)`**: Uma função wrapper que orquestra a execução de um experimento completo para o Perceptron: gera os dados, treina o modelo, imprime os resultados (pesos finais, acurácia, predições) e chama as funções de plotagem.
5.  **Testes e Análise**: Células que utilizam `run_perceptron_experiment` para testar o Perceptron com as funções AND e OR para $n=2, 3, 5$, e para demonstrar sua incapacidade de resolver o XOR ($n=2, 3$). Inclui células de markdown com a análise dos resultados.

### Exercício 2: Backpropagation (MLP)
Esta seção utiliza a biblioteca `tensorflow.keras` para construir e treinar Redes Neurais de Múltiplas Camadas (MLP).
1.  **Reutilização de `gerar_dados_logicos`**: A mesma função do Exercício 1 é usada para gerar os dados para AND, OR e XOR com $n$ entradas.
2.  **`criar_modelo_mlp(...)`**: Função que define e compila um modelo MLP sequencial do Keras. Permite configurar o número de entradas, neurônios na camada oculta, funções de ativação (para camada oculta e de saída), taxa de aprendizado do otimizador (Adam), e a inclusão ou não de bias.
3.  **`plotar_historico_treinamento(history, title)`**: Plota as curvas de acurácia e perda (loss) do modelo ao longo das épocas de treinamento, tanto para o conjunto de treino quanto para o de validação (se aplicável).
4.  **`run_mlp_experiment(...)`**: Função wrapper similar à do Perceptron, mas para MLP. Ela:
    * Gera os dados.
    * Cria o modelo MLP usando `criar_modelo_mlp`.
    * Treina o modelo (`model.fit()`) e registra o tempo de treinamento.
    * Avalia o modelo treinado (`model.evaluate()`) e imprime a acurácia e perda finais.
    * Chama `plotar_historico_treinamento`.
    * Mostra predições para um subconjunto dos dados para verificação.
    * Esta função é central para as investigações, pois permite variar facilmente os hiperparâmetros.
5.  **Testes Base e Investigações**:
    * Células para testes base com AND, OR, XOR (para $n=2$) para garantir que a MLP funciona.
    * **Investigação 1 (Taxa de Aprendizado)**: Loop que chama `run_mlp_experiment` com diferentes taxas de aprendizado para a função XOR, plotando um gráfico comparativo das curvas de perda.
    * **Investigação 2 (Importância do Bias)**: Compara o treinamento de modelos XOR com e sem o termo de bias nas camadas, plotando um gráfico comparativo.
    * **Investigação 3 (Função de Ativação)**: Testa diferentes funções de ativação (sigmoid, tanh, relu) na camada oculta para o problema XOR, também com um gráfico comparativo.
    * **Testes com Diferentes $N$ Entradas**: Aplica a MLP para AND, OR e XOR com $n=3, 4$.
6.  **Análise e Conclusões**: Células de markdown discutindo os resultados de cada investigação e as conclusões gerais do Exercício 2.